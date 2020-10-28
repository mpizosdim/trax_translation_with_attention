from trax import layers as tl
from trax.supervised import training
from trax.fastmath import numpy as fastnp
import trax
import numpy as np
from process_utils import tokenize, detokenize


def input_encoder_fn(input_vocab_size, d_model, n_encoder_layers):
    input_encoder = tl.Serial(
        tl.Embedding(input_vocab_size, d_model),
        [tl.LSTM(d_model) for _ in range(n_encoder_layers)])
    return input_encoder


def pre_attention_decoder_fn(mode, target_vocab_size, d_model):
    pre_attention_decoder = tl.Serial(
        tl.ShiftRight(mode=mode),
        tl.Embedding(target_vocab_size, d_model),
        tl.LSTM(d_model))
    return pre_attention_decoder


def prepare_attention_input(encoder_activations, decoder_activations, inputs):
    keys = encoder_activations
    values = encoder_activations
    queries = decoder_activations

    mask = (inputs != 0) # generate the mask to distinguish real tokens from padding
    mask = fastnp.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1])) # add axes to the mask for attention heads and decoder length.
    mask = mask + fastnp.zeros((1, 1, decoder_activations.shape[1], 1)) # broadcast so mask shape is [batch size, attention heads, decoder-len, encoder-len].

    return queries, keys, values, mask


def NMTAttn(input_vocab_size=33300,
            target_vocab_size=33300,
            d_model=1024,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_attention_heads=4,
            attention_dropout=0.0,
            mode='train'):

    input_encoder = input_encoder_fn(input_vocab_size, d_model, n_encoder_layers)
    pre_attention_decoder = pre_attention_decoder_fn(mode, target_vocab_size, d_model)

    model = tl.Serial(
        tl.Select([0, 1, 0, 1]),
        tl.Parallel(input_encoder, pre_attention_decoder),
        tl.Fn('PrepareAttentionInput', prepare_attention_input, n_out=4),

        # nest it inside a Residual layer to add to the pre-attention decoder activations(i.e. queries)
        tl.Residual(tl.AttentionQKV(d_model, n_heads=n_attention_heads, dropout=attention_dropout, mode=mode)),

        # Step 6: drop attention mask (i.e. index = None
        tl.Select([0, 2]),
        [tl.LSTM(d_model) for _ in range(n_decoder_layers)],
        tl.Dense(target_vocab_size),
        tl.LogSoftmax())
    return model


def set_model(model, train_stream, eval_stream, output_dir):
    train_task = training.TrainTask(
        labeled_data=train_stream,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(.01),
        lr_schedule=trax.lr.warmup_and_rsqrt_decay(1000, .01),
        n_steps_per_checkpoint=10)

    eval_task = training.EvalTask(
        labeled_data=eval_stream,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()])

    training_loop = training.Loop(model,
                                  train_task,
                                  eval_tasks=[eval_task],
                                  output_dir=output_dir)
    return training_loop


def load_model(path):
    model = NMTAttn(mode='eval')
    model.init_from_file(path, weights_only=True)
    model = tl.Accelerate(model)
    return model


def next_symbol(NMTAttn_model, input_tokens, cur_output_tokens, temperature):
    token_length = len(cur_output_tokens)
    padded_length = 2 ** int(np.ceil(np.log2(token_length + 1)))
    padded = cur_output_tokens + [0] * (padded_length - token_length)
    # model expects the output to have an axis for the batch size in front so
    # convert `padded` list to a numpy array with shape (x, <padded_length>) where the
    # x position is the batch axis. (hint: you can use np.expand_dims() with axis=0 to insert a new axis)
    padded_with_batch = np.expand_dims(padded, axis=0)
    # get the model prediction. remember to use the `NMAttn` argument defined above.
    # hint: the model accepts a tuple as input (e.g. `my_model((input1, input2))`)
    output, _ = NMTAttn_model((input_tokens, padded_with_batch))
    # get log probabilities from the last token output
    log_probs = output[0, token_length, :]
    # get the next symbol by getting a logsoftmax sample (*hint: cast to an int)
    symbol = int(tl.logsoftmax_sample(log_probs, temperature))
    return symbol, float(log_probs[symbol])


def sampling_decode(input_sentence, NMTAttn_model, temperature, vocab_file, vocab_dir, EOS_index):
    input_tokens = tokenize(input_sentence, vocab_file, vocab_dir, EOS_index)
    cur_output_tokens = []
    # initialize an integer that represents the current output index
    cur_output = 0
    # check that the current output is not the end of sentence token
    count = 1
    while cur_output != EOS_index:
        # update the current output token by getting the index of the next word (hint: use next_symbol)
        cur_output, log_prob = next_symbol(NMTAttn_model, input_tokens, cur_output_tokens, temperature)
        cur_output_tokens.append(cur_output)
        if count > 3:
            print(detokenize(cur_output_tokens, vocab_file, vocab_dir, EOS_index))
        count += 1

    sentence = detokenize(cur_output_tokens, vocab_file, vocab_dir, EOS_index)

    return cur_output_tokens, log_prob, sentence