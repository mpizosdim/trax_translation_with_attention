import trax
import numpy as np

__boundaries = [8,   16,  32, 64, 128, 256, 512]
__batch_sizes = [256, 128, 64, 32, 16,    8,   4,  2]


def get_data(dataset, data_dir, keys):
    train_stream_fn = trax.data.TFDS(dataset, data_dir=data_dir, keys=keys, eval_holdout_size=0.01, train=True)
    eval_stream_fn = trax.data.TFDS(dataset, data_dir=data_dir, keys=keys, eval_holdout_size=0.01, train=False)
    return train_stream_fn, eval_stream_fn


def tokenize_training(train_stream, eval_stream, vocab_file, vocab_dir):
    tokenized_train_stream = trax.data.Tokenize(vocab_file=vocab_file, vocab_dir=vocab_dir)(train_stream)
    tokenized_eval_stream = trax.data.Tokenize(vocab_file=vocab_file, vocab_dir=vocab_dir)(eval_stream)
    return tokenized_train_stream, tokenized_eval_stream


def append_eos(stream, EOS):
    for (inputs, targets) in stream:
        inputs_with_eos = list(inputs) + [EOS]
        targets_with_eos = list(targets) + [EOS]
        yield np.array(inputs_with_eos), np.array(targets_with_eos)


def bucketing(data_stream):
    batch_stream = trax.data.BucketByLength(__boundaries, __batch_sizes, length_keys=[0, 1])(data_stream)
    return batch_stream


def mask(data_stream, mask_id):
    batch_stream = trax.data.AddLossWeights(id_to_mask=mask_id)(data_stream)
    return batch_stream


def process_data(dataset, data_dir_dataset, vocab_file, vocab_dir, keys, EOS_index):
    train_stream_fn, eval_stream_fn = get_data(dataset, data_dir_dataset, keys)

    tokenized_train_stream, tokenized_eval_stream = tokenize_training(train_stream_fn(), eval_stream_fn(), vocab_file, vocab_dir)

    tokenized_train_stream = append_eos(tokenized_train_stream, EOS_index)
    tokenized_eval_stream = append_eos(tokenized_eval_stream, EOS_index)

    filtered_train_stream = trax.data.FilterByLength(max_length=256, length_keys=[0, 1])(tokenized_train_stream)
    filtered_eval_stream = trax.data.FilterByLength(max_length=512, length_keys=[0, 1])(tokenized_eval_stream)

    train_batch_stream = bucketing(filtered_train_stream)
    eval_batch_stream = bucketing(filtered_eval_stream)

    train_batch_stream = mask(train_batch_stream, 0)
    eval_batch_stream = mask(eval_batch_stream, 0)

    return train_batch_stream, eval_batch_stream


def tokenize(input_str, vocab_file, vocab_dir, EOS_index):
    inputs = next(trax.data.tokenize(iter([input_str]), vocab_file=vocab_file, vocab_dir=vocab_dir))
    inputs = list(inputs) + [EOS_index]
    batch_inputs = np.reshape(np.array(inputs), [1, -1]) # Adding the batch dimension to the front of the shape
    return batch_inputs


def detokenize(integers, vocab_file, vocab_dir, EOS_index):
    integers = list(np.squeeze(integers)) # Remove the dimensions of size 1
    if EOS_index in integers: # Remove the EOS to decode only the original tokens
        integers = integers[:integers.index(EOS_index)]

    return trax.data.detokenize(integers, vocab_file=vocab_file, vocab_dir=vocab_dir)
