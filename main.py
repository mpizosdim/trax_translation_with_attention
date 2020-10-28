from process_utils import process_data
from ml_utils import NMTAttn, set_model, load_model, sampling_decode


def train_process():
    dataset = 'opus/medical'
    data_dir_dataset = '/home/dimitris/Documents/tutorials/NLP_specialization_coursera/course4_attension_models/week1new/data/opus/medical/0.1.0/data/'
    vocab_file = 'ende_32k.subword'
    vocab_dir = '/home/dimitris/Documents/tutorials/NLP_specialization_coursera/course4_attension_models/week1new/data/'
    output_dir_for_model = './data/'
    keys = ('en', 'de')
    EOS_index = 1

    train_stream, eval_stream = process_data(dataset, data_dir_dataset, vocab_file, vocab_dir, keys, EOS_index)
    model = NMTAttn(mode='train')
    training_loop = set_model(model, train_stream, eval_stream, output_dir_for_model)
    training_loop.run(100)


def test_process():
    output_dir_for_model = './data/model.pkl.gz'
    vocab_file = 'ende_32k.subword'
    vocab_dir = '/home/dimitris/Documents/tutorials/NLP_specialization_coursera/course4_attension_models/week1new/data/'
    EOS_index = 1

    model = load_model(output_dir_for_model)
    result = sampling_decode("i love you", model, 0.0, vocab_file, vocab_dir, EOS_index)
    print(result)


if __name__ == '__main__':
    test_process()
