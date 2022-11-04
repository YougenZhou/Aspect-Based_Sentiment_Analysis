import os
from collections import Counter
from utils.vocab import Vocab
from src.preprocess import load_data_form_json


def prepare(config):
    base_path = '../data'
    dataset = config['dataset']

    train_file = os.path.join(base_path, dataset, 'train.json')
    valid_file = os.path.join(base_path, dataset, 'valid.json')
    test_file = os.path.join(base_path, dataset, 'test.json')

    vocab_tok_file = os.path.join(base_path, dataset, 'vocab_tok.vocab')
    vocab_dep_file = os.path.join(base_path, dataset, 'vocab_dep.vocab')
    vocab_pos_file = os.path.join(base_path, dataset, 'vocab_pos.vocab')
    vocab_pol_file = os.path.join(base_path, dataset, 'vocab_pol.vocab')
    vocab_spd_file = os.path.join(base_path, dataset, 'vocab_spd.vocab')

    print('Loading files...')
    train_tok, train_label, train_pos, train_dis, train_dep = load_files(train_file)
    valid_tok, valid_label, valid_pos, valid_dis, valid_dep = load_files(valid_file)
    test_tok, test_label, test_pos, test_dis, test_dep = load_files(test_file)

    tok_counter = Counter(train_tok + test_tok + valid_tok)
    label_counter = Counter(train_label + valid_label + test_label)
    pos_counter = Counter(train_pos + valid_pos + test_pos)
    dis_counter = Counter(train_dis + valid_dis + test_dis)
    syn_counter = Counter(train_dep + valid_dep + test_dep)

    print('building vocab...')
    tok_vocab = Vocab(tok_counter, specials=[])
    lable_vocab = Vocab(label_counter, specials=[])
    dep_vocab = Vocab(syn_counter, specials=[])
    pos_vocab = Vocab(pos_counter, specials=[])
    spd_vocab = Vocab(dis_counter, specials=[])
    print('tok_vocab: {}, label_vocab: {}, pos_vocab: {}, spd_vocab: {}, dep_vocab: {}, '.format(len(tok_vocab),
                                                                                                 len(lable_vocab),
                                                                                                 len(pos_vocab),
                                                                                                 len(spd_vocab),
                                                                                                 len(dep_vocab)))

    print('saving vocab...')
    tok_vocab.save_vocab(vocab_tok_file)
    dep_vocab.save_vocab(vocab_dep_file)
    pos_vocab.save_vocab(vocab_pos_file)
    lable_vocab.save_vocab(vocab_pol_file)
    spd_vocab.save_vocab(vocab_spd_file)


def load_files(file_path):
    data = load_data_form_json(file_path)
    dep_list = []
    pos_list = []
    tok_list = []
    dis_list = []
    label_list = []
    for d in data:
        token, _, label, pos, dis, syn, _, _ = d
        tok_list.extend(token)
        pos_list.extend(pos)
        dep_list.extend(syn)
        dis_list.extend(dis)
        label_list.extend([label])
    return tok_list, label_list, pos_list, dis_list, dep_list


if __name__ == '__main__':
    config = {
        'dataset': 'Laptops',
    }
    prepare(config)
