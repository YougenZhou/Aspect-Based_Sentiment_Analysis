import json
import dgl
from transformers import BartTokenizer, BartModel, RobertaTokenizer, RobertaModel, BertModel, BertTokenizer
import os
import torch
from utils.vocab import Vocab


def data2graph(data, raw_path, config):
    graphs, labels = [], []
    for sample in data:
        token, term, label, pos, dis, syn, frm, to = sample
        src = list(range(len(token)))
        dst = [0] * len(token)
        graph = dgl.heterograph({('sentence', 'channel', 'aspect'): (src, dst)})
        token_, label_, pos, dis, syn = get_pos_embedding(token, label, pos, dis, syn, raw_path)
        graph.nodes['sentence'].data['input_ids'] = token_
        graph.nodes['aspect'].data['input_ids'] = torch.tensor([[frm, to]], dtype=torch.long)
        graph.edata['pos'] = pos
        graph.edata['dis'] = dis
        graph.edata['syn'] = syn
        graphs.append(graph)
        labels.append(label_)
    torch.tensor(labels, dtype=torch.long)
    return graphs, labels


def load_data_form_json(file_path):
    processed = []
    with open(file_path, 'r', encoding='utf-8') as js:
        file = json.load(js)
        js.close()
    for instance in file:
        for aspect in instance['aspects']:
            token = list(instance['token'])
            term = list(aspect['term'])
            frm = aspect['from']
            to = aspect['to']
            label = aspect['polarity']
            pos = list(instance['pos'])
            head = list([int(h) for h in instance['head']])
            deprel = list(instance['deprel'])
            length = len(token)
            dis = get_distance_between_nodes(length, frm, to)
            syn_relations = get_syn_relations(token, term, head, deprel, frm, to)
            processed.append((token, term, label, pos, dis, syn_relations, frm, to))
    return processed


def get_distance_between_nodes(length, frm, to):
    info = 1e10
    nodes_distance = []
    for i in range(length):
        if i in range(frm, to):
            distance = 0
        elif i < frm:
            distance = frm - i
        else:
            distance = i - to + 1
        if distance > 4:
            distance = info
        nodes_distance.append(distance)
    return nodes_distance


def get_syn_relations(token, term, head, deprel, frm, to):
    word_list = []
    for i in range(frm, to):
        w = token[head[i] - 1]
        if w not in term:
            word_list.append(w)
    syn_relations = []
    for idx, word in enumerate(token):
        relation = None
        if word in term:
            relation = 'self'
        elif token[(head[idx] - 1)] in term or word in word_list:
            relation = deprel[idx]
        else:
            path = []
            path = get_path(idx, head, token, path)
            for item in term:
                if item in path:
                    relation = str(len(path)) + ':con'
                else:
                    relation = 'outer'
        syn_relations.append(relation)
    return syn_relations


def get_path(idx, head, token, path):
    if head[idx] == 0:
        path.append(token[idx])
        return path
    else:
        path.append(token[idx])
        return get_path(head[idx] - 1, head, token, path)


def get_nodes_embedding(token, term, config):
    if config['pretrained_model'] == 'bart':
        model_path = os.path.join(os.path.abspath('..'), 'Amax_DLGM/ptms/bart_base')
        tokenizer = BartTokenizer.from_pretrained(model_path)
        pretrained_model = BartModel.from_pretrained(model_path)
    elif config['pretrained_model'] == 'bert':
        model_path = os.path.join(os.path.abspath('..'), 'Amax_DLGM/ptms/bert_base_uncased')
        tokenizer = BertTokenizer.from_pretrained(model_path)
        pretrained_model = BertModel.from_pretrained(model_path)
    else:
        model_path = os.path.join(os.path.abspath('..'), 'Amax_DLGM/ptms/roberta_base')
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        pretrained_model = RobertaModel.from_pretrained(model_path)
    token_input_ids = tokenizer(token, is_split_into_words=True, return_tensors='pt')
    token_embeddings = pretrained_model(**token_input_ids).last_hidden_state
    tokens_embedding = get_embeddings(token_input_ids['input_ids'].squeeze(0), tokenizer, token_embeddings)
    term_input_ids = tokenizer(term, is_split_into_words=True, return_tensors='pt')['input_ids']
    term_embeddings = pretrained_model(torch.cat([token_input_ids['input_ids'], term_input_ids[:, 1:]], dim=1)).last_hidden_state
    term_embeddings = term_embeddings[:, -len(term_input_ids.squeeze(0)):, :]
    terms_embedding = torch.sum(term_embeddings.squeeze(0), dim=0) / len(term_input_ids.squeeze(0))
    return tokens_embedding, terms_embedding.unsqueeze(0)


def get_embeddings(ids, tokenizer, embeddings):
    tokenizer_tokens = tokenizer.convert_ids_to_tokens(ids)
    token_index = []
    for i, tt in enumerate(tokenizer_tokens):
        if tt[0] == 'Ġ':
            token_index.append(i)
    token_embedding_list = []
    for i, idx in enumerate(token_index):
        if (i + 1) != len(token_index) and token_index[i + 1] - idx != 1:
            pooling_embeddings = embeddings.squeeze(0)[idx: token_index[i + 1], :]
            embedding = torch.sum(pooling_embeddings, 0) / (token_index[i + 1] - idx)
        else:
            embedding = embeddings.squeeze(0)[idx, :]
        token_embedding_list.append(embedding.unsqueeze(0))
    return torch.cat(token_embedding_list)


def get_pos_embedding(token, label, pos, dis, syn, raw_path):
    tok_vocab = Vocab.load_vocab(os.path.join(raw_path, 'vocab_tok.vocab'))
    pos_vocab = Vocab.load_vocab(os.path.join(raw_path, 'vocab_pos.vocab'))
    spd_vocab = Vocab.load_vocab(os.path.join(raw_path, 'vocab_spd.vocab'))
    dep_vocab = Vocab.load_vocab(os.path.join(raw_path, 'vocab_dep.vocab'))
    pol_vocab = Vocab.load_vocab(os.path.join(raw_path, 'vocab_pol.vocab'))
    token_embedding = torch.tensor([tok_vocab.toks2index[t] for t in token], dtype=torch.long)
    label_embedding = pol_vocab.toks2index[label]
    pos_embedding = torch.tensor([pos_vocab.toks2index[p] for p in pos], dtype=torch.long)
    dis_embedding = torch.tensor([spd_vocab.toks2index[d] for d in dis], dtype=torch.long)
    syn_embedding = torch.tensor([dep_vocab.toks2index[s] for s in syn], dtype=torch.long)
    return token_embedding, label_embedding, pos_embedding, dis_embedding, syn_embedding

