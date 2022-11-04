import dgl
import torch
from torch import nn
import os
from dgl import function as fn
from dgl.nn.functional import edge_softmax
from utils.vocab import Vocab
from transformers import BartTokenizer, BartModel, RobertaTokenizer, RobertaModel, BertModel, BertTokenizer
import torch.nn.functional as F
from utils.visual import visual_embedding, visual_linguistic


class DLGM(nn.Module):
    def __init__(self, config):
        super(DLGM, self).__init__()
        self.config = config
        self.get_embedding = GetEmbedding(config)
        self.linguistic_property_extraction = LPE(config)
        self.linguistic_routing_mechanism = LRM()
        self.iterate_times = config['iterate_times']
        self.ffn = nn.Sequential(
            nn.Linear(config['hidden_dim'] * 3, 256),
            nn.ReLU(),
            nn.Dropout(config['cls_dropout'])
        )
        self.cls = nn.Linear(256, config['label_vocab'])

    def forward(self, graphs):
        graph = self.get_embedding(graphs)

        X = graph.nodes['sentence'].data['embedding']
        Y = graph.nodes['sentence'].data['input_ids']

        visual_embedding(X, Y)

        graph = self.linguistic_property_extraction(graph)

        graph = self.linguistic_routing_mechanism(graph)

        final_embedding = torch.cat([graph.dstdata['fina_pos'], graph.dstdata['fina_dis'], graph.dstdata['fina_syn']
                                     ], dim=1)

        logit = self.cls(self.ffn(final_embedding + graph.nodes['aspect'].data['embedding']))
        return logit


class GetEmbedding(nn.Module):
    def __init__(self, config):
        super(GetEmbedding, self).__init__()
        vocab_path = os.path.join(os.path.abspath('..'), 'Amax_DLGM', 'data', config['dataset'], 'vocab_tok.vocab')
        tok_vocab = Vocab.load_vocab(vocab_path).toks2index
        self.tok_vocab = {v: k for k, v in tok_vocab.items()}
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
        self.tokenizer = tokenizer
        self.pretrained_model = pretrained_model

    def forward(self, graphs):
        graphs = dgl.unbatch(graphs)
        graph_all = []
        for graph in graphs:
            token_ids = graph.nodes['sentence'].data['input_ids'].tolist()
            tokens = [self.tok_vocab[ids] for ids in token_ids]

            aspect_ids = graph.nodes['aspect'].data['input_ids'].squeeze(0).tolist()
            aspects = tokens[aspect_ids[0]: aspect_ids[1]]

            token_input_ids = self.tokenizer(tokens, is_split_into_words=True, return_tensors='pt')['input_ids'].to(
                'cuda')
            aspect_input_ids = self.tokenizer(aspects, is_split_into_words=True, return_tensors='pt')['input_ids'].to(
                'cuda')

            token_embeddings = self.pretrained_model(token_input_ids).last_hidden_state
            aspect_embeddings = self.pretrained_model(
                torch.cat([token_input_ids, aspect_input_ids[:, 1:]], dim=1)).last_hidden_state
            terms_embedding = aspect_embeddings[:, -len(aspect_input_ids.squeeze(0)):, :]
            terms_embedding = torch.sum(terms_embedding.squeeze(0), dim=0) / len(aspect_input_ids.squeeze(0))

            graph.nodes['sentence'].data['embedding'] = self.get_node_embedding(token_input_ids, token_embeddings)
            graph.nodes['aspect'].data['embedding'] = terms_embedding.unsqueeze(0)
            graph_all.append(graph)

        return dgl.batch(graph_all)

    def get_node_embedding(self, ids, embeddings):
        tokenizer_tokens = self.tokenizer.convert_ids_to_tokens(ids.squeeze(0))

        token_index = []
        for i, token in enumerate(tokenizer_tokens):
            if token[0] == 'Ġ':
                token_index.append(i)

        token_embedding_list = []
        for i, idx in enumerate(token_index):
            if (i + 1) != len(token_index) and token_index[i + 1] - idx != 1:
                pooling_embeddings = embeddings.squeeze(0)[idx: token_index[i + 1], :]
                embedding = torch.sum(pooling_embeddings, 0) / (token_index[i + 1] - idx)
            else:
                embedding = embeddings.squeeze(0)[idx, :]
            token_embedding_list.append(embedding.unsqueeze(0))

        return torch.cat(token_embedding_list, 0)


class LPE(nn.Module):
    def __init__(self, config):
        super(LPE, self).__init__()
        self.pos_emb = nn.Embedding(config['pos_vocab'], config['hidden_dim'])
        self.dis_emb = nn.Embedding(config['spd_vocab'], config['hidden_dim'])
        self.syn_emb = nn.Embedding(config['dep_vocab'], config['hidden_dim'])
        self.linear = nn.Linear(768, config['embedding_dim'])
        self.dropout = nn.Dropout(config['embedding_dropout'])
        self.pos_neural = nn.Linear(768, config['hidden_dim'])
        self.dis_neural = nn.Linear(768, config['hidden_dim'])
        self.syn_neural = nn.Linear(768, config['hidden_dim'])
        self.init_model_params(self.pos_emb)
        self.init_model_params(self.dis_emb)
        self.init_model_params(self.syn_emb)
        self.init_model_params(self.linear)
        self.init_model_params(self.pos_neural)
        self.init_model_params(self.dis_neural)
        self.init_model_params(self.syn_neural)

    def init_model_params(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def get_linguistic_loss(self, inputs, targets):
        linguistic_loss = (1 - F.cosine_similarity(torch.cat(inputs), torch.cat(targets)))
        linguistic_loss = torch.sum(linguistic_loss) / linguistic_loss.shape[0]
        ATM = (targets[0] + targets[1] + targets[2]) / len(targets)
        loss1 = F.cosine_similarity(targets[0], ATM) + F.cosine_similarity(targets[1], ATM) + F.cosine_similarity(targets[2], ATM)
        independence_loss = torch.sum(loss1 / 3) / loss1.shape[0]
        return linguistic_loss, independence_loss

    def forward(self, graph):
        tokens_embedding = graph.srcdata['embedding']
        terms_embedding = graph.dstdata['embedding']
        graph.nodes['sentence'].data['h'] = self.linear(self.dropout(tokens_embedding))
        graph.nodes['aspect'].data['h'] = self.linear(self.dropout(terms_embedding))
        graph.edata['pos_embedding'] = self.pos_emb(graph.edata['pos'])
        graph.edata['dis_embedding'] = self.dis_emb(graph.edata['dis'])
        graph.edata['syn_embedding'] = self.syn_emb(graph.edata['syn'])

        tokens_pos_embedding = self.pos_neural(tokens_embedding)
        tokens_dis_embedding = self.dis_neural(tokens_embedding)
        tokens_syn_embedding = self.syn_neural(tokens_embedding)
        terms_pos_embedding = self.pos_neural(terms_embedding)
        terms_dis_embedding = self.dis_neural(terms_embedding)
        terms_syn_embedding = self.syn_neural(terms_embedding)
        graph.srcdata.update({'pos': tokens_pos_embedding})
        graph.srcdata.update({'dis': tokens_dis_embedding})
        graph.srcdata.update({'syn': tokens_syn_embedding})
        graph.dstdata.update({'pos': terms_pos_embedding})
        graph.dstdata.update({'dis': terms_dis_embedding})
        graph.dstdata.update({'syn': terms_syn_embedding})

        visual_linguistic([tokens_pos_embedding, tokens_dis_embedding, tokens_syn_embedding])

        # inputs = [tokens_pos_embedding, tokens_dis_embedding, tokens_syn_embedding]
        # targets = [graph.edata['pos_embedding'], graph.edata['dis_embedding'], graph.edata['syn_embedding']]

        # linguistic_loss, inde_loss = self.get_linguistic_loss(inputs, targets)
        return graph


class LRM(nn.Module):
    def __init__(self):
        super(LRM, self).__init__()
        self.pos_linguistic_bias = nn.Linear(256, 256)
        self.dis_linguistic_bias = nn.Linear(256, 256)
        self.syn_linguistic_bias = nn.Linear(256, 256)

    def forward(self, graph):
        h_0 = graph.nodes['aspect'].data['h']

        graph.apply_edges(fn.u_dot_v('pos', 'pos', 'intra_pos'))
        graph.apply_edges(fn.u_dot_v('dis', 'dis', 'intra_dis'))
        graph.apply_edges(fn.u_dot_v('syn', 'syn', 'intra_syn'))

        graph.edata['pos_intra_linguistic_metrix'] = edge_softmax(graph, graph.edata['intra_pos'])
        graph.edata['dis_intra_linguistic_metrix'] = edge_softmax(graph, graph.edata['intra_dis'])
        graph.edata['syn_intra_linguistic_metrix'] = edge_softmax(graph, graph.edata['intra_syn'])

        graph.update_all(fn.u_mul_e('h', 'pos_intra_linguistic_metrix', 'attn_pos'), fn.sum('attn_pos', 'i_pos'))
        graph.update_all(fn.u_mul_e('h', 'dis_intra_linguistic_metrix', 'attn_dis'), fn.sum('attn_dis', 'i_dis'))
        graph.update_all(fn.u_mul_e('h', 'syn_intra_linguistic_metrix', 'attn_syn'), fn.sum('attn_syn', 'i_syn'))

        graph.edata['pos_cross_linguistic_metrix'] = torch.exp(graph.edata['intra_pos']) / (
                torch.exp(graph.edata['intra_pos']) + torch.exp(graph.edata['intra_dis']) + torch.exp(
            graph.edata['intra_syn']))

        graph.edata['dis_cross_linguistic_metrix'] = torch.exp(graph.edata['intra_dis']) / (
                torch.exp(graph.edata['intra_pos']) + torch.exp(graph.edata['intra_dis']) + torch.exp(
            graph.edata['intra_syn']))

        graph.edata['syn_cross_linguistic_metrix'] = torch.exp(graph.edata['intra_syn']) / (
                torch.exp(graph.edata['intra_pos']) + torch.exp(graph.edata['intra_dis']) + torch.exp(
            graph.edata['intra_syn']))

        graph.update_all(fn.u_mul_e('h', 'pos_cross_linguistic_metrix', 'cross_pos'), fn.sum('cross_pos', 'c_pos'))
        graph.update_all(fn.u_mul_e('h', 'dis_cross_linguistic_metrix', 'cross_dis'), fn.sum('cross_dis', 'c_dis'))
        graph.update_all(fn.u_mul_e('h', 'syn_cross_linguistic_metrix', 'cross_syn'), fn.sum('cross_syn', 'c_syn'))

        pos_bias = self.pos_linguistic_bias(graph.nodes['aspect'].data['pos'])
        dis_bias = self.dis_linguistic_bias(graph.nodes['aspect'].data['dis'])
        syn_bias = self.syn_linguistic_bias(graph.nodes['aspect'].data['syn'])

        u_pos = graph.nodes['aspect'].data['c_pos'] + graph.nodes['aspect'].data['i_pos'] + pos_bias
        u_dis = graph.nodes['aspect'].data['c_dis'] + graph.nodes['aspect'].data['i_dis'] + dis_bias
        u_syn = graph.nodes['aspect'].data['c_syn'] + graph.nodes['aspect'].data['i_syn'] + syn_bias

        graph.dstdata.update({'fina_pos': u_pos + h_0})
        graph.dstdata.update({'fina_dis': u_dis + h_0})
        graph.dstdata.update({'fina_syn': u_syn + h_0})

        return graph
