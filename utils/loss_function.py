import torch.nn.functional as F
import torch
import dgl.function as fn


def fina_loss(graph, logit, label, config):
    alpha = config['alpha']
    beta = config['beta']
    loss = alpha * main_loss(logit, label) + beta * (information_loss(graph) + independence_loss(graph))
    return loss


def information_loss(graph):
    graph.apply_edges(fn.copy_src('cls_pos', 'out_pos'))
    graph.apply_edges(fn.copy_src('cls_dis', 'out_dis'))
    graph.apply_edges(fn.copy_src('cls_syn', 'out_syn'))

    pos_loss = F.cross_entropy(graph.srcdata['cls_pos'], graph.edata['pos'])
    dis_loss = F.cross_entropy(graph.srcdata['cls_dis'], graph.edata['dis'])
    syn_loss = F.cross_entropy(graph.srcdata['cls_syn'], graph.edata['syn'])
    info_loss = (pos_loss + dis_loss + syn_loss) / 3

    return info_loss


def independence_loss(graph):
    pos = graph.nodes['sentence'].data['pos']
    dis = graph.nodes['sentence'].data['dis']
    syn = graph.nodes['sentence'].data['syn']

    AMT = (pos + dis + syn) / 3
    pos_loss = torch.cosine_similarity(pos, AMT)
    dis_loss = torch.cosine_similarity(dis, AMT)
    syn_loss = torch.cosine_similarity(syn, AMT)
    ind_loss = (pos_loss + dis_loss + syn_loss) / 3
    i_loss = torch.sum(ind_loss) / ind_loss.shape[0]

    return i_loss


def main_loss(logit, label):
    c_loss = F.cross_entropy(logit, label)
    return c_loss
