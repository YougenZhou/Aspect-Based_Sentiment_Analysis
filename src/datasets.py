import torch
from dgl.data import DGLDataset
import os
from dgl.data.utils import save_graphs, load_graphs
from src.preprocess import load_data_form_json, data2graph


class ABSADataset(DGLDataset):
    def __init__(self, config, split_name, save_graph=True):
        self.config = config
        self.save_graph = save_graph
        self.split = split_name
        raw_dir = os.path.join(os.path.abspath('..'), 'Amax_DLGM', 'data')
        name = config['dataset']
        self.graphs, self.labels = [], []
        super(ABSADataset, self).__init__(raw_dir=raw_dir, name=name)

    def process(self):
        data_path = os.path.join(self.raw_path, self.split + '.json')
        data = load_data_form_json(data_path)
        self.graphs, labels = data2graph(data, self.raw_path, self.config)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, item):
        return self.graphs[item], self.labels[item]

    def __len__(self):
        return len(self.labels)

    def save(self):
        if self.save_graph:
            graph_path = os.path.join(self.save_path, 'dgl_graph_{}_{}.bin'.format(self.name, self.split))
            save_graphs(str(graph_path), self.graphs, {'labels': self.labels})

    def load(self):
        graphs, labels_dict = load_graphs(
            os.path.join(self.save_path, 'dgl_graph_{}_{}.bin'.format(self.name, self.split)))
        self.graphs = graphs
        self.labels = labels_dict['labels']

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph_{}_{}.bin'.format(self.name, self.split))
        if os.path.exists(graph_path):
            return True
        return False