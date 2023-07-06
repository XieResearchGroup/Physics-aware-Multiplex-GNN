# Modified based on https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/tu_dataset.py

import os
import os.path as osp
import glob

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset


names = [
    'A', 'graph_indicator', 'node_labels', 'node_attributes'
    'edge_labels', 'edge_attributes', 'graph_labels', 'graph_attributes'
]

class TUDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False):
        self.name = name
        self.cleaned = cleaned
        super(TUDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


def read_tu_data(folder, prefix):
    files = glob.glob(osp.join(folder, '{}_*.txt'.format(prefix)))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    node_attributes = node_labels = None
    if 'node_attributes' in names:
        node_attributes = read_file(folder, prefix, 'node_attributes')
    if 'node_labels' in names:
        node_labels = read_file(folder, prefix, 'node_labels')
    x = cat([node_attributes, node_labels])

    edge_attributes, edge_labels = None, None
    if 'edge_attributes' in names:
        edge_attributes = read_file(folder, prefix, 'edge_attributes')
    if 'edge_labels' in names:
        edge_labels = read_file(folder, prefix, 'edge_labels', torch.long)
        if edge_labels.dim() == 1:
            edge_labels = edge_labels.unsqueeze(-1)
        edge_labels = edge_labels - edge_labels.min(dim=0)[0]
        edge_labels = edge_labels.unbind(dim=-1)
        edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
        edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
    edge_attr = cat([edge_attributes, edge_labels])

    y = None
    if 'graph_attributes' in names:
        y = read_file(folder, prefix, 'graph_attributes')
    elif 'graph_labels' in names:
        y = read_file(folder, prefix, 'graph_labels')

    data = Data(x=x, y=y)
    data, slices = split(data, batch)

    return data, slices

def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, '{}_{}.txt'.format(prefix, name))
    return read_txt_array(path, sep=',', dtype=dtype)

def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'x': node_slice}
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices