import os
import torch
import numpy as np
import pickle
from torch_geometric.data import Data, Dataset

class RNAPDBDataset(Dataset):
    
    def __init__(self,
                 path: str,
                 name: str,
                 file_extension: str='.pkl',
                 ):
        super(RNAPDBDataset, self).__init__(path)
        self.path = os.path.join(path, name)
        self.files = sorted(os.listdir(self.path))
        self.files = os.listdir(self.path)
        self.files = [f for f in self.files if f.endswith(file_extension)]
        self.to_tensor = torch.tensor

    def len(self):
        return len(self.files)

    def get(self, idx):
        data_x, batch, name = self.get_raw_sample(idx)
        if self.transform:
            torsions = self.transform(torsions)
        name = self.files[idx]
        data = Data(
            x=data_x
        )
        return data, name

    def get_raw_sample(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = self.files[idx]
        path = os.path.join(self.path, file)
        sample = self.load_pickle(path)
        atoms_types = self.to_tensor(sample['atoms']).unsqueeze(1).float()
        atoms_pos = self.to_tensor(sample['pos']).float()
        indicator = self.to_tensor(sample['indicator'])
        name = sample['name']
        data_x = torch.cat((atoms_pos, atoms_types), dim=1)
        return data_x, indicator, name

    @property
    def raw_file_names(self):
        return self.files

    @property
    def processed_file_names(self):
        return self.files

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)