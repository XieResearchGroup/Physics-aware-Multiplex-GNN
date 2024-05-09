import os
import torch
import numpy as np
import pickle
from torch_geometric.data import Data, Dataset
from preprocess_rna_pdb import REV_RESIDUES, COARSE_GRAIN_MAP

class RNAPDBDataset(Dataset):
    backbone_atoms = ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']
    
    def __init__(self,
                 path: str,
                 name: str,
                 file_extension: str='.pkl',
                 mode: str='backbone'
                 ):
        super(RNAPDBDataset, self).__init__(path)
        self.path = os.path.join(path, name)
        self.files = sorted(os.listdir(self.path))
        self.files = os.listdir(self.path)
        self.files = [f for f in self.files if f.endswith(file_extension)]
        self.to_tensor = torch.tensor
        if mode not in ['backbone', 'all', 'coarse-grain']:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    def len(self):
        return len(self.files)

    def get(self, idx):
        data_x, edges, name, edges_type = self.get_raw_sample(idx)
        data = Data(
            x=data_x,
            edge_index=torch.tensor(edges).t().contiguous(),
            edge_attr=edges_type
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
        atoms_pos_mean = atoms_pos.mean(dim=0)
        atoms_pos -= atoms_pos_mean # Center around point (0,0,0)
        atoms_pos /= 10
        c2 = c4_or_c6 = n1_or_n9 = None
        if self.mode == 'backbone':
            atoms_pos, atoms_types, c4_primes, residues = self.backbone_only(atoms_pos, atoms_types, sample)
        elif self.mode == 'coarse-grain':
            atoms_pos, atoms_types, c4_primes, residues, c2, c4_or_c6, n1_or_n9 = self.coarse_grain(atoms_pos, atoms_types, sample)
        elif self.mode == 'all':
            c4_primes = sample['c4_primes']
            residues = sample['residues']
            c2 = sample['c2']
            c4_or_c6 = sample['c4_or_c6']
            n1_or_n9 = sample['n1_or_n9']

        name = sample['name'].replace('.pkl', '')
        # convert atom_types to one-hot encoding (C, O, N, P)
        atoms_types = torch.nn.functional.one_hot(atoms_types.to(torch.int64), num_classes=4).float()
        atoms_types = atoms_types.squeeze(1)

        c4_primes = torch.tensor(c4_primes).float().unsqueeze(1)
        if c2 is not None:
            c2 = torch.tensor(c2).float().unsqueeze(1)
            c4_or_c6 = torch.tensor(c4_or_c6).float().unsqueeze(1)
            n1_or_n9 = torch.tensor(n1_or_n9).float().unsqueeze(1)
        residues = torch.nn.functional.one_hot(torch.tensor(residues).to(torch.int64), num_classes=4).float()

        if c2 is not None:
            data_x = torch.cat((atoms_pos, atoms_types, residues, c4_primes, c2, c4_or_c6, n1_or_n9), dim=1)
        else:
            data_x = torch.cat((atoms_pos, atoms_types, residues, c4_primes), dim=1)

        return data_x, sample['edges'], name, torch.nn.functional.one_hot(torch.tensor(sample['edge_type']).to(torch.int64), num_classes=3).float()

    def backbone_only(self, atom_pos, atom_types, sample):
        mask = [True if atom in self.backbone_atoms else False for atom in sample['symbols']]
        c4_primes = sample['c4_primes']
        residues = sample['residues']
        return atom_pos[mask], atom_types[mask], c4_primes[mask], residues[mask]
    
    def coarse_grain(self, atom_pos, atom_types, sample):
        # mask = sample['crs-grain-mask']
        c4_primes = sample['c4_primes']
        c2 = sample['c2']
        c4_or_c6 = sample['c4_or_c6']
        n1_or_n9 = sample['n1_or_n9']
        residues = sample['residues']
        return atom_pos, atom_types, c4_primes, residues, c2, c4_or_c6, n1_or_n9

    @property
    def raw_file_names(self):
        return self.files

    @property
    def processed_file_names(self):
        return self.files

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)