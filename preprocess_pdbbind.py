import os
import numpy as np
from tqdm import tqdm
from openbabel import pybel
from scipy.spatial import distance

import torch
from torch_geometric.nn import radius
from torch_geometric.utils import remove_self_loops

from utils.featurizer import Featurizer


def find_interacting_atoms(decoy, target, cutoff=6.0):
    distances = distance.cdist(decoy, target)
    decoy_atoms, target_atoms = np.nonzero(distances < cutoff)
    decoy_atoms, target_atoms = decoy_atoms.tolist(), target_atoms.tolist()
    return decoy_atoms, target_atoms

def pocket_atom_num_from_mol2(name, path):
    n = 0
    with open('%s/%s/%s_pocket.mol2' % (path, name, name)) as f:
        for line in f:
            if '<TRIPOS>ATOM' in line:
                break
        for line in f:
            cont = line.split()
            if '<TRIPOS>BOND' in line or cont[7] == 'HOH':
                break
            n += int(cont[5][0] != 'H')
    return n

def construct_graphs(data_dir, save_dir, data_name, save_name, label_dict, cutoff, exclude_data_name=None):
    """
    For each ligand-protein complex, a graph G is constructed by concatenating 3 subgraphs:
    1. Complex subgraph: pocket & ligand
    2. Pocket subgraph: pocket + shift 100 angstroms along the x-axis
    3. Ligand subgraph: ligand + shift 200 angstroms along the x-axis

    The choice of 100/200 angstroms is to use a distance >> the scale of any complex in our dataset.
    By doing so, all 3 subgraphs are far away from each other in 3D space, and have no interactions.
    Then the message passings can be applied to the subgraphs in parallel by simply loading G.
    """
    pybel.ob.obErrorLog.StopLogging()
    print("Preprocessing", data_name)

    # Get list of directories to be excluded if needed
    if exclude_data_name != None:
        exclude_dir = os.path.join(data_dir, exclude_data_name)
        exclude_name_list = []
        for dir_name in os.listdir(exclude_dir):
            if dir_name not in ['index', 'readme']:
                exclude_name_list.append(dir_name)
    
    # Get list of directories for constructing graphs
    data_dir_full = os.path.join(data_dir, data_name)
    
    name_list = []
    for dir_name in os.listdir(data_dir_full):
        if dir_name not in ['index', 'readme']:
            if exclude_data_name != None:
                if dir_name not in exclude_name_list:
                    name_list.append(dir_name)
            else:
                name_list.append(dir_name)

    save_dir_full = os.path.join(save_dir, save_name, "raw")
    
    if not os.path.exists(save_dir_full):
        os.makedirs(save_dir_full)

    for file_name in [save_name + '_node_labels.txt', save_name + '_graph_indicator.txt', 
                save_name + '_node_attributes.txt', save_name + '_graph_labels.txt']:
        if os.path.isfile(os.path.join(save_dir_full, file_name)):
            os.remove(os.path.join(save_dir_full, file_name))

    for i in tqdm(range(len(name_list))):
        name = name_list[i]
        pdb_label = label_dict[name]
        pdb_label = np.array(pdb_label).reshape(-1, 1)

        featurizer = Featurizer(save_molecule_codes=False)

        charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')

        ligand = next(pybel.readfile('mol2', os.path.join(data_dir_full, name, name + '_ligand.mol2')))
        ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)

        pocket = next(pybel.readfile('mol2', os.path.join(data_dir_full, name, name + '_pocket.mol2')))
        pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)

        node_num = pocket_atom_num_from_mol2(name, data_dir_full)
        pocket_coords = pocket_coords[:node_num]
        pocket_features = pocket_features[:node_num]
        
        assert (ligand_features[:, charge_idx] != 0).any()
        assert (pocket_features[:, charge_idx] != 0).any()
        assert (ligand_features[:, :9].sum(1) != 0).all()
        assert ligand_features.shape[0] == ligand_coords.shape[0]
        assert pocket_features.shape[0] == pocket_coords.shape[0]
        
        pocket_interact, ligand_interact = find_interacting_atoms(pocket_coords, ligand_coords, cutoff)
        
        pocket_atoms = set([])
        pocket_atoms = pocket_atoms.union(set(pocket_interact))
        ligand_atoms = range(len(ligand_coords))
        
        pocket_atoms = np.array(list(pocket_atoms))
        
        pocket_coords = pocket_coords[pocket_atoms]
        pocket_features = pocket_features[pocket_atoms]

        ligand_pos = np.array(ligand_coords)
        pocket_pos = np.array(pocket_coords)

        pos = torch.tensor(pocket_pos)
        row, col = radius(pos, pos, 0.5, max_num_neighbors=1000)

        full_edge_index_long = torch.stack([row, col], dim=0)
        full_edge_index_long, _ = remove_self_loops(full_edge_index_long)
        if full_edge_index_long.size()[1] > 0:
            j_long, i_long = full_edge_index_long
            pocket_pos = np.delete(pocket_pos, j_long[:len(j_long)//2], axis=0)
            pocket_features = np.delete(pocket_features, j_long[:len(j_long)//2], axis=0)
        
        # Concat three subgraphs:
        complex_pos = np.concatenate((pocket_pos, ligand_pos), axis=0)
        complex_features = np.concatenate((pocket_features, ligand_features), axis=0)
        
        x_shift = np.mean(complex_pos[:, 0])
        complex_pos -= [x_shift, 0.0, 0.0]
        pocket_pos -= [x_shift, 0.0, 0.0]
        ligand_pos -= [x_shift, 0.0, 0.0]

        pocket_pos += [100.0, 0.0, 0.0]    # shift 100 angstroms along the x-axis
        ligand_pos += [200.0, 0.0, 0.0]    # shift 200 angstroms along the x-axis

        final_pos = np.concatenate((complex_pos, pocket_pos, ligand_pos), axis=0)
        final_features = np.concatenate((complex_features, pocket_features, ligand_features), axis=0)

        # Generate files for loading graphs
        indicator = np.ones((final_features.shape[0], 1)) * (i + 1)

        with open(os.path.join(save_dir_full, save_name + '_graph_indicator.txt'),'ab') as f:
            np.savetxt(f, indicator, fmt='%i', delimiter=', ')
        f.close()
    
        with open(os.path.join(save_dir_full, save_name + '_node_labels.txt'),'ab') as f:
            np.savetxt(f, final_features, fmt='%.4f', delimiter=', ')
        f.close()
 
        with open(os.path.join(save_dir_full, save_name + '_node_attributes.txt'),'ab') as f:
            np.savetxt(f, final_pos, fmt='%.3f', delimiter=', ')
        f.close()
        
        with open(os.path.join(save_dir_full, save_name + '_graph_labels.txt'),'ab') as f:
            np.savetxt(f, pdb_label, fmt='%.2f', delimiter=', ')
        f.close()


def main():
    data_dir = os.path.join(".", "data", "PDBbind")
    index_labels_file = os.path.join(data_dir, "refined-set", "index", "INDEX_refined_data.2016")
    
    # Loaded lines have format:
    # PDB code, resolution, release year, -logKd/Ki, Kd/Ki, reference, ligand name
    # The base-10 logarithm, -log kd/pk
    
    with open(index_labels_file, "r") as g:
        labels = np.array([
            float(line.split()[3]) for line in g.readlines() if line[0] != "#"
        ])
        
    with open(index_labels_file, "r") as g:
        pdbs = np.array([
            str(line.split()[0]) for line in g.readlines() if line[0] != "#"
        ])

    label_dict = {}
    for idx in range(len(pdbs)):
        label_dict[pdbs[idx]] = labels[idx]
    
    cutoff = 6.0

    # Use core-set as testing set
    # Use refined-set (excluding core-set) as training+validation set
    construct_graphs(data_dir, data_dir, "core-set", "test", label_dict, cutoff, None)
    construct_graphs(data_dir, data_dir, "refined-set", "train_val", label_dict, cutoff, "core-set")


if __name__ == "__main__":
    main()