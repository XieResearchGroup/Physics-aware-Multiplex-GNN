import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem


def load_molecule(molecule_file):
    if ".mol2" in molecule_file:
        my_mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=True)
    elif ".sdf" in molecule_file:
        suppl = Chem.SDMolSupplier(str(molecule_file), sanitize=False, removeHs=True)
        my_mol = suppl[0]
    elif ".pdb" in molecule_file:
        my_mol = Chem.MolFromPDBFile(
            str(molecule_file), sanitize=False, removeHs=True)
    else:
        raise ValueError("Unrecognized file type for %s" % str(molecule_file))
    if my_mol is None:
        raise ValueError("Unable to read non None Molecule Object")
    xyz = get_xyz_from_mol(my_mol)
    return xyz, my_mol

def get_xyz_from_mol(mol):
    xyz = np.zeros((mol.GetNumAtoms(), 3))
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        xyz[i, 0] = position.x
        xyz[i, 1] = position.y
        xyz[i, 2] = position.z
    return (xyz)

def get_rms(molecule_file):
    with open(molecule_file) as f:
        for line in f:
            if 'TER' in line:
                break
        for line in f:
            cont = line.split()
            if cont[0] == 'rms':
                break
    return float(cont[-1])

def construct_graphs(data_dir, save_dir, data_name, save_name):
    print("Preprocessing", data_name)

    data_dir_full = os.path.join(data_dir, data_name)
    save_dir_full = os.path.join(save_dir, save_name, "raw")

    if not os.path.exists(save_dir_full):
        os.makedirs(save_dir_full)
       
    name_list = [x for x in os.listdir(data_dir_full)]

    for file_name in [save_name + '_node_labels.txt', save_name + '_graph_indicator.txt', 
                save_name + '_node_attributes.txt', save_name + '_graph_labels.txt',
                save_name + '_graph_names.txt']:
        if os.path.isfile(os.path.join(save_dir_full, file_name)):
            os.remove(os.path.join(save_dir_full, file_name))

    for i in tqdm(range(len(name_list))):
        name = name_list[i]
        rna_file = os.path.join(data_dir_full, name)
        
        rna_coords, rna_mol = load_molecule(rna_file)
        rna_label = get_rms(rna_file)

        rna_x = list()
        for atom_id in rna_mol.GetAtoms():
            rna_x.append(atom_id.GetAtomicNum())

        x_indices = [i for i,x in enumerate(rna_x) if (x == 6 or x == 7 or x == 8)] 
        rna_x = np.array([rna_x[i] for i in x_indices])
        rna_pos = np.array(rna_coords[x_indices])

        types = {
            6: 0,   #C
            7: 1,   #N
            8: 2,   #O
        }

        rna_x = np.array([types[x] for x in rna_x])

        name = np.array(name).reshape(-1, 1)

        # Generate files for loading graphs
        indicator = np.ones((rna_x.shape[0], 1)) * (i + 1)

        with open(os.path.join(save_dir_full, save_name + '_graph_indicator.txt'),'ab') as f:
            np.savetxt(f, indicator, fmt='%i', delimiter=', ')
        f.close()
    
        with open(os.path.join(save_dir_full, save_name + '_node_labels.txt'),'ab') as f:
            np.savetxt(f, rna_x, fmt='%i', delimiter=', ')
        f.close()
  
        with open(os.path.join(save_dir_full, save_name + '_node_attributes.txt'),'ab') as f:
            np.savetxt(f, rna_pos, fmt='%.3f', delimiter=', ')
        f.close()
        
        with open(os.path.join(save_dir_full, save_name + '_graph_labels.txt'),'ab') as f:
            np.savetxt(f, [rna_label], fmt='%.3f', delimiter=', ')
        f.close()

        with open(os.path.join(save_dir_full, save_name + '_graph_names.txt'),'ab') as f:
            np.savetxt(f, name, fmt='%s', delimiter=', ')
        f.close()


def main():
    data_dir = os.path.join(".", "data", "RNA-Puzzles", "classics_train_val")
    save_dir = os.path.join(".", "data", "RNA-Puzzles")

    construct_graphs(data_dir, save_dir, "example_train", "train")
    construct_graphs(data_dir, save_dir, "example_val", "val")
    

if __name__ == "__main__":
    main()