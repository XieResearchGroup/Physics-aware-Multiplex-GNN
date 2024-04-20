import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import pickle


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

def construct_graphs(data_dir, save_dir, data_name, save_name):
    print("Preprocessing", data_name)

    data_dir_full = os.path.join(data_dir, data_name)
    save_dir_full = os.path.join(save_dir, save_name)

    if not os.path.exists(save_dir_full):
        os.makedirs(save_dir_full)
       
    name_list = [x for x in os.listdir(data_dir_full)]

    for i in tqdm(range(len(name_list))):
        name = name_list[i]
        rna_file = os.path.join(data_dir_full, name)
        
        try:
            rna_coords, rna_mol = load_molecule(rna_file)
        except ValueError:
            print("Error reading molecule", rna_file)
            continue

        rna_x = list()
        for atom_id in rna_mol.GetAtoms():
            rna_x.append(atom_id.GetAtomicNum())

        x_indices = [i for i,x in enumerate(rna_x) if (x == 6 or x == 7 or x == 8)] # Remove Hydrogen, ions, etc. Keep only C, N, O
        rna_x = np.array([rna_x[i] for i in x_indices])
        rna_pos = np.array(rna_coords[x_indices])

        types = {
            6: 0,   #C
            7: 1,   #N
            8: 2,   #O
        }

        rna_x = np.array([types[x] for x in rna_x]) # Convert atomic numbers to types

        # Assign a unique label to each graph (RNA molecule).
        indicator = np.ones((rna_x.shape[0], 1)) * (i + 1)
        data = {}
        data['atoms'] = rna_x
        data['pos'] = rna_pos
        data['indicator'] = indicator
        data['name'] = name

        with open(os.path.join(save_dir_full, name.replace(".pdb", ".pkl")), "wb") as f:
            pickle.dump(data, f)


def main():
    data_dir = os.path.join(".", "data", "RNA-PDB")
    save_dir = os.path.join(".", "data", "RNA-PDB")

    construct_graphs(data_dir, save_dir, "train-pdb", "train-pkl")
    construct_graphs(data_dir, save_dir, "val-pdb", "val-pkl")

if __name__ == "__main__":
    main()