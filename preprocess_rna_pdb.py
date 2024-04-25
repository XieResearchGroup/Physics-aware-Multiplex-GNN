import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import pickle
from Bio.PDB import PDBParser

ATOM_TYPES = {
            'C': 0,   #C
            'N': 1,   #N
            'O': 2,   #O
            'P': 3,   #P
        }

REV_ATOM_TYPES = {v: k for k, v in ATOM_TYPES.items()}

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

def load_with_bio(molecule_file):
    parser = PDBParser()
    structure = parser.get_structure("rna", molecule_file)
    coords = []
    atoms_elements = []
    atoms_names = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
                    atoms_elements.append(atom.element)
                    atoms_names.append(atom.get_name())
                    assert len(coords) == len(atoms_elements) == len(atoms_names)
    return np.array(coords), atoms_elements, atoms_names

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
    name_list = [x for x in name_list if ".pdb" in x]

    for i in tqdm(range(len(name_list))):
        name = name_list[i]
        rna_file = os.path.join(data_dir_full, name)
        
        try:
            rna_coords, elements, symbols = load_with_bio(rna_file)
        except ValueError:
            print("Error reading molecule", rna_file)
            continue

        x_indices = [i for i,x in enumerate(elements) if (x != 'H' and x != 'X')] # Remove Hydrogen, ions, etc. Keep only C, N, O, P
        elements = [elements[i] for i in x_indices]
        symbols = [symbols[i] for i in x_indices]
        rna_pos = np.array(rna_coords[x_indices])

        rna_x = np.array([ATOM_TYPES[x] for x in elements]) # Convert atomic numbers to types

        assert len(rna_x) == len(rna_pos) == len(symbols)

        # Assign a unique label to each graph (RNA molecule).
        indicator = np.ones((rna_x.shape[0], 1)) * (i + 1)
        data = {}
        data['atoms'] = rna_x
        data['pos'] = rna_pos
        data['symbols'] = symbols
        data['indicator'] = indicator
        data['name'] = name

        with open(os.path.join(save_dir_full, name.replace(".pdb", ".pkl")), "wb") as f:
            pickle.dump(data, f)


def main():
    # data_dir = os.path.join(".", "data", "RNA-PDB")
    data_dir = "/data/3d/"
    save_dir = os.path.join(".", "data", "RNA-PDB")
    
    construct_graphs(data_dir, save_dir, "bgsu-pdbs-unpack" , "train-raw-pkl")
    # construct_graphs(data_dir, save_dir, "train-pdb" , "train-pkl")
    # construct_graphs(data_dir, save_dir, "val-pdb", "val-pkl")

if __name__ == "__main__":
    main()