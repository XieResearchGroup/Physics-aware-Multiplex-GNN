import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import pickle
import Bio
from Bio.PDB import PDBParser

ATOM_TYPES = {
            'C': 0,   #C
            'N': 1,   #N
            'O': 2,   #O
            'P': 3,   #P
        }

REV_ATOM_TYPES = {v: k for k, v in ATOM_TYPES.items()}

RESIDUES = {
    'A': 0,
    'G': 1,
    'U': 2,
    'C': 3,
}
REV_RESIDUES = {v: k for k, v in RESIDUES.items()}

COARSE_GRAIN_MAP = {
        'A': ["P", "C4'", "N9", "C2", "C6"],
        'G': ["P", "C4'", "N9", "C2", "C6"],
        "U": ["P", "C4'", "N1", "C2", "C4"],
        "C": ["P", "C4'", "N1", "C2", "C4"],
    }

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
    residues_names = []
    c4_prime = []
    c2 = []
    c4_or_c6 = []
    n1_or_n9 = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.get_coord())
                    atoms_elements.append(atom.element)
                    atoms_names.append(atom.get_name())
                    residues_names.append(residue.get_resname())
                    c4_prime.append(atom.get_name() == "C4'")
                    c2.append(atom.get_name() == "C2")
                    c4_or_c6.append(atom.get_name() == "C4" or atom.get_name() == "C6")
                    n1_or_n9.append(atom.get_name() == "N1" or atom.get_name() == "N9")
    return np.array(coords), atoms_elements, atoms_names, residues_names, c4_prime, c2, c4_or_c6, n1_or_n9

def get_xyz_from_mol(mol):
    xyz = np.zeros((mol.GetNumAtoms(), 3))
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        position = conf.GetAtomPosition(i)
        xyz[i, 0] = position.x
        xyz[i, 1] = position.y
        xyz[i, 2] = position.z
    return (xyz)

def get_coarse_grain_mask(data, residues):
    coarse_atoms = [COARSE_GRAIN_MAP[x] for x in residues]
    mask = [True if atom in coars_atoms else False for atom, coars_atoms in zip(data['symbols'], coarse_atoms)]
    return np.array(mask)

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
        
        # if rna_file exists, skip
        if os.path.exists(os.path.join(save_dir_full, name.replace(".pdb", ".pkl"))):
            continue
        
        try:
            rna_coords, elements, atoms_symbols, residues_names, c4_primes, c2, c4_or_c6, n1_or_n9 = load_with_bio(rna_file)
        except ValueError:
            print("Error reading molecule", rna_file)
            continue
        except Bio.PDB.PDBExceptions.PDBConstructionException as e:
            print("Error reading molecule (invalid or missing coordinate)", rna_file)
            continue

        x_indices = [i for i,x in enumerate(elements) if (x != 'H' and x != 'X')] # Remove Hydrogen, ions, etc. Keep only C, N, O, P
        elements = [elements[i] for i in x_indices]
        atoms_symbols = [atoms_symbols[i] for i in x_indices]
        residues_names = [residues_names[i] for i in x_indices]
        c4_primes = [c4_primes[i] for i in x_indices]
        c2 = [c2[i] for i in x_indices]
        c4_or_c6 = [c4_or_c6[i] for i in x_indices]
        n1_or_n9 = [n1_or_n9[i] for i in x_indices]
        rna_pos = np.array(rna_coords[x_indices])

        rna_x = np.array([ATOM_TYPES[x] for x in elements]) # Convert atomic numbers to types
        residues_x = np.array([RESIDUES[x] for x in residues_names]) # Convert residues to types

        assert len(rna_x) == len(rna_pos) == len(atoms_symbols) == len(residues_x) == len(c4_primes)

        # Assign a unique label to each graph (RNA molecule).
        indicator = np.ones((rna_x.shape[0], 1)) * (i + 1)
        
        data = {}
        data['atoms'] = rna_x
        data['pos'] = rna_pos
        data['symbols'] = atoms_symbols
        data['indicator'] = indicator
        data['name'] = name
        data['residues'] = residues_x
        data['c4_primes'] = np.array(c4_primes)
        data['c2'] = np.array(c2)
        data['c4_or_c6'] = np.array(c4_or_c6)
        data['n1_or_n9'] = np.array(n1_or_n9)
        crs_gr_mask = get_coarse_grain_mask(data, residues_names)
        data['crs-grain-mask'] = crs_gr_mask

        with open(os.path.join(save_dir_full, name.replace(".pdb", ".pkl")), "wb") as f:
            pickle.dump(data, f)


def main():
    # data_dir = os.path.join(".", "data", "RNA-PDB")
    data_dir = "../input_data/diffusion-desc-pdbs/"
    save_dir = os.path.join(".", "data", "RNA-PDB")
    
    # construct_graphs(data_dir, save_dir, "bgsu-pdbs-unpack" , "train-raw-pkl")
    construct_graphs(data_dir, save_dir, "desc-pdbs" , "desc-pkl")
    # construct_graphs(data_dir, save_dir, "train-pdb" , "train-pkl")
    # construct_graphs(data_dir, save_dir, "val-pdb", "val-pkl")

if __name__ == "__main__":
    main()