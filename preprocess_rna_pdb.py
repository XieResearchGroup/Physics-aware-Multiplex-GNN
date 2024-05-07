import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import pickle
import Bio
from Bio.PDB import PDBParser
from rnapolis.annotator import extract_secondary_structure
from rnapolis.parser import read_3d_structure
# from torch_geometric.data import Data

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

RESIDUE_CONNECTION_GRAPH = [
    [0, 1], # P -> C4'
    [1, 0], # C4' -> P
    [1, 2], # C4' -> N
    [2, 1], # N -> C4'
    [2, 3], # N -> C2
    [3, 2], # C2 -> N
    [3, 4], # C2 -> C4/6
    [4, 3], # C4/6 -> C2
    [4, 2], # C4/6 -> N
    [2, 4], # N -> C4/6
]

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

def get_coarse_grain_mask(symbols, residues):
    coarse_atoms = [COARSE_GRAIN_MAP[x] for x in residues]
    mask = [True if atom in coars_atoms else False for atom, coars_atoms in zip(symbols, coarse_atoms)]
    return np.array(mask)

def get_edges_in_COO(data:dict, seq_segments:list[str], bpseq: list[tuple[int, int]] = None):
    # Order of encoded atoms: "P", "C4'", "Nx", "C2", "Cx"
    edges = []
    edge_type = [] # True: covalent, False: other interaction
    segments_lengs = [len(x) for x in seq_segments]
    segments_lengs = np.cumsum(segments_lengs) # get the end index of each segment

    p = data['atoms'] == ATOM_TYPES['P']
    c4_prime = data['c4_primes']
    c2 = data['c2']
    c4_or_c6 = data['c4_or_c6']
    n1_or_n9 = data['n1_or_n9']
    nodes_indecies = np.arange(data['atoms'].shape[0])
    combined = np.stack([p, c4_prime, c2, c4_or_c6, n1_or_n9], axis=1)

    added = 0
    for index in np.concatenate([np.array([0]), segments_lengs[:-1]]):
        if p[index*5 - added] == False:
            combined = np.concatenate([combined[:index*5], np.array([[True, False, False, False, False]]), combined[index*5:]])
            nodes_indecies = np.concatenate([nodes_indecies[:index*5], np.array([nodes_indecies[index*5]]), nodes_indecies[index*5:]])
            added += 1
    
    combined = combined.reshape((-1, 5, 5))
    nodes_indecies = nodes_indecies.reshape((-1, 5))
    comb_arg_max = np.argmax(combined, axis=2) # sometimes the order of atoms is 0,1,2,3,4, and sometimes it's different
    for res_ni, res_arg_max in zip(nodes_indecies, comb_arg_max): # create edges in each residue
        for i, j in RESIDUE_CONNECTION_GRAPH:
            edge = [res_ni[res_arg_max[i]], res_ni[res_arg_max[j]]]
            if edge[0] == edge[1]: # remove self loops, effect of adding missing P atoms
                continue
            edges.append(edge)
            edge_type.append(True)

    # connect residues
    for i in range(1, len(nodes_indecies)):
        if i in segments_lengs:
            continue
        prev_c4p = nodes_indecies[i-1][comb_arg_max[i-1][1]] # C4' atom index in previous residue
        curr_p = nodes_indecies[i][comb_arg_max[i][0]] # P atom index in current residue
        edges.append([prev_c4p, curr_p])
        edges.append([curr_p, prev_c4p])
        edge_type.extend([True, True])

    # edges based on bpseq (2D structure)
    if bpseq is not None:
        for pair in bpseq:
            for i in range(2, 5): # atoms: N, C2, Cx
                p1 = nodes_indecies[pair[0]][comb_arg_max[pair[0]][i]] # atom i (e.g. N) connect with the corresponding atom in the paired residue
                p2 = nodes_indecies[pair[1]][comb_arg_max[pair[1]][i]]
                edges.append([p1, p2])
                edges.append([p2, p1])
                edge_type.extend([False, False])
    assert len(edges) == len(edge_type)
    return edges, edge_type

def read_seq_segments(seq_file):
    with open(seq_file, "r") as f:
        seq = f.readline()
    return seq.strip().split()

def bpseq_to_res_ids(bpseq):
    bpseq = bpseq.split("\n")
    bpseq = [x.split() for x in bpseq]
    bpseq = [(int(x[0])-1, int(x[2])-1) for x in bpseq if int(x[2]) != 0 and int(x[0]) < int(x[2])] # -1, because the indices in bpseq are 1-based, and we need 0-based (numpy indicies)
    return bpseq

def construct_graphs(seq_dir, pdbs_dir, save_dir, save_name):
    save_dir_full = os.path.join(save_dir, save_name)

    if not os.path.exists(save_dir_full):
        os.makedirs(save_dir_full)
       
    name_list = [x for x in os.listdir(seq_dir)]
    name_list = [x for x in name_list if ".seq" in x]

    for i in tqdm(range(len(name_list))):
        name = name_list[i]
        seq_segments = read_seq_segments(os.path.join(seq_dir, name))
        name = name.replace(".seq", ".pdb")
        rna_file = os.path.join(pdbs_dir, name)
        
        # if rna_file exists, skip
        # if os.path.exists(os.path.join(save_dir_full, name.replace(".pdb", ".pkl"))):
        #     continue
        
        try:
            rna_coords, elements, atoms_symbols, residues_names, c4_primes, c2, c4_or_c6, n1_or_n9 = load_with_bio(rna_file)
        except ValueError:
            print("Error reading molecule", rna_file)
            continue
        except Bio.PDB.PDBExceptions.PDBConstructionException as e:
            print("Error reading molecule (invalid or missing coordinate)", rna_file)
            continue

        with open(rna_file) as f:
            structure3d = read_3d_structure(f, 1)
            structure2d = extract_secondary_structure(structure3d, 1)
        res_pairs = bpseq_to_res_ids(structure2d.bpseq)

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

        crs_gr_mask = get_coarse_grain_mask(atoms_symbols, residues_names)

        data = {}
        data['atoms'] = rna_x[crs_gr_mask]
        data['pos'] = rna_pos[crs_gr_mask]
        data['symbols'] = np.array(atoms_symbols)[crs_gr_mask]
        # data['indicator'] = graph_indicator[crs_gr_mask]
        data['name'] = name
        data['residues'] = residues_x[crs_gr_mask]
        data['c4_primes'] = np.array(c4_primes)[crs_gr_mask]
        data['c2'] = np.array(c2)[crs_gr_mask]
        data['c4_or_c6'] = np.array(c4_or_c6)[crs_gr_mask]
        data['n1_or_n9'] = np.array(n1_or_n9)[crs_gr_mask]
        edges, edge_type = get_edges_in_COO(data, seq_segments, bpseq=res_pairs)
        data['edges'] = edges
        data['edge_type'] = edge_type

        with open(os.path.join(save_dir_full, name.replace(".pdb", ".pkl")), "wb") as f:
            pickle.dump(data, f)


def main():
    # data_dir = "/home/mjustyna/data/test_structs/"
    # seq_dir = os.path.join(data_dir, "seqs")
    # pdbs_dir = os.path.join(data_dir, "pdbs")
    data_dir = "/home/mjustyna/data/"
    seq_dir = os.path.join(data_dir, "sim_desc")
    pdbs_dir = os.path.join(data_dir, "desc-pdbs")
    save_dir = os.path.join(".", "data", "RNA-PDB")
    
    construct_graphs(seq_dir, pdbs_dir, save_dir, "desc-pkl")
    # construct_graphs(seq_dir, pdbs_dir, save_dir, "test-pkl")

if __name__ == "__main__":
    main()