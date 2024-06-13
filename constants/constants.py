BACKBONE_ATOMS = ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']

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

KEEP_ELEMENTS = ['C', 'N', 'O', 'P']

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

DOT_OPENINGS = ['(', '[', '{', '<', 'A', 'B', 'C', 'D']
DOT_CLOSINGS_MAP = {
    ')': '(',
    ']': '[',
    '}': '{',
    '>': '<',
    'a': 'A',
    'b': 'B',
    'c': 'C',
    'd': 'D'
}

RIBOSE = ['C1\'', 'C2\'', 'C3\'', 'C4\'', 'O4\'', 'C5\'', 'O5\'', 'P']
BASES = {
    "A": ['N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'],
    "G": ['N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'],
    "C": ['N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'],
    "U": ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6'],
}