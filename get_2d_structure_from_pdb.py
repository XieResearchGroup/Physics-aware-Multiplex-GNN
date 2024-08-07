from rnapolis.annotator import extract_secondary_structure
from rnapolis.parser import read_3d_structure

# struct_3d = "/home/mjustyna/data/desc-pdbs/6YW5_1_aa_A_398_G.pdb"
struct_3d = "fold_6yw5_398_g_model_0.cif"


with open(struct_3d) as f:
    structure3d = read_3d_structure(f, 1)
    structure2d = extract_secondary_structure(structure3d, 1)

print(structure2d.dotBracket.split('\n'))