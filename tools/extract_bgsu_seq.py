import os
from tqdm import tqdm
from rnapolis.annotator import extract_secondary_structure
from rnapolis.parser import read_3d_structure

def main():
    BGSU_PDB_PATH = '/home/mjustyna/data/bgsu-pdbs-unpack/'
    OUT = '/home/mjustyna/data/bgsu-seq/'
    files = os.listdir(BGSU_PDB_PATH)
    files = [f for f in files if f.endswith('.pdb')]
    os.makedirs(OUT, exist_ok=True)
    for fname in tqdm(files):
        with open(os.path.join(BGSU_PDB_PATH, fname), 'r') as f:
            s3d = read_3d_structure(f, 1)
            ss = extract_secondary_structure(s3d)
        dot = ss.dotBracket.split('\n')
        segment_split = dot[-1].find("()")+1
        segment1 = dot[1][:segment_split]
        segment2 = dot[1][segment_split:]
        seq_name = fname.replace('.pdb', '.seq')
        with open(os.path.join(OUT, seq_name), 'w') as f:
            f.write(f'{segment1} {segment2}')
        
        dot_name = fname.replace('.pdb', '.dot')
        with open(os.path.join(OUT, dot_name), 'w') as f:
            f.write(ss.dotBracket)


    pass

if __name__ == '__main__':
    main()