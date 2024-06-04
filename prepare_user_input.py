import argparse
import os
from tqdm import tqdm

from preprocess_rna_pdb import construct_graphs

SIM_RNA = "/home/mjustyna/software/SimRNA_64bitIntel_Linux_staticLibs_withoutOpenMP"
OUTPUT_DIR = "dotseq_out"
OUT_PDBS = "pdbs"

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare user input')
    parser.add_argument('--input-dir', type=str, help='Input file')
    return parser.parse_args()

def read_dotseq_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    seq = lines[1].strip()
    dot = [l.replace(" ", "") for l in lines]
    return dot, seq

def save(file_path, file_name, content):
    os.makedirs(file_path, exist_ok=True)
    with open(os.path.join(file_path, file_name), 'w') as f:
        f.writelines(content)

def main():
    args = parse_args()
    print(f"Preparing files from: {args.input_dir}")
    if os.path.exists(args.input_dir):
        files = os.listdir(args.input_dir)
    else:
        raise FileNotFoundError(f"Directory {args.input_dir} not found")
    files = [f for f in files if f.endswith('.dotseq')]
    for f in tqdm(files):
        dot, seq = read_dotseq_file(os.path.join(args.input_dir, f))
        save(os.path.join(args.input_dir, OUTPUT_DIR), f.replace('.dotseq', '.dot'), dot)
        save(os.path.join(args.input_dir, OUTPUT_DIR), f.replace('.dotseq', '.seq'), seq)
    
    os.makedirs(os.path.join(args.input_dir, OUT_PDBS), exist_ok=True)
    # Generate initial structures with SimRNA
    for f in files:
        inp = os.path.join(args.input_dir, OUTPUT_DIR, f.replace('.dotseq', '.seq'))
        inp = os.path.abspath(inp)
        pdbs_out = os.path.join(os.path.abspath(args.input_dir), OUT_PDBS)
        os.system(f"./run_SimRNA.sh {SIM_RNA} {inp} {pdbs_out}")
    # Fix the names
    pdbs_files = os.listdir(os.path.join(args.input_dir, OUT_PDBS))
    for f in pdbs_files:
        os.rename(os.path.join(args.input_dir, OUT_PDBS, f), os.path.join(args.input_dir, OUT_PDBS, f.replace(".seq-000001", "")))

    # Construct graphs and store as pickle    
    seq_dir = os.path.join(args.input_dir, OUTPUT_DIR)
    pdbs_dir = os.path.join(args.input_dir, OUT_PDBS)
    save_dir = save_dir = os.path.join(".", "data", "user_inputs")
    construct_graphs(seq_dir, pdbs_dir, save_dir, save_name="test-pkl", extended_dotbracket=False)


if __name__ == '__main__':
    main()