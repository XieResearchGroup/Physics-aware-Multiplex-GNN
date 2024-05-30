import os
import argparse
import math
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
import pandas as pd
# ignore Bio python warnings
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

from rnapolis.annotator import extract_secondary_structure
from rnapolis.parser import read_3d_structure

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--preds-path', type=str, default=None, help='Path to the predictions directory (*.trafl files)')
    args.add_argument('--templates-path', type=str, default=None, help='Path to the pdb templates directory')
    args.add_argument('--targets-path', type=str, default=None, help='Path to the pdb targets directory')
    args.add_argument('--output-name', type=str, default="rmsd.csv", help='Name of the output file')
    args.add_argument('--sim_rna', type=str, default=None, help='Path to the simRNA executable')
    args.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    return args.parse_args()

def generate_pdbs_from_trafl(trafl_path, targets_path, sim_rna_path, overwrite=False, pdb_postfix='.pdb', out_postfix='-000001_AA.pdb'):
    trafls = os.listdir(trafl_path)
    trafls = [trafl for trafl in trafls if trafl.endswith('.trafl')]
    for t in tqdm(trafls):
        base_name = t.replace('.trafl', '')
        pdb_name = base_name + pdb_postfix
        out_name = base_name + out_postfix
        if not overwrite and os.path.exists(os.path.join(trafl_path, out_name)):
            continue
        
        if os.path.exists(f"{targets_path}/{pdb_name}"):
            os.system(f"./run_SimRNA_trafl_to_pdb.sh {sim_rna_path} {targets_path}/{pdb_name} {trafl_path}/{t}")
        else:
            print(f"Skipping {t} as the target pdb file: {targets_path}/{pdb_name} does not exist")

def extract_2d_structure(pdb_path):
    with open(pdb_path, 'r') as f:
        structure3d = read_3d_structure(f, 1)
        structure2d = extract_secondary_structure(structure3d, 1)
    s2d = structure2d.dotBracket.split('>strand')
    if len(s2d) > 1:
        s2d = [s.strip().split('\n')[-1] for s in s2d]
        return "".join(s2d)
    else:
        return structure2d.dotBracket.split('\n')[-1]

def get_inf(s_pred, s_gt):
    assert len(s_pred) == len(s_gt), ValueError(f"Length of the predicted and ground truth sequences should be the same")
    s_pred = [1 if c != '.' else 0 for c in s_pred]
    s_gt = [1 if c != '.' else 0 for c in s_gt]
    tp = sum([1 for i in range(len(s_pred)) if s_pred[i] == 1 and s_gt[i] == 1])
    fp = sum([1 for i in range(len(s_pred)) if s_pred[i] == 1 and s_gt[i] == 0])
    fn = sum([1 for i in range(len(s_pred)) if s_pred[i] == 0 and s_gt[i] == 1])
    ppv = tp / (tp + fp) if tp + fp > 0 else 0
    sty = tp / (tp + fn) if tp + fn > 0 else 0
    inf = math.sqrt(ppv * sty)
    return inf

def superimpose_pdbs(trafl_path, targets_path, out_postfix='-000001_AA.pdb'):
    pdbs = os.listdir(trafl_path)
    pdbs = [pdb for pdb in pdbs if pdb.endswith(out_postfix)]
    outs = []
    for pdb in tqdm(pdbs):
        base_name = pdb.replace(out_postfix, '')
        pdb_name = base_name + '.pdb'
        if os.path.exists(f"{targets_path}/{pdb_name}"):
            parser = PDBParser()
            ref_2d_structure = extract_2d_structure(f"{targets_path}/{pdb_name}")
            ref_structure = parser.get_structure('ref', f"{targets_path}/{pdb_name}")
            ref_model = ref_structure[0]
            ref_atoms = [atom for atom in ref_model.get_atoms()]
            ref_coords = [atom.get_coord() for atom in ref_atoms]

            pred_2d_structure = extract_2d_structure(f"{trafl_path}/{pdb}")
            pred_structure = parser.get_structure('structure', f"{trafl_path}/{pdb}")
            model = pred_structure[0]
            atoms = [atom for atom in model.get_atoms()]
            coords = [atom.get_coord() for atom in atoms]

            sup = Superimposer()
            try:
                inf = get_inf(pred_2d_structure, ref_2d_structure)
                sup.set_atoms(ref_atoms, atoms) # if there is a missing P atom in some chain, then is should be removed from prediction to do superposition
                sup.apply(model.get_atoms())
                outs.append((pdb, round(sup.rms, 3), round(inf, 3)))
            except ValueError as e:
                print(f"Skipping {pdb}. The INF calculation failed")
                print(e)
                continue
            except:
                print(f"Skipping {pdb} as the superimposition failed")
                continue
        else:
            print(f"Skipping {pdb} as the target pdb file: {targets_path}/{pdb_name} does not exist")
    return outs


def main():
    args = parse_args()
    # out_postfix = '.seq-000001_AA.pdb'
    generate_pdbs_from_trafl(args.preds_path, args.templates_path, args.sim_rna, args.overwrite) #, out_postfix=out_postfix, pdb_postfix=".pdb")
    print("Superimposing...")
    outs = superimpose_pdbs(args.preds_path, args.targets_path) #, out_postfix=out_postfix)
    print("Results:")
    df = pd.DataFrame(outs, columns=['pdb', 'rms', 'inf'])
    # sort df by rms column
    df = df.sort_values(by='rms', ascending=True)
    print(df.head())
    print(f"Mean RMSD: {df['rms'].mean()}, Median RMSD: {df['rms'].median()}")
    print(f"Mean INF: {df['inf'].mean()}, Median INF: {df['inf'].median()}")
    df.to_csv(args.output_name, index=False)
    print(f"Results saved to {args.output_name}")
    pass

if __name__ == "__main__":
    main()