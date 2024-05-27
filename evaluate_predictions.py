import os
import argparse
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
import pandas as pd
# ignore Bio python warnings
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--preds-path', type=str, default=None, help='Path to the predictions directory (*.trafl files)')
    args.add_argument('--templates-path', type=str, default=None, help='Path to the pdb templates directory')
    args.add_argument('--targets-path', type=str, default=None, help='Path to the pdb targets directory')
    args.add_argument('--output-name', type=str, default="rmsd.csv", help='Name of the output file')
    args.add_argument('--sim_rna', type=str, default=None, help='Path to the simRNA executable')
    args.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    return args.parse_args()

def generate_pdbs_from_trafl(trafl_path, targets_path, sim_rna_path, overwrite=False, pdb_postfix='.seq-000001.pdb', out_postfix='-000001_AA.pdb'):
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
    pass

def superimpose_pdbs(trafl_path, targets_path, out_postfix='-000001_AA.pdb'):
    pdbs = os.listdir(trafl_path)
    pdbs = [pdb for pdb in pdbs if pdb.endswith(out_postfix)]
    outs = []
    for pdb in tqdm(pdbs):
        base_name = pdb.replace(out_postfix, '')
        pdb_name = base_name + '.pdb'
        if os.path.exists(f"{targets_path}/{pdb_name}"):
            parser = PDBParser()
            ref_structure = parser.get_structure('ref', f"{targets_path}/{pdb_name}")
            ref_model = ref_structure[0]
            ref_atoms = [atom for atom in ref_model.get_atoms()]
            ref_coords = [atom.get_coord() for atom in ref_atoms]

            pred_structure = parser.get_structure('structure', f"{trafl_path}/{pdb}")
            model = pred_structure[0]
            atoms = [atom for atom in model.get_atoms()]
            coords = [atom.get_coord() for atom in atoms]

            sup = Superimposer()
            sup.set_atoms(ref_atoms, atoms)
            sup.apply(model.get_atoms())
            outs.append((pdb, round(sup.rms, 3)))
        else:
            print(f"Skipping {pdb} as the target pdb file: {targets_path}/{pdb_name} does not exist")
    return outs


def main():
    args = parse_args()
    generate_pdbs_from_trafl(args.preds_path, args.templates_path, args.sim_rna, args.overwrite)
    print("Superimposing...")
    outs = superimpose_pdbs(args.preds_path, args.targets_path)
    print("Results:")
    df = pd.DataFrame(outs, columns=['pdb', 'rms'])
    # sort df by rms column
    df = df.sort_values(by='rms', ascending=True)
    print(df.head())
    print(f"Mean RMSD: {df['rms'].mean()}, Median RMSD: {df['rms'].median()}")
    df.to_csv(args.output_name, index=False)
    print(f"Results saved to {args.output_name}")
    pass

if __name__ == "__main__":
    main()