#!/usr/bin/bash -i
#SBATCH -p hgx
#SBATCH --gres=gpu:01
#SBATCH -t 48:00:00

source ~/.bashrc
conda activate gnn

python main_rna_pdb_single.py --dataset RNA-bgsu-j3 --epoch=20 --batch_size=32 --dim=256 --n_layer=8 --lr=1e-5 --timesteps=5000 --mode=coarse-grain --knn=20 --wandb --lr-step=200
