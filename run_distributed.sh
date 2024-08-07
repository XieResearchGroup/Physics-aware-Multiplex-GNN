#!/usr/bin/bash -i
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -p hgx
#SBATCH --gres=gpu:08
#SBATCH -t 168:00:00
#SBATCH -w hgx2

source ~/.bashrc
conda activate gnn

torchrun \
    --standalone \
    --nproc_per_node=8 \
    main_rna_pdb.py --dataset RNA-PDB-clean --epoch=1201 --batch_size=16 --dim=256 --n_layer=6 --lr=1e-3 --timesteps=4000 --mode=coarse-grain --knn=20 --wandb --lr-step=30