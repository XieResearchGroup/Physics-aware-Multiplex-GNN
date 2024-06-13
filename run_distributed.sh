#!/usr/bin/bash -i
#SBATCH -p hgx
#SBATCH --gres=gpu:06
#SBATCH -t 168:00:00

source ~/.bashrc
conda activate gnn

torchrun \
    --standalone \
    --nproc_per_node=6 \
    main_rna_pdb.py --dataset RNA-PDB-clean --epoch=801 --batch_size=8 --dim=256 --n_layer=6 --lr=1e-3 --timesteps=5000 --mode=coarse-grain --knn=20 --wandb --lr-step=30