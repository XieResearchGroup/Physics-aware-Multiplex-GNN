#!/usr/bin/bash -i
#SBATCH -p hgx
#SBATCH --gres=gpu:08
#SBATCH -t 56:00:00

source ~/.bashrc
conda activate gnn

torchrun \
    --standalone \
    --nproc_per_node=8 \
    main_rna_pdb.py --dataset RNA-PDB-noncan --epoch=801 --batch_size=32 --dim=256 --n_layer=8 --lr=1e-3 --timesteps=5000 --mode=coarse-grain --knn=20 --wandb --lr-step=30