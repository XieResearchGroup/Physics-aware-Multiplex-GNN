#!/usr/bin/bash -i
#SBATCH -p hgx
#SBATCH --gres=gpu:08
#SBATCH -t 168:00:00

source ~/.bashrc
conda activate gnn

torchrun \
    --standalone \
    --nproc_per_node=8 \
    main_rna_pdb.py --dataset RNA-PDB --epoch=801 --batch_size=32 --dim=512 --n_layer=4 --lr=2e-4 --timesteps=4000 --mode=coarse-grain --knn=10 --wandb