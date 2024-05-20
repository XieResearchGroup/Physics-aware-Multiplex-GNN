#!/usr/bin/bash -i
#SBATCH -p hgx
#SBATCH --gres=gpu:02
#SBATCH -t 168:00:00

source ~/.bashrc
conda activate gnn

torchrun \
    --standalone \
    --nproc_per_node=2 \
    main_rna_pdb.py --dataset RNA-PDB --epoch=401 --batch_size=64 --dim=256 --n_layer=4 --lr=1e-3 --timesteps=4000 --mode=coarse-grain --knn=10 --wandb