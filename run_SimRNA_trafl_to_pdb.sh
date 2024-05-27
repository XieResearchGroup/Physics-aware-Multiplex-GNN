#!/bin/bash

# Set path to SIM_RNA_PATH as the first argument
SIM_RNA_PATH=$1

cd $SIM_RNA_PATH
SimRNA_trafl2pdbs $2 $3 0: AA