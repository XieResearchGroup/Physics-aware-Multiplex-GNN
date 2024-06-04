#!/bin/bash

SIM_RNA_PATH=$1
INP_PATH=$2
OUT_PATH=$3

cd $SIM_RNA_PATH
./SimRNA -s $INP_PATH
basename=$(basename $INP_PATH)
mv $basename* $OUT_PATH