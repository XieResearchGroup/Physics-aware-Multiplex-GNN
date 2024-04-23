#!/bin/bash

for file in /data/3d/bgsu-raw-pdb/*
do
    dirname=$file
    # if dirname is a directory
    basename=$(basename $dirname)
    echo $dirname

    if [ -d $dirname ]; then
        cp $dirname/*.pdb /data/3d/bgsu-pdbs-unpack/$basename.pdb
    fi
done