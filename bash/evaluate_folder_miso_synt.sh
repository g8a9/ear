#!/bin/bash

# To make everything work as expected, run from the parent directory.
# args: [model_directory]

if test $# -ne 2; then
	echo "Specify input folder containing models and output folder"
	exit -1
fi

modeldir=$1

for m in $modeldir/*; do
    python evaluate_model.py --dataset miso_synt_test --model_path $m \
        --subgroups_path ./data/miso_it.txt \
        --out_folder $2
done
