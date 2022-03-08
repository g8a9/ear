#!/bin/bash

# To make everything work as expected, run from the parent directory.

if test $# -ne 3; then
	echo "Specify input folder containing models and output folder"
	exit -1
fi

modeldir=$1
out_folder=$2
src_tokenizer=$3

for m in $modeldir/*; do
    python evaluate_model.py \
        --dataset miso-ita-synt \
        --model_path $m \
        --subgroups_path ./data/AMI2020_test_identityterms.txt \
        --out_folder $out_folder \
        --src_tokenizer $src_tokenizer \
        --n_jobs 8
done
