#!/bin/bash

# To make everything work as expected, run from the parent directory.

if test $# -ne 3; then
    echo "Usage: ./evaluate_folder_madlibs_pattern.sh <in_dir> <out_dir> <ckpt_pattern>"
    exit -1
fi

modeldir=$1
out_folder=$2
ckpt_pattern=$3

# Test on Madlibs
# add madlibs89k in the next instruction if needed
for d in madlibs77k; do
    for m in $modeldir/*; do
        python evaluate_model.py \
            --dataset $d \
            --model_path $m \
            --subgroups_path ./data/bias_madlibs_data/adjectives_people.txt \
            --out_folder $out_folder \
            --n_jobs 8 \
            --src_tokenizer bert-base-uncased \
            --ckpt_pattern $ckpt_pattern

    done 
done
