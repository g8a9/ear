#!/bin/bash

# To make everything work as expected, run from the parent directory.
# args: [model_directory]

if test $# -ne 5; then
    echo "Usage: ./test_folder.sh <in_dir> <out_dir> <dataset> <src_tokenizer> <ckpt_pattern>"
    exit -1
fi


in_dir=$1
out_dir=$2
dataset=$3
src_tokenizer=$4
ckpt_pattern=$5


for m in $in_dir/*; do
    echo "--------------------"
    echo "Evaluating $m"
    echo "--------------------"
    
    python evaluate_model.py \
        --dataset ${dataset} \
        --out_folder ${out_dir} \
        --model_path ${m} \
        --no_bias_metrics \
        --ckpt_pattern ${ckpt_pattern} \
        --src_tokenizer bert-base-uncased
done
