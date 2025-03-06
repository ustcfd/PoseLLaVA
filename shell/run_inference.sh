#!/bin/bash
source ~/switch_cuda.sh 11.8
export PYTHONPATH=`pwd`:$PYTHONPATH

python evaluation/eval_posellava.py \
    --model_family_id posellava \
    --model_path /yq/fengd/finetuning_model/stage3_model_weight \
    --eval_data_path ds_config/eval_ds.json \
    --device cuda:0 
