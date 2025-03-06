#!/bin/bash
source ~/switch_cuda.sh 11.8
export PYTHONPATH=`pwd`:$PYTHONPATH

OUTPUT_DIR=work_dirs/test
Pretained_model_path=/datanfs/fengd/posegpt_dataset/llava-v1.6-mistral-7b
# Pretained_model_path=/24T/fd/posellava_demo/model_checkpoints/text_and_image_and_pose_data_mistral_finetuning
Pretained_vision_path=/datanfs/fengd/posegpt_dataset/mistral_clip-vit-large-patch14-336
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACC=2

deepspeed mllm/train/train_posellava.py \
    --deepspeed "shell/script/zero_stage2_config.json" \
    --model_family_id "posellava" \
    --use_custom_trainer True \
    --model_name_or_path ${Pretained_model_path} \
    --vision_path ${Pretained_vision_path} \
    --conv_style "llava_v1" \
    --output_dir ${OUTPUT_DIR} \
    --meta_path "ds_config/test_ds.json" \
    --overwrite_output_dir False \
    --drop_path_rate 0.0 \
    --freeze_backbone True \
    --use_llm_lora 16 \
    --bf16 True \
    --learning_rate 4e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --max_steps 20 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --evaluation_strategy "no" \
    --save_total_limit 1 \
    --save_strategy "no" \
    --save_steps 200 \
    --logging_steps 1 \
    --max_seq_length 4096 \
    --grad_checkpoint True \
    --group_by_length True \
2>&1 | tee "${OUTPUT_DIR}/training_log.txt"