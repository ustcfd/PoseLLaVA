#!/bin/bash
source ~/switch_cuda.sh 11.8
export PYTHONPATH=`pwd`:$PYTHONPATH
# model-path：Model的权重路径，set model-path

python -m gradio_demos.pose_task.local_demo.posechat_web_server \
    --host 127.0.0.1 \
    --port 6790 \
    --model_family_id posellava \
    --model_path /24T/fd/posellava_demo/model_checkpoints/text_and_image_and_pose_data_mistral_finetuning \
    --conv_style llava_v1 \
    --device cuda:0