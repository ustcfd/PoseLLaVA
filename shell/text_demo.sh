#!/bin/bash
source ~/switch_cuda.sh 11.8
export PYTHONPATH=`pwd`:$PYTHONPATH
# model-path：Model的权重路径

python -m gradio_demos.text_task.local_demo.posechat_imgpair_demo \
    --host 172.16.30.3 \
    --port 6791 \
    --model_family_id posechat \
    --model_path /yq/fengd/text_task_imgdata_test \
    --conv_style llava_v1 \
    --device cuda:0