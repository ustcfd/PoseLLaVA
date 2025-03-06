#!/bin/bash
source ~/switch_cuda.sh 11.8
export PYTHONPATH=`pwd`:$PYTHONPATH
# model-path：Model的权重路径

#  首先运行下列命令启动服务端
# python gradio_demos/api/posechat_server.py 

python -m gradio_demos.pose_task.posechat_web_client \
    --host 172.16.30.3 \
    --port 6790 \
    --model_family_id posechat \
    --conv_style llava_v1 \
    --device cuda:0