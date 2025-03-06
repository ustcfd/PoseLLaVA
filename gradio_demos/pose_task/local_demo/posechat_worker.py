"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
import os
import requests
import torch
from functools import partial
from transformers import StoppingCriteria


from mllm.utils.log_utils import (LOGDIR, build_logger, server_error_msg)


from mllm.utils.mm_utils import  tokenizer_image_token, process_images
from mllm.models.constants.llava_constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN
from transformers import TextIteratorStreamer
from threading import Thread
import time

import os.path as osp
import cv2
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
from loaders.model_loaders import LOADERS
from PIL import Image
from io import BytesIO
import base64

GB = 1 << 30
worker_id = str(uuid.uuid4())[:3]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")

def create_temp_folder(directory_path):
    temp_folder_path = os.path.join(directory_path, 'temp')
    # 检查temp文件夹是否已经存在
    if not os.path.exists(temp_folder_path):
        # 如果不存在，创建temp文件夹
        os.makedirs(temp_folder_path)
    return temp_folder_path
  
def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
 
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
             
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

class ModelWorker:
    def __init__(self, 
                 model_family_id, 
                 model_name_or_path, 
                 compute_dtype = torch.float16,
                 device = 'cuda'):
        self.worker_id = worker_id
        self.device = device
        self.compute_dtype = compute_dtype
        logger.info(f"Loading the model {model_family_id} on worker {worker_id} ...")
        loader = LOADERS[model_family_id](
            model_local_path= model_name_or_path,
            compute_dtype= compute_dtype,
            use_flash_attn= True,
            device_map='auto')
        
        self.model, self.tokenizer  = loader.load(load_model=False)
        self.image_processor = self.model.get_vision_tower().image_processor


        # self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        #     model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device)
        
        self.is_multimodal = True
        self.temp_dir = create_temp_folder(directory_path=LOGDIR)
        smpl_path = "pose_utils/visualizer/support_dir"
        self.body_model_config = dict(type='smpl', model_path=smpl_path)
        
        
    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
       
        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <|image|> tokens in prompt")

                images = [load_image_from_base64(image) for image in images]
                images = process_images(images, image_processor, model.config)

                if type(images) is list:
                    images = [image.to(self.model.device, dtype=self.compute_dtype) for image in images]
                else:
                    images = images.to(self.model.device, dtype=self.compute_dtype)

                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = IMG_START_TOKEN + replace_token + IMG_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
            image_args = {"pixel_values": images}
        else:
            images = None
            image_args = {}

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)
        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return
        
        thread = CustomThread(target=model.chat, kwargs=dict(
            input_ids =input_ids,
            # do_sample=do_sample,
            # temperature=temperature,
            do_sample=False,
            # temperature=0.6,
            num_beams=1,
            # top_p=top_p,
            length_penalty=1,
            repetition_penalty=1.5,
            max_length =500,
            # max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ))
        thread.start()
        
        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode()

        _ , return_batch_pose = thread.join()
        return_pose = return_batch_pose[0]
        if return_pose is not None:
            # smpl2body_visualizer = Body_pose_render(
            #     bm_fname_path= 'pose_utils/visualizer/support_dir/smplh/male_model.npz', 
            #     dmpl_fname_path= 'pose_utils/visualizer/support_dir/dmpls/male_model.npz', 
            #     device= self.device
            # )
            # temp_pose_img_path = smpl2body_visualizer.save_pose(return_pose, self.temp_dir)
            return_pose = return_pose.cpu().numpy()
            return_img_tensors = visualize_smpl_pose(
                poses=return_pose,
                output_path = self.temp_dir,
                resolution=(1024, 1024),
                verbose= True,
                batch_size=1, 
                overwrite = True,
                return_tensor =True,
                body_model_config=self.body_model_config)
            
            return_img_tensors = return_img_tensors.cpu().numpy()
            batch_size =  return_img_tensors.shape[0]
            assert batch_size==1, "The inference batch size is not equal to 1"
            curr_img = return_img_tensors[0]
            temp_img_name = str(uuid.uuid4())[:5]
            temp_pose_img_path = osp.join(self.temp_dir, f'{temp_img_name}.png')
            cv2.imwrite(temp_pose_img_path, curr_img)
            
            yield json.dumps({"text": temp_pose_img_path, "error_code": 1}).encode()
        
    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
                
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() 
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode()
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode()


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)