import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
print(parent_dir)
sys.path.append(parent_dir)
# 获取当前脚本文件所在目录
import torch
import uuid
from threading import Thread
import litserve as ls

from transformers import TextIteratorStreamer
from mllm.utils.mm_utils import process_images, tokenizer_image_token, fetch_image, KeywordsStoppingCriteria, load_image_from_base64
from mllm.models.constants.llava_constants import IMAGE_TOKEN_INDEX , DEFAULT_IMAGE_TOKEN
import json
from mllm.utils.log_utils import server_error_msg, LOGDIR, create_temp_folder
import os.path as osp
import cv2
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
from gradio_demos.api.config import api_cfg
from loaders.model_loaders import LOADERS

default_model_id = 'posechat'
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
    
class PoseChatAPI(ls.LitAPI):
    def setup(self, device, model_family_id=default_model_id):
        self.compute_dtype = torch.float16
        model_path = api_cfg[model_family_id]['model_path']
        print(f"Loading model from {model_path}")
        if not os.path.exists(model_path):
            raise ValueError(f"Invalid model ID: {model_family_id}")
        
        # Load model and tokenizer by model_args.model_family_id
        loader = LOADERS[model_family_id](
            model_local_path= model_path,
            model_max_length = 2048,
            compute_dtype= self.compute_dtype ,
            use_flash_attn= True,
            device_map= device,     
        )

        # Load pretrained model, tokenizer, and image processor
        self.model, self.tokenizer, self.image_processor = loader.load(load_lora_model=False)
        self.model_config = self.model.config
        # self.processor = LlavaProcessor.from_pretrained(model_path)
        # self.image_processor = image_processor
        self.streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.device = device
        self.model_id = model_family_id

        self.temp_dir = create_temp_folder(directory_path=LOGDIR)
        smpl_path = "pose_utils/visualizer/support_dir"
        self.body_model_config = dict(type='smpl', model_path=smpl_path)
        

    def decode_request(self, request):
        if request['model_id'] != self.model_id:
            self.setup(self.device, request['model'])

        prompt = request["prompt"]
        ori_prompt = prompt
        images = request.get("images", None)
        num_image_tokens = 0
        if images is not None and len(images) > 0:
            if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                raise ValueError("Number of images does not match number of <|image|> tokens in prompt")
            
            # images = [fetch_image(image) for image in images]
            images = [load_image_from_base64(image) for image in images]
            images = process_images(images, self.image_processor, self.model_config)
            if type(images) is list:
                images = [image.to(self.model.device, dtype=self.compute_dtype) for image in images]
            else:
                images = images.to(self.model.device, dtype=self.compute_dtype)

            num_image_tokens = prompt.count(DEFAULT_IMAGE_TOKEN) * self.model.get_vision_tower().num_patches  
        else:   
            images = None
         
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device) 
        # temperature = float(request.get("temperature", 1.0))
        # top_p = float(request.get("top_p", 1.0))
        max_context_length = getattr(self.model_config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(request.get("max_new_tokens", 256)), 1024)
        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)
        
        stop_str = request.get("stop", None)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        model_inputs = {
            "input_ids": input_ids,
            "pixel_values": images,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.5,
            "stopping_criteria":[stopping_criteria],
            "num_beams" : 1,
            "length_penalty":1,
            "stop_str": stop_str,
            "org_prompt": ori_prompt,    
        }
       
        return model_inputs
    
    def predict(self, model_inputs):
        try:
            generated_text = model_inputs.pop("org_prompt")
            stop_str = model_inputs.pop("stop_str", None)

            generation_kwargs = dict(
                **model_inputs,
                streamer=self.streamer,
            )
            thread = CustomThread(target=self.model.chat, kwargs=generation_kwargs)
            thread.start()
        
            for new_text in self.streamer:
                generated_text += new_text
                if generated_text.endswith(stop_str):
                    generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode()
            
            _ , return_batch_pose = thread.join()
            return_pose = return_batch_pose[0]
            if return_pose is not None:
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


    def encode_response(self, output_stream):
        for output in output_stream:
            yield output
       
if __name__ == "__main__":
     
    port_id = api_cfg[default_model_id]['port']
    api = PoseChatAPI()
    server = ls.LitServer(api,stream=True)
    server.run(port=port_id)
    '''
    url = "http://0.0.0.0:8899/predict"
    resp = requests.post(url, json={"input": "Hello world"}, headers=None, stream=True)
    for line in resp.iter_content(5000):
        if line:
            print(line.decode("utf-8"))
    '''