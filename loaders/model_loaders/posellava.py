from typing import Tuple

from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, PreTrainedTokenizer, AutoConfig
from . import register_loader
from .base import BaseModelLoader
import json
from mllm.models.llava import PoseLlavaForCausalLM 
from peft import PeftModel
import os
import torch
from mllm.models.constants.llava_constants import DEFAULT_IMAGE_TOKEN, SPECICAL_POSE_TOKEN

@register_loader("posellava")
class PoseLLaVAModelLoader(BaseModelLoader):
    def load(self, load_lora_model: bool = False, 
             ) -> Tuple[LlavaForConditionalGeneration, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
        
        tokenizer = AutoTokenizer.from_pretrained(
                self.model_local_path,
                model_max_length = self.model_max_length,
                padding_side="right",
                use_fast=False,
        )
        if not load_lora_model: 
            model = PoseLlavaForCausalLM.from_pretrained(
                self.model_local_path, 
                **self.loading_kwargs,
            )
        else:
            # loading the lora model
            # lora_cfg_pretrained = AutoConfig.from_pretrained(self.model_local_path)
            # print(lora_cfg_pretrained)
            adapter_config_path = os.path.join(self.model_local_path, "adapter_config.json")
            if not os.path.exists(adapter_config_path):
                raise ValueError(f"adapter_config.json not found in {self.model_local_path}")
            with open(adapter_config_path, "r") as f:
                base_model_path = json.load(f)['base_model_name_or_path']
                print(f"model_base: {base_model_path}")
            base_model = PoseLlavaForCausalLM.from_pretrained(base_model_path,
                                                             low_cpu_mem_usage=True, 
                                                            #  config=lora_cfg_pretrained,
                                                             **self.loading_kwargs)
            # print(base_model)
            cur_vocab_size = len(tokenizer)
            # base_token_num = base_model.lm_head.out_features
            base_token_num = base_model.lm_head.weight.shape[0]
            if cur_vocab_size > base_token_num:
                print(f"cur_vocab_size: {cur_vocab_size}, base_token_num: {base_token_num}")
                num_new_tokens = cur_vocab_size - base_token_num
                assert num_new_tokens >0, "new tokens should be >0"
                # resize input embeddings
                base_model.resize_token_embeddings(len(tokenizer)) 

                # resize output embeddings
                # output_embeddings = base_model.get_output_embeddings().weight.data
                # output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                # output_embeddings[-num_new_tokens:] = output_embeddings_avg
            
            non_lora_state_path = os.path.join(self.model_local_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_state_path):
                non_lora_trainables = torch.load(non_lora_state_path, map_location='cpu')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                print('Loading additional PoseChat weights...')
                base_model.load_state_dict(non_lora_trainables, strict=False)

            model = PeftModel.from_pretrained(base_model,
                                            self.model_local_path,
                                            torch_dtype=self.torch_dtype,
                                            device_map = self.device_map)
            print('PeftModel is loaded...')

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=model.device, dtype=self.torch_dtype)
        
        pose_tower = model.get_pose_tower()
        # pose_tower.load_model()
        pose_tower.to(device=model.device, dtype=self.torch_dtype)

        add_token_list = list(tokenizer.get_added_vocab().keys())
        if SPECICAL_POSE_TOKEN in add_token_list:
            model.set_pose_token_id(tokenizer.convert_tokens_to_ids(SPECICAL_POSE_TOKEN))   
        if DEFAULT_IMAGE_TOKEN in add_token_list:
            model.set_image_token_id(tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN))

        image_processor = vision_tower.image_processor
        tokenizer.pad_token = "[PAD]"
       
        return model, tokenizer, image_processor
    
    def get_lora_skip_keywords(self, use_llm_lora=False, use_backbone_lora = False):
        all_module_keywords = ['vision_tower', 
                               'mm_projector', 
                               'pose_projector', 
                               'embed_tokens',
                               'model.layers',
                               'lm_head',
                               'pose_decoder' 
                              ]
        lora_target_keywords= []
        if use_llm_lora:
            lora_target_keywords.append('model.layers')
        if use_backbone_lora:
            lora_target_keywords.remove("visual.blocks") 
        return lora_target_keywords