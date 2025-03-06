import torch
import torch.nn as nn
from torch.nn import MSELoss
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import  AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from mllm.utils.smpl_utils import rot6d_to_rotmat, batch_rodrigues, rotation_matrix_to_angle_axis

from mllm.models.constants.llava_constants import IMAGE_TOKEN_INDEX,IGNORE_INDEX
from mllm.utils.mm_utils import get_anyres_image_grid_shape,unpad_image

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector


class PoseChatConfig(LlamaConfig):
    model_type = "posechat"

class PoseChatLlamaModel(LlamaModel):
    config_class = PoseChatConfig
    def __init__(self, config: LlamaConfig):
        super(PoseChatLlamaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", True)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower
    


class PoseChatForCausalLM(LlamaForCausalLM):
    config_class = PoseChatConfig
    supports_report_metrics: bool = True # IMPORTANT
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        
        self.model = PoseChatLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        
        self.specical_pose_token_idx = 32000
        self.img_token_idx = IMAGE_TOKEN_INDEX
    
        hidden_fc = [ 
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 144)        
        ]
        self.alpha = 10
        self.pose_decoder = nn.Sequential(*hidden_fc) 
        # self.pose_decoder.train()
        # for param in self.pose_decoder.parameters():
        #     param.requires_grad = True
        # self.pose_decoder = build_pose_decoder_tower(delay_load=False)
        self.post_init()
    
    def get_model(self):
        return self.model
    
    def set_img_token_idx(self, new_token_id):
        print("Setting the image token id:", new_token_id)
        self.img_token_idx = new_token_id

    def set_pose_token_id(self, new_token_id):
        print("Setting the pose token id:", new_token_id)
        self.specical_pose_token_idx = new_token_id

    def get_img_token_id(self):
        return self.img_token_idx
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def get_vision_tower(self):
        return self.model.get_vision_tower()

    def encode_images(self, images):
        image_features = self.model.get_vision_tower()(images)
        image_features = self.model.mm_projector(image_features)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,  # Add this line
        # image_sizes: Optional[List[List[int]]] = None,
        gt_smpl_values: Optional[torch.FloatTensor] = None, # Add this line
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:     
            (
                input_ids, 
                position_ids, 
                attention_mask, 
                past_key_values, 
                inputs_embeds,
                labels 
            )= self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    pixel_values,
                    image_sizes=image_sizes
            )
        
        
       
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states= True, # 此处output_hidden_states需设置为True
            return_dict=return_dict,
            # cache_position=cache_position
            )
        
        llm_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        logits = outputs.logits if isinstance(outputs, dict) else outputs[1]
        loss = llm_loss
        if labels is not None:
            last_hidden_state = outputs.hidden_states[-1] if isinstance(outputs, dict) else outputs[-2][-1] # last_hidden_state
            # Compute the pose loss
            batch_size = labels.shape[0]
            pose_token_mask = (labels[:,1:]==self.specical_pose_token_idx)
            pose_token_mask = torch.cat([pose_token_mask,
                torch.zeros((batch_size, 1)).bool().to(labels.device),
                ],dim=1)
            pose_token_counts = pose_token_mask.int().sum() # [n pose] 
            if pose_token_counts > 0:         
                pose_hidden_states = last_hidden_state[pose_token_mask]  # [bs * num_sentence, out_dim]
                pose_pred = self.pose_decoder(pose_hidden_states) # [bs * num_sentence, 72]
                batch_pose_token_counts = pose_token_mask.int().sum(-1) #[batch_size]
                pose_token_offset = batch_pose_token_counts.cumsum(-1)  # [bs] e.g., [3, 6, 9, 12, 15, 18, 21]
                pose_token_offset = torch.cat([torch.zeros(1).long().to(pose_token_offset.device),
                                            pose_token_offset], dim=0)
                pose_pred_list = []
                pose_target_list = []
                for i in range(batch_size):
                    start_i, end_i = pose_token_offset[i], pose_token_offset[i + 1]
                    pose_target_list.append(gt_smpl_values[start_i:end_i])
                    pose_pred_list.append(pose_pred[start_i:end_i])
                
                pose_target = torch.cat(pose_target_list,dim=0).to(pose_token_mask.device) #[B,144]
                pose_preds = torch.cat(pose_pred_list,dim=0).to(pose_token_mask.device) #[B,144]
                assert pose_preds.size(0) == pose_target.size(0) == pose_token_counts, "The pose_targets does not equal to pose_token_counts, please check the training data"             
                
                pose_batch_size = pose_preds.shape[0]

                # pose_pred_rotmat = rotation_6d_to_matrix(pose_preds.float()).reshape(pose_batch_size, 24, 3, 3).type_as(pose_preds)
                # gt_rotmat = axis_angle_to_matrix(pose_target.float().reshape(-1,24,3)).type_as(pose_preds)
                loss_smpl = MSELoss(reduction='mean')
                pose_pred_rotmat = rot6d_to_rotmat(pose_preds.reshape(-1,24,6).float()).reshape(pose_batch_size, 24, 3, 3).type_as(pose_preds)
                gt_rotmat = batch_rodrigues(pose_target.float().view(-1,3)).reshape(-1, 24, 3, 3).type_as(pose_preds)
                
                pose_loss = loss_smpl(pose_pred_rotmat, gt_rotmat)               
                # pose_pred_rotmat = rot6d_to_rotmat(pose_preds.float()).reshape(pose_batch_size*24,3,3)
                # pose_pred_aa = rotation_matrix_to_angle_axis(pose_pred_rotmat).contiguous().reshape(pose_batch_size,72).type_as(pose_target)
                # pose_loss = self.pose_loss_fn(pose_pred_aa, pose_target)
                # pose_loss = pose_loss
                loss = llm_loss + self.alpha * pose_loss 
                if hasattr(self, 'report_metrics') and callable(self.report_metrics): 
                    self.report_metrics(poseloss = self.alpha * pose_loss)
            else:
                loss = llm_loss
 
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss) + output if loss is not None else output
        

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states= outputs.hidden_states[-1] if output_hidden_states else None,
            attentions= outputs.attentions,
        )
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        # IMAGE_TOKEN_INDEX = self.get_img_token_id()
        img_token_idx = self.get_img_token_id()
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == img_token_idx).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == img_token_idx)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def prepare_inputs_for_generation(self, 
                                      input_ids, 
                                      past_key_values=None,
                                      inputs_embeds=None,
                                      pixel_values=None,
                                      attention_mask=None,
                                      cache_position=None, 
                                      **kwargs):
        image_sizes = kwargs.pop("image_sizes", None)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs
        )
        if pixel_values is not None:
            model_inputs['pixel_values'] = pixel_values
        # elif cache_position[0] == 0:
        #     model_inputs["pixel_values"] = pixel_values
            
        if image_sizes is not None:
            model_inputs['image_sizes'] = image_sizes
        return model_inputs
    

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if pixel_values is not None: 
            (
                _,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                pixel_values,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
        
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    @torch.no_grad()
    def chat(self, 
            input_ids: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            image_sizes: Optional[torch.Tensor] = None,
            **kwargs,):
        
        assert input_ids.size(0)==1, "batch size >1 is not supported"
        kwargs.update(return_dict_in_generate=True,
                    output_hidden_states=True)  
        
        outputs = self.generate(input_ids=input_ids,
                                pixel_values= pixel_values,
                                image_sizes = image_sizes,
                                **kwargs)
        output_hidden_states = outputs.hidden_states # 这里抽取的是last_hidden_state
        output_ids = outputs.sequences

        pose_token_searched = (output_ids[:, 1:]==self.specical_pose_token_idx).nonzero() 
        # pose_token_searched = (output_ids==self.pose_token_idx).nonzero()
        beam_search = False
        if hasattr(outputs, "beam_indices"): # beam size > 1
            beam_indices = outputs.beam_indices
            beam_search = True
        if len(pose_token_searched) > 0 :       
            pose_token_num_batch = [0] * input_ids.size(0)
            pose_embedding = []
            for batch_id, pose_token_id in pose_token_searched: 
                pose_token_num_batch[batch_id] +=1
                if beam_search:
                    beam_idx = beam_indices[batch_id, pose_token_id]
                    pose_loc_feat = output_hidden_states[pose_token_id][beam_idx]
                else:
                    pose_loc_feat = output_hidden_states[pose_token_id]
                pose_embedding.append(pose_loc_feat.squeeze(1)) # [(1,5120)]
            pose_embedding = torch.cat(pose_embedding, dim=0)
            pose_pred = self.pose_decoder(pose_embedding) # [batch_size,144]
            
            batch_size = pose_pred.shape[0]
            # pose_pred = pose_pred + self.init_pose.expand(batch_size,-1).to(pose_pred.device)
            pose_pred_rotmat = rot6d_to_rotmat(pose_pred.float()).reshape(batch_size*24,3,3)
            pose_pred_aa = rotation_matrix_to_angle_axis(pose_pred_rotmat).contiguous().reshape(batch_size,72)
            pose_pred_result = torch.split_with_sizes(pose_pred_aa, pose_token_num_batch)
            return output_ids, list(pose_pred_result)
        else:
            return output_ids, [None]*input_ids.size(0)
        
        
AutoConfig.register("posechat", PoseChatConfig)   
AutoModelForCausalLM.register(PoseChatConfig, PoseChatForCausalLM)