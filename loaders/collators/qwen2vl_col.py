
from typing import Dict, List, Sequence, Union

import torch
from .base import BaseDataCollator
from . import register_collator


@register_collator("qwen2vl")
class CustomQwen2VLDataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_image_grid_thw = []
        batch_pixel_values_videos = []
        batch_video_grid_thw = []
        # model_inputs.keys =  ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]
        for instance in instances:
            batch_input_ids.append(instance["input_ids"])
            batch_label_ids.append(instance["labels"])
            if 'pixel_values' in instance:
                batch_pixel_values.append(instance["pixel_values"])
                batch_image_grid_thw.append(instance["image_grid_thw"])    
            if 'pixel_values_videos' in instance:
                batch_pixel_values_videos.append(instance["pixel_values_videos"])
                batch_video_grid_thw.append(instance["video_grid_thw"])
                
        input_ids = self.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(batch_label_ids, batch_first=True, padding_value=self.IGNORE_TOKEN_ID)
 
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels.long() if labels.dtype == torch.int32 else labels
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if len(batch_pixel_values)>0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_grid_thw = torch.cat(batch_image_grid_thw, dim=0)
            batch['pixel_values'] = pixel_values
            batch['image_grid_thw'] = image_grid_thw

        if len(batch_pixel_values_videos)>0:
            pixel_values_videos = torch.cat(batch_pixel_values_videos, dim=0)
            video_grid_thw = torch.cat(batch_video_grid_thw, dim=0)
            batch['pixel_values_videos'] = pixel_values_videos
            batch['video_grid_thw'] = video_grid_thw
        
        return batch
        
        
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, 
                                                    batch_first=batch_first, 
                                                    padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids
    
