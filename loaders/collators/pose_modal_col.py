from typing import Dict, List, Sequence, Union

import torch
from .base import BaseDataCollator
from . import register_collator

@register_collator("posechat")
@register_collator("posellava")
class SMPLPOSEForDataCollator(BaseDataCollator):
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, 
                                                    batch_first=batch_first, 
                                                    padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.

        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=self.IGNORE_TOKEN_ID)
 
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels.long() if labels.dtype == torch.int32 else labels
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        if 'pixel_values' in instances[0]:
            all_pixel_values = [instance["pixel_values"] for instance in instances]
            batch["pixel_values"] = torch.concat(all_pixel_values, dim=0)
        
        if 'gt_smpl_values' in instances[0]:
            all_gt_smpl_values = [instance["gt_smpl_values"] for instance in instances]
            batch["gt_smpl_values"] = torch.concat(all_gt_smpl_values, dim=0)
        
        if 'input_smpl_values' in instances[0]:
            all_input_smpl_values = [instance["input_smpl_values"] for instance in instances]
            batch["input_smpl_values"] = torch.concat(all_input_smpl_values, dim=0)
        
         
        return batch