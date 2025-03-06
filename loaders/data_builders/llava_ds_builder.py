import json
import math
from .base import BaseDatasetBuilder, WeightedConcatDataset
from torch.utils.data import ConcatDataset, Dataset
from typing import Dict, Sequence, Optional
from transformers import PreTrainedTokenizer, AutoProcessor
from ..custom_dataset import CustomLlavaSupervisedDataset
from . import register_data_builder


@register_data_builder('intern2vl')
class LLava_ds_builder(BaseDatasetBuilder):
    def __call__(self, data_args: Dict, 
                 tokenizer: PreTrainedTokenizer, 
                 processor: AutoProcessor,
                 **kwargs) -> Dataset:
        datasets = []
        lengths = []
        ds_collections = json.loads(open(data_args.meta_path).read())
        for ds_idx, ds_name in enumerate(ds_collections.keys()):
            repeat_time = ds_collections[ds_name].get('repeat_time', 1)
            if 'max_dynamic_patch' in ds_collections[ds_name]:
                max_num = ds_collections[ds_name]['max_dynamic_patch']
            else:
                max_num = 12

            dataset = CustomLlavaSupervisedDataset(
                ds_name= ds_name,
                template_name= data_args.conv_style,
                meta=ds_collections[ds_name],
                tokenizer = tokenizer,
                image_processor= processor,
                pad2square= data_args.pad2square,
                group_by_length= data_args.group_by_length,
                repeat_time= repeat_time,
                random_seed= ds_idx if self.random_seed is None else self.random_seed,
                dynamic_image_size=data_args.dynamic_image_size,
                use_thumbnail = data_args.use_thumbnail,
                min_dynamic_patch=data_args.min_dynamic_patch,
                max_dynamic_patch= max_num,
            )
            datasets.append(dataset)
            if data_args.use_data_resampling:
                lengths.append(math.sqrt(len(dataset)))
            else:
                lengths.append(len(dataset))


        if data_args.use_data_resampling:
            total_length = sum(lengths)
            weights = [l / total_length for l in lengths]
            train_dataset = WeightedConcatDataset(datasets, weights)
        else:
            train_dataset = ConcatDataset(datasets)
        
       
        return train_dataset
