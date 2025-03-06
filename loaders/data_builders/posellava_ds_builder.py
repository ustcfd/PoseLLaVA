import json
import math
from .base import BaseDatasetBuilder, WeightedConcatDataset
from torch.utils.data import ConcatDataset, Dataset
from typing import Dict, Sequence, Optional
from transformers import PreTrainedTokenizer, AutoProcessor
from ..custom_dataset import CustomPoseLlavaSupervisedDataset
from . import register_data_builder


@register_data_builder('posechat')
@register_data_builder('posellava')
class SMPLPOSE_ds_builder(BaseDatasetBuilder):
    def __call__(self, data_args: Dict, 
                 tokenizer: PreTrainedTokenizer, 
                 processor: AutoProcessor,
                 **kwargs) -> Dataset:
        datasets = []
        lengths = []
        
        assert data_args.model_family_id in ['posechat', 'posellava'], "The model_id and the data builder is not compatible"
        if data_args.model_family_id == 'posechat':
            is_smpl_multimodal = False
        elif data_args.model_family_id == 'posellava':
            is_smpl_multimodal = True

        ds_collections = json.loads(open(data_args.meta_path).read())
        for ds_idx, ds_name in enumerate(ds_collections.keys()):
            repeat_time = ds_collections[ds_name].get('repeat_time', 1)
            dataset = CustomPoseLlavaSupervisedDataset(
                ds_name= ds_name,
                template_name= data_args.conv_style,
                meta=ds_collections[ds_name],
                tokenizer = tokenizer,
                image_processor= processor,
                pad2square= data_args.pad2square,
                group_by_length= data_args.group_by_length,
                repeat_time= repeat_time,
                random_seed= ds_idx if self.random_seed is None else self.random_seed,
                is_smpl_multimodal = is_smpl_multimodal
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
