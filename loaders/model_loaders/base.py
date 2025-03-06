from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoProcessor, BitsAndBytesConfig


class BaseModelLoader(ABC):
    def __init__(
        self, 
        # model_hf_path: str,
        model_local_path: str, 
        compute_dtype: torch.dtype = torch.float16,
        model_max_length: int = 4096,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        use_flash_attn: bool = False,
        device_map: Optional[Union[Dict, str]] = 'auto',
    ) -> None:
        # self.model_hf_path = model_hf_path
        self.model_local_path = model_local_path
        self.torch_dtype = compute_dtype
        self.device_map = device_map
        self.model_max_length = model_max_length

        self.loading_kwargs = dict(
            torch_dtype=compute_dtype,
            quantization_config=bnb_config,
            # low_cpu_mem_usage=False,
            device_map=device_map,
        )
     
        if use_flash_attn:
            self.loading_kwargs["attn_implementation"] = "flash_attention_2"

    @abstractmethod
    def load(self, load_model: bool = True) -> Tuple[
        PreTrainedModel, Union[None, PreTrainedTokenizer], Union[None, AutoProcessor]
    ]: ...

    def get_lora_skip_keywords(self, **kwargs) -> List:
        ...
       