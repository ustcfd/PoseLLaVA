from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional

import torch
from transformers import PreTrainedTokenizer, AutoProcessor, AutoConfig


class BaseDataCollator(ABC, object):
    """Collate examples for supervised fine-tuning."""
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[AutoConfig] = None,
        processor: Optional[AutoProcessor] = None,
        mask_question_tokens: bool = True
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.mask_question_tokens = mask_question_tokens
    
    @property
    def IGNORE_TOKEN_ID(self) -> int:
        return -100

    @property
    def PAD_TOKEN_ID(self) -> int:
        return self.tokenizer.pad_token_id
    
    @abstractmethod
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]: ...

    def pad_sequence(self, input_ids, batch_first, padding_value): 
        pass