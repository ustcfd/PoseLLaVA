from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import transformers

@dataclass
class LoraArguments:
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={'help': 'The maximum number of dynamic patches. Default is 6.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the backbone model. Default is 0.'}
    )
    q_lora: bool = field(default=False)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_custom_trainer: bool = field(
        default=False,
        metadata={'help': 'Set to True to enable the use of a custom trainer.'},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use gradient checkpointing.'},
    )
    mm_projector_lr: Optional[float] = field(
        default=0,
        metadata={'help': 'Set the learning rate for the vision mm_projector. Default is 0.'}
    )
    vision_tower_lr: Optional[float] = field(
        default=0,
        metadata={'help': 'Set the learning rate for the pose mm_projector. Default is 0.'}
    )
    
    # pose_projector_lr: Optional[float] = field(
    #     default=0,
    #     metadata={'help': 'Set the learning rate for the pose mm_projector. Default is 0.'}
    # )


  
   



