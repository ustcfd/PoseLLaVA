COLLATORS = {}

def register_collator(name):
    def register_collator_cls(cls):
        if name in COLLATORS:
            return COLLATORS[name]
        COLLATORS[name] = cls
        return cls
    return register_collator_cls

from .img_modal_col import IMGForDataCollator
from .qwen2vl_col import CustomQwen2VLDataCollator
from .pose_modal_col import SMPLPOSEForDataCollator
