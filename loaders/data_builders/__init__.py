DATA_BUILDERS = {}

def register_data_builder(name):
    def register_data_builder_cls(cls):
        if name in DATA_BUILDERS:
            return DATA_BUILDERS[name]
        DATA_BUILDERS[name] = cls
        return cls
    return register_data_builder_cls


from .llava_ds_builder import LLava_ds_builder
from .posellava_ds_builder import SMPLPOSE_ds_builder
