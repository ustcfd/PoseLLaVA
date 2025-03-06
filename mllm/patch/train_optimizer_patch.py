
import transformers
from transformers import Trainer, logging
from transformers.trainer import is_sagemaker_mp_enabled
from transformers.trainer import logger
import torch.nn as nn

from packaging import version
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

# deepspeed.config 不能配置"optimizer"，才能起作用
def create_optimizer(self):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    """
    opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
    if self.optimizer is None:
        
        decay_parameters = self.get_decay_parameter_names(opt_model)
        lr_mapper = {}
        if hasattr(self.args, 'mm_projector_lr') and float(self.args.mm_projector_lr)>0:
            lr_mapper["mm_projector"] = self.args.mm_projector_lr

        if hasattr(self.args, 'pose_projector_lr') and float(self.args.pose_projector_lr)>0:
            lr_mapper["pose_projector"] = self.args.pose_projector_lr
        
        if hasattr(self.args, 'vision_tower_lr') and float(self.args.vision_tower_lr)>0:
            lr_mapper["vision_tower"] = self.args.vision_tower_lr
  
        if len(lr_mapper) > 0:
            special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                    "weight_decay": 0.0,
                },
            ]
            for module_keyword, lr in lr_mapper.items():
                module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                            "weight_decay": self.args.weight_decay,
                            "lr": lr,
                        },
                        {
                            "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                            "weight_decay": 0.0,
                            "lr": lr,
                        },
                    ]
                )
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
        # e.g. for GaLore optimizer.
        if "params" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("params")

        # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
        # e.g. for LOMO optimizer.
        if "model" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("model")

        # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
        # to avoid arguments conflicts.
        if "optimizer_dict" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if optimizer_cls.__name__ == "Adam8bit":
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    logger.info(f"skipped {module}: {skipped/2**20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            logger.info(f"skipped: {skipped/2**20}M params")
            
    if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)
    return self.optimizer



def replace_create_optimizer():
    
    transformers.Trainer.create_optimizer = create_optimizer
    print('Replace original create_optimizer with custom create_optimizer')



