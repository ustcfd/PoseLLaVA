import torch
import transformers
import logging
import torch.distributed as dist
import pandas as pd
import os
import matplotlib.pyplot as plt

def set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def rank0_print_non_lora_params(module):
     if dist.get_rank() == 0:
        print("The non-lora trainable parameters are:")
        for name, param in module.named_parameters():
            if "lora_" not in name and param.requires_grad:
                print(name)

def find_all_linear_names(model, target_keywords=[]):
    linear_cls = torch.nn.Linear
    # embedding_cls = torch.nn.modules.Embedding
    lora_module_names = set()
    for name, module in model.named_modules():
        # 不属于target_keywords，跳过
        if not any(module_name in name for module_name in target_keywords):
            continue
        if isinstance(module, linear_cls):
            lora_module_names.add(name)
    for name in list(lora_module_names):
        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir) # 保存训练后的模型
        return
    
   
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        trainer.model.config.save_pretrained(output_dir)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias='none'):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def vlog_loss_curve(log_history, out_dir, plt_pose_loss=True):
    train_history = [x for x in log_history if 'loss' in x.keys()]
    dfhistory_train = pd.DataFrame(train_history)
    if 'step' in dfhistory_train.columns:
        dfhistory_train['step'] = dfhistory_train['step'].astype(int)
    plt.figure()
    # 绘制曲线图
    plt.plot(dfhistory_train['step'], dfhistory_train['loss'], color="red", linestyle="-", label="loss")
    # plt.plot(dfhistory_train['step'], dfhistory_train['pose_loss'], color="green", linestyle="--", label="pose_loss")
    # plt.plot(dfhistory_train['step'], dfhistory_train['llm_loss'], color="blue", linestyle="-.", label="llm_loss")
    # 设置x轴和y轴标签
    plt.legend(loc="upper right")
    plt.xlabel('step')
    plt.ylabel('train_loss')
    # 添加标题，为"Loss Curve"
    plt.title("LLM Loss Curve")
    # 保存图片
    out_path = os.path.join(out_dir, 'loss_curve.png')
    plt.savefig(out_path)
    if plt_pose_loss:
        plt.figure()
        plt.plot(dfhistory_train['step'][5:], dfhistory_train['poseloss'][5:], color="green", linestyle="--", label="pose_loss")
        # 设置x轴和y轴标签
        plt.legend(loc="upper right")
        plt.xlabel('step')
        plt.ylabel('train_loss')
        # 添加标题，为"Loss Curve"
        plt.title("Pose Loss Curve")
        # 保存图片
        out_path = os.path.join(out_dir, 'pose_loss_curve.png')
        plt.savefig(out_path)