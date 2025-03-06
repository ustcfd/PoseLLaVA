import sys
import os
import transformers
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from peft import LoraConfig, get_peft_model
from mllm.train.arguments import TrainingArguments, LoraArguments
from loaders.model_loaders import LOADERS
from loaders.collators import COLLATORS
import torch.distributed as dist
from mllm.models.constants.llava_constants import DEFAULT_IMAGE_TOKEN, DEFAULT_POSE_TOKEN, SPECICAL_POSE_TOKEN
from loaders.data_builders import DATA_BUILDERS
from transformers import set_seed, Trainer
from mllm.patch import replace_train_sampler, replace_create_optimizer
import logging
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler, enable_explicit_format, set_verbosity)
from mllm.utils.train_utils import (rank0_print, safe_save_model_for_hf_trainer, rank0_print_non_lora_params,
                                    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3,
                                    set_requires_grad, find_all_linear_names, vlog_loss_curve)
from hf_mtask_trainer import HfMultiTaskTrainer


replace_train_sampler()
replace_create_optimizer()
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_family_id : Optional[str] = field(
        default='internvl',
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )

    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )

    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the vision backbone of the model.'},
    )
    freeze_contector: bool = field(
        default= False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )
    
    tune_head: bool = field(
        default= False,
        metadata={'help': 'Set to True to freeze the MLP layers of the model.'},
    )

    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is last layer.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )


@dataclass
class DataArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    pad2square: Optional[bool] = field(
        default=True,
        metadata={'help': 'Pad the image to a square shape if set to True.'},
    )
    conv_style: Optional[str] = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling.'},
    )

def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
   
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # if training_args.should_log:
    #     # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    #     transformers.utils.logging.set_verbosity_info()
    
    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # set_verbosity(log_level)
    # enable_default_handler()
    # enable_explicit_format()

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    device_map = None
    # Load model and tokenizer by model_args.model_family_id
    loader = LOADERS[model_args.model_family_id](
        model_local_path= model_args.model_name_or_path,
        model_max_length = data_args.max_seq_length,
        compute_dtype=compute_dtype,
        # bnb_config=bnb_config,
        use_flash_attn = model_args.use_flash_attn,
        device_map=device_map,
    )
    # Load pretrained model, tokenizer, and image processor
    model, tokenizer, processor = loader.load()
    data_args.model_family_id = model_args.model_family_id
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad) 
        training_args.gradient_checkpointing_kwargs={"use_reentrant":True}
    
    tokenizer.pad_token = tokenizer.unk_token
    # 增加特殊token
    # token_list = [DEFAULT_POSE_TOKEN, DEFAULT_IMAGE_TOKEN, SPECICAL_POSE_TOKEN] + ["SMPL"]
    token_list = [SPECICAL_POSE_TOKEN] + ["SMPL"]
    num_new_tokens = tokenizer.add_tokens(token_list)
    resize_embedding_flag = False
    if num_new_tokens > 0:
        org_token_num = model.get_input_embeddings().weight.data.shape[0]
        cur_token_num = len(tokenizer)
        if cur_token_num > org_token_num:
            # only when cur_token_num > org_token_num ,resize input embeddings
            model.resize_token_embeddings(len(tokenizer)) 
            resize_embedding_flag = True
            # 非必须，resize output embeddings ,
            # output_embeddings = model.get_output_embeddings().weight.data
            # output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            # output_embeddings[-num_new_tokens:] = output_embeddings_avg
            model.config.vocab_size = len(tokenizer)
   
    add_token_list = list(tokenizer.get_added_vocab().keys())
    if SPECICAL_POSE_TOKEN in add_token_list:
        # 防止<POSE> token id 不一致
        model.set_pose_token_id(tokenizer.convert_tokens_to_ids(SPECICAL_POSE_TOKEN)) 
    if DEFAULT_IMAGE_TOKEN in add_token_list:
        model.set_image_token_id(tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN))
    
    
    # Finally, add Lora peft module by config
    if lora_args.use_llm_lora or lora_args.use_backbone_lora:
        # rank0_print("Adding LoRA adapters...")
        # model = lora_setting(model, training_args)
        lora_target_keywords = loader.get_lora_skip_keywords(lora_args.use_llm_lora, lora_args.use_backbone_lora)
        rank0_print("The lora target keywords are:",lora_target_keywords)
        lora_config = LoraConfig(
            r= lora_args.use_llm_lora,
            lora_alpha= 2*lora_args.use_llm_lora,
            target_modules= find_all_linear_names(model, target_keywords=lora_target_keywords),
            # target_modules= "model\..*layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|down_proj|gate_proj|up_proj)",
            lora_dropout = lora_args.lora_dropout,
            # use_dora=True
            # modules_to_save=['visual']  # 请确保与模型中所使用的参数名一致
            # modules_to_save = module_save_keywords
        )
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
       
    
    if not lora_args.use_backbone_lora:   
        set_requires_grad(model.get_vision_tower(), not model_args.freeze_backbone)
    
    # 请确保与模型中所使用的参数名一致
    set_requires_grad(model.get_model().mm_projector, not model_args.freeze_contector)
    set_requires_grad(model.pose_decoder, not model_args.freeze_contector)
    if hasattr(model.get_model(), 'pose_projector'):
        set_requires_grad(model.get_model().pose_projector, not model_args.freeze_contector)

    # 设置embed_tokens和lm_head是否要参与训练
    if model_args.tune_head or resize_embedding_flag:
        set_requires_grad(model.get_input_embeddings(), True)
        set_requires_grad(model.get_output_embeddings(), True)


    # print trainable parameters
    rank0_print_non_lora_params(model)
  
    data_args.group_by_length = training_args.group_by_length
    data_builder = DATA_BUILDERS[model_args.model_family_id]()
    train_dataset = data_builder(
        data_args, tokenizer=tokenizer, processor = processor
    )

    data_collator = COLLATORS[model_args.model_family_id](tokenizer=tokenizer)

    
    # Initialize our Trainer
    if training_args.use_custom_trainer:
        trainer = HfMultiTaskTrainer(
                model = model,
                args=training_args,
                train_dataset= train_dataset,
                eval_dataset=None,
                tokenizer=tokenizer,
                data_collator= data_collator,)
    else:
        trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=None,
                tokenizer=tokenizer,
                data_collator= data_collator,
        )

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    try:
        metrics['train_samples'] = len(train_dataset)
    except:
        metrics['train_samples'] = -1

    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state() # 保存训练状态,便于后续继续训练或评估
    
    if lora_args.use_llm_lora or lora_args.use_backbone_lora: 

        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters()
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )
        rank0_print("non_lora_state_dict:") 
        rank0_print(non_lora_state_dict.keys())
        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
            tokenizer.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)
  
    vlog_loss_curve(trainer.state.log_history, 
                    training_args.output_dir)  
        
 


if __name__ == "__main__":
    train()