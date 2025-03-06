import argparse
import torch
import os
import json
from loaders.model_loaders import LOADERS


argparse = argparse.ArgumentParser()
argparse.add_argument("--model_id", type=str, help='Model name')
argparse.add_argument("--lora_path", type=str, help='Path to the input model')
argparse.add_argument("--save_path", type=str, help='Path to the output model')
argparse.add_argument("--device_map", type=str, default='cuda')
argparse.add_argument("--fp16", type=bool, default=True)
argparse.add_argument("--save_processor", type=bool, default=True)

args = argparse.parse_args()
model_family_id = args.model_id
lora_path = args.lora_path
save_path = args.save_path
device_map = args.device_map


torch_dtype=torch.float16 if args.fp16 else torch.bfloat16
if model_family_id not in LOADERS:
    raise ValueError(f"Model {model_family_id} not found")

model_loader = LOADERS[model_family_id](
    model_local_path = lora_path, 
    compute_dtype = torch_dtype, 
    device_map=device_mapc
)
   
lora_model, tokenizer, processor = model_loader.load(
    load_lora_model=True
)

model=lora_model.merge_and_unload() # 注意一定要加 model=
print('Saving model...')
model.save_pretrained(save_path)
print('Saving tokenizer...')
tokenizer.save_pretrained(args.output_path)



save_processor = (bool(args.save_processor) or processor.chat_template is not None)
if save_processor:
    try:
        print('Saving processor...')
        processor.save_pretrained(args.output_path)
        if processor.chat_template is not None:
            output_chat_template_file = os.path.join(save_path, "chat_template.json")
            chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
            with open(output_chat_template_file, "w", encoding="utf-8") as writer:
                writer.write(chat_template_json_string)
        # 如果不能一起保存chat_template.json, 则需要手动保存
    except:
        raise ValueError('No processer')

print('All Done!')