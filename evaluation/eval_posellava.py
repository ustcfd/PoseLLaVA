import sys
sys.path.append('./')
import torch
import json
import os
import funcy
import time
import argparse
from PIL import Image
from typing import Dict,  Sequence
from mllm.utils.mm_utils import expand2square
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import numpy as np
from mllm.utils.smpl_utils import matrix_to_rotation_6d, axis_angle_to_matrix
from mllm.conversation import get_conv_template
from mllm.models.constants.llava_constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from mllm.utils.mm_utils import tokenizer_image_and_pose_token
import transformers
from transformers.image_processing_utils import BaseImageProcessor
from loaders.collators.base import BaseDataCollator
from loaders.model_loaders import LOADERS
from evaluation.pose_metric import compute_mpjpe, compute_pa_mpjpe


class EvalDataset(Dataset):
    def __init__(self, template_name, 
                ds_name,
                meta_data, 
                tokenizer: transformers.PreTrainedTokenizer,
                image_processor: BaseImageProcessor,
                pad2square = True,
                test_samples:int = 0,
                is_smpl_multimodal = False):
        super(EvalDataset, self).__init__()
        
        self.ds_name = ds_name
        self.template_name = template_name
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.pad2square = pad2square
        self.is_smpl_multimodal = is_smpl_multimodal

        with open(meta_data['annotation'], 'r') as f:
            if meta_data['annotation'].endswith('.json'):
                self.raw_data = json.load(f)
            elif meta_data['annotation'].endswith('.jsonl'):
                self.raw_data = [json.loads(line) for line in f.readlines()]
            else:
                raise ValueError(f"Unsupported annotation file format: {meta_data['annotation']}")
        if isinstance(test_samples,int) and test_samples > 0:
            self.raw_data = self.raw_data[:test_samples]
        self.root = meta_data['root']

    def __len__(self):
        return len(self.raw_data)
    
    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        # if self.tcs_loader is not None and 's3://' in image_path:
        #     return self.tcs_loader(image_path)
        image = Image.open(image_path).convert('RGB')
        # max_edge = max(image.size)
        # image = image.resize((max_edge, max_edge))
        return image
    
    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        data_item = self.raw_data[i]
        return self.multi_modal_get_item(data_item)
    
    def multi_modal_get_item(self, data_item) -> Dict[str, torch.Tensor]:
        question_id =  data_item['id'] 

        if 'image' in data_item:
            if isinstance(data_item["image"], list):
                image_sources = data_item["image"]
            elif isinstance(data_item["image"], str):
                image_sources = [data_item["image"]]
            else:
                raise ValueError(f"Invalid image source type: {type(data_item['image'])}")
            num_image = len(image_sources)
            images = []
            image_paths = []
            for image_path in image_sources:
                image_path = self.get_image_path(image_path)
                image_paths.append(image_path)
                image = self.load_image(image_path)
                if self.pad2square:
                    image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                    images.append(image) 
                else: # Otherwise, use the original image as a single patch 
                    images.append(image)  
            num_image = len(images)   
        else:
            image_paths = None
            crop_size = self.image_processor.crop_size
            # Create a blank white image
            image = Image.new('RGB', (crop_size["height"], crop_size["width"]), (255, 255, 255))
            images = [image]
            num_image = 0
             
        pixel_values = [self.image_processor.preprocess(image,return_tensors='pt')['pixel_values'][0] for image in images] 
        pixel_values = torch.stack(pixel_values)
        
        text_only = False if 'image' in data_item or 'input_pose' in data_item else True
        
        if 'conversations' in data_item:
            question = data_item['conversations'][0]["value"]
            question = question.replace('<image>', '').strip()
            question = (DEFAULT_IMAGE_TOKEN + '\n') * num_image + question
        elif 'question' in data_item:
            question = data_item['question']
        else:
            raise ValueError("The question is not right, please check it")


        conv = get_conv_template(self.template_name)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if not text_only:
            prompt_input_ids = tokenizer_image_and_pose_token(prompt, self.tokenizer, return_tensors="pt")
        else:
            prompt_input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding= False,
                max_length= self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids[0]
        
        if 'target_pose' in data_item:
            target_smpl_list = data_item['target_pose']
            gt_smpl_values = torch.from_numpy(np.array(target_smpl_list, dtype=np.float32)).reshape(-1,72)
        else:
            gt_smpl_values = torch.zeros(1,72)
        
        if not self.is_smpl_multimodal:
            ret = dict(
                    input_ids= prompt_input_ids,
                    pixel_values = pixel_values,
                    gt_smpl_values = gt_smpl_values,
                    question_id= question_id,
                    input_prompt = prompt,
                    image_path = image_paths
                )
        else:
            if 'input_pose' in data_item:
                input_smpl_list = data_item['input_pose'] 
                input_smpl_values = torch.from_numpy(np.array(input_smpl_list, dtype=np.float32)).reshape(-1,24,3)  
                input_smpl_values = matrix_to_rotation_6d(axis_angle_to_matrix(input_smpl_values)) #[B,24,6]
            else:
                input_smpl_values = torch.zeros(1,24,6) 
        # Create the final return dictionary
            ret = dict(
                input_ids= prompt_input_ids,
                pixel_values = pixel_values,
                input_smpl_values = input_smpl_values,
                gt_smpl_values = gt_smpl_values,
                question_id= question_id,
                input_prompt = prompt,
                image_path = image_paths
            )
        return ret


class EvalDataCollator(BaseDataCollator):
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, 
                                                    batch_first=batch_first, 
                                                    padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.

        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
       
        input_prompt = [_['input_prompt'] for _ in instances]
        question_ids = [_['question_id'] for _ in instances]
        image_paths = [_['image_path'] for _ in instances]
        
        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_prompt = input_prompt,
            question_ids = question_ids,
            image_paths = image_paths
        )

        if 'pixel_values' in instances[0]:
            all_pixel_values = [instance["pixel_values"] for instance in instances]
            batch["pixel_values"] = torch.concat(all_pixel_values, dim=0)
        
        if 'gt_smpl_values' in instances[0]:
            all_gt_smpl_values = [instance["gt_smpl_values"] for instance in instances]
            batch["gt_smpl_values"] = all_gt_smpl_values
        
        if 'input_smpl_values' in instances[0]:
            all_input_smpl_values = [instance["input_smpl_values"] for instance in instances]
            batch["input_smpl_values"] = torch.concat(all_input_smpl_values, dim=0)
         
        return batch

def eval_single_dataset(ds_name, dataset_cfg, 
                        model, tokenizer, processor, args):
    
    if args.model_family_id == 'posechat':
        is_smpl_multimodal = False
    elif args.model_family_id == 'posellava':
        is_smpl_multimodal = True 
    dataset = EvalDataset(
        template_name= args.conv_style,
        ds_name =ds_name,
        meta_data= dataset_cfg,
        tokenizer= tokenizer,
        image_processor= processor,
        is_smpl_multimodal= is_smpl_multimodal,
    )
    data_collator = EvalDataCollator(tokenizer=tokenizer)
    eval_dataloader = DataLoader(
        dataset= dataset,
        batch_size= 1, # "batch size >1 is not supported"
        num_workers= 1,
        pin_memory=True,
        drop_last=False,
        collate_fn = data_collator,
    )
    final_results = []
   
    for batch in eval_dataloader:
        model_inputs = {
            "input_ids" : batch['input_ids'].to(args.device),
            "pixel_values" : batch['pixel_values'].to(dtype=model.dtype).to(args.device),
            "attention_mask" : batch['attention_mask'].to(args.device),
        }
        if is_smpl_multimodal:
            model_inputs.update({
                "input_smpl_values" : batch['input_smpl_values'].to(dtype=model.dtype).to(args.device),
            })

        response, pred_smpl_poses = model.chat(
            **model_inputs,
            num_beams=1,
            length_penalty=1,
            max_length=2048,
            repetition_penalty=1.5,
            output_scores=True,
            use_cache=True,
        )
        pred_answers = tokenizer.batch_decode(response, skip_special_tokens=True)
        print(pred_answers)
        for (qid, image_file, prompt, pred_answer, pred_smpl, gt_smpl) in \
            zip(batch['question_ids'], batch['image_paths'], batch['input_prompt'], 
            pred_answers, pred_smpl_poses, batch["gt_smpl_values"]):
            base_info = {
                'question_id': qid,
                'image': image_file,
                'prompt': prompt,
                'pred_answer': pred_answer,  
                
            }
            if pred_smpl is not None and torch.numel(pred_smpl) != 0 and gt_smpl is not None:
                if pred_smpl.shape == gt_smpl.shape:
                    base_info.update({
                        'pred_smpl': pred_smpl.cpu().squeeze().tolist(), 
                        'gt_smpl': gt_smpl.cpu().squeeze().tolist(),
                    })
                    final_results.append(base_info)

    if args.output_json and len(final_results)>0:
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{ds_name}_{time_prefix}.json'
        json.dump(final_results, open(results_file, 'w'), ensure_ascii=False)

    if args.eval_metric and len(final_results)>0:
        target_data = funcy.lmap(lambda x: torch.from_numpy(np.array(x['gt_smpl'],dtype=np.float32)).reshape(-1,72), final_results)
        target_data = torch.concat(target_data, dim=0).cuda()
        pred_data = funcy.lmap(lambda x: torch.from_numpy(np.array(x['pred_smpl'],dtype=np.float32)).reshape(-1,72), final_results)
        pred_data = torch.concat(pred_data, dim=0).cuda()
        from evaluation.smpl.smpl_official import SMPL
        smplModelPath = 'evaluation/smpl/additional/SMPL_NEUTRAL.pkl'
        smpl = SMPL(smplModelPath, batch_size=1, create_transl=False).to(pred_data.device)
        fake_beats_data = torch.zeros([1,10]).to(pred_data.device)
        target_smpl_output = smpl(body_pose=target_data[:, 3:],
                        global_orient=target_data[:,:3],
                        betas=fake_beats_data,
                        pose2rot=True)
        
        pred_smpl_output = smpl(body_pose=pred_data[:, 3:],
                        global_orient=pred_data[:,:3],
                        betas=fake_beats_data,
                        pose2rot=True)
        pred_smpl_output = pred_smpl_output - pred_smpl_output[:, [0],:].clone()
        target_smpl_output = target_smpl_output- target_smpl_output[:, [0],:].clone()
        print(f"Evaluating {ds_name} Dataset, The len of evaluation data: {len(final_results)}")
        mpjpe_result = 1000 * compute_mpjpe(pred_smpl_output, target_smpl_output).cpu().item()
        pa_mpjpe_result = 1000 * compute_pa_mpjpe(pred_smpl_output, target_smpl_output).cpu().item()
        print(f"MPJPE: {round(mpjpe_result,3)}, PA_MPJPE: {round(pa_mpjpe_result,3)}")
        file_name = 'exp_results.txt'
        with open(file_name, 'a', encoding='utf-8') as file:
            file.write(f"Evaluating {ds_name} Dataset. MPJPE: {round(mpjpe_result,3)}, PA_MPJPE: {round(pa_mpjpe_result,3)}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_family_id", type=str, default="posellava")
    parser.add_argument('--model_path', type=str, default='/yq/fengd/finetuning_model/stage3_model_weight')
    parser.add_argument("--eval_data_path", type=str, default="ds_config/eval_ds.json")
    parser.add_argument("--conv_style", type=str, default="llava_v1")
    parser.add_argument("--output_json", type=bool, default=False)
    parser.add_argument("--eval_metric", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    model_path = args.model_path
    model_family_id = args.model_family_id
    eval_data_path = args.eval_data_path
    device = args.device

    loader = LOADERS[model_family_id](
        model_local_path= model_path,
        model_max_length = 2048,
        compute_dtype= torch.float16 ,
        use_flash_attn= True,
        device_map= device,     
    )

    # Load pretrained model, tokenizer, and image processor
    model, tokenizer, processor = loader.load(load_lora_model=False)
    ds_collections = json.loads(open(eval_data_path).read())
    
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        dataset_cfg = ds_collections[ds_name]
        eval_single_dataset(ds_name, dataset_cfg, model, tokenizer, processor, args)
    print("########################################################")
    print("Finish all datasets evaluation")

    
    

    