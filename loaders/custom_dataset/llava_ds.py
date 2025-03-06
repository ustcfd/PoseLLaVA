import torch
import transformers
import json
import gc
import os
import sys
import traceback
import random
import numpy as np
from typing import Dict,  Sequence
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError

from torch.utils.data import Dataset
from transformers.image_processing_utils import BaseImageProcessor
from copy import deepcopy
from packaging import version
import tokenizers
from mllm.utils.mm_utils import expand2square
from mllm.conversation import get_conv_template
from mllm.models.constants.llava_constants import IGNORE_INDEX
from mllm.utils.mm_utils import tokenizer_image_token


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

class CustomLlavaSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                ds_name,
                template_name,
                meta,  
                tokenizer: transformers.PreTrainedTokenizer,
                image_processor: BaseImageProcessor,
                pad2square=True,
                repeat_time=1, 
                random_seed=0,
                group_by_length=False,
                dynamic_image_size =False,
                use_thumbnail =False,
                min_dynamic_patch=1,
                max_dynamic_patch=6,):
        # meta 数据集元数据
        super(CustomLlavaSupervisedDataset, self).__init__()
        
        self.ds_name = ds_name
        self.template_name = template_name
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_size = self.image_processor.crop_size["height"]
        self.pad2square = pad2square

        with open(meta['annotation'], 'r') as f:
            if meta['annotation'].endswith('.json'):
                self.raw_data = json.load(f)
            elif meta['annotation'].endswith('.jsonl'):
                self.raw_data = [json.loads(line) for line in f.readlines()]
            else:
                raise ValueError(f"Unsupported annotation file format: {meta['annotation']}")

            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)
        gc.collect()
        self.root = meta['root']
        
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                # Compute token length using the tokenizer
                conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                str_length = len(conversations)
                if str_length not in self.conv2length:
                    token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                    ).input_ids.size(1)
                    self.conv2length[str_length] = token_length
                else:
                    token_length = self.conv2length[str_length]
                self.length.append(token_length)
        gc.collect()
               
    def __len__(self):
        return len(self.raw_data)
    
    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        # if self.tcs_loader is not None and 's3://' in image_path:
        #     return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')
    
    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        if i >= len(self.raw_data):
            i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration 
            try:
                data_item = self.raw_data[i]
                ret = self.multi_modal_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()
                data_item = self.raw_data[i]
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def multi_modal_get_item(self, data_item) -> Dict[str, torch.Tensor]:
        if 'image' in data_item:
            if isinstance(data_item["image"], list):
                image_sources = data_item["image"]
            elif isinstance(data_item["image"], str):
                image_sources = [data_item["image"]]
            else:
                raise ValueError(f"Invalid image source type: {type(data_item['image'])}")
            num_image = len(image_sources)
            images = []
            for image_path in image_sources:
                image_path = self.get_image_path(image_path)
                image = self.load_image(image_path)
                if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                    image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                            max_num=self.max_dynamic_patch // num_image,
                                            image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                    images += image
                elif self.pad2square:
                    image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                    images.append(image) 
                else: # Otherwise, use the original image as a single patch
                    images.append(image)   
        else:
            crop_size = self.image_processor.crop_size
            # Create a blank white image
            image = Image.new('RGB', (crop_size["height"], crop_size["width"]), (255, 255, 255))
            images = [image]
        
        pixel_values = [self.image_processor.preprocess(image,return_tensors='pt')['pixel_values'][0] for image in images] 
        pixel_values = torch.stack(pixel_values)  
        text_only = False if 'image' in data_item else True
        ret = proprecess_llavadata(
                    self.template_name,[deepcopy(data_item['conversations'])],
                    self.tokenizer, text_only= text_only,
                    group_by_length=self.group_by_length,
                    ds_name=self.ds_name)
        
        if 'target_pose' in data_item:
            target_pose_list = data_item['target_pose']
            gt_pose_values = torch.from_numpy(np.array(target_pose_list, dtype=np.float32)).reshape(-1,72)
        else:
            gt_pose_values = torch.zeros(1,72)

        # ret = proprecess_interlm(
        #     self.template_name,[deepcopy(data_item['conversations'])],
        #     self.tokenizer, text_only= text_only,
        #     group_by_length=self.group_by_length,ds_name=self.ds_name)


        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values = pixel_values,
            gt_pose_values = gt_pose_values
        )
        return ret
    


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images



def proprecess_llavadata(template_name,
                    sources,
                    tokenizer: transformers.PreTrainedTokenizer,
                    text_only: bool = False,
                    group_by_length: bool = False,
                    ds_name: str = None)-> Dict:
    
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            sentence['value'] = sentence['value'].strip()
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())
    # Tokenize conversations

    if not text_only:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding= False if group_by_length else 'max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if not text_only:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. This dataset is {ds_name}" )
    
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )




def proprecess_interlm(template_name,
                    sources,
                    tokenizer: transformers.PreTrainedTokenizer,
                    text_only: bool = False,
                    group_by_length: bool = False,
                    ds_name: str = None)-> Dict:
    
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            sentence['value'] = sentence['value'].strip()
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())
    # Tokenize conversations
    input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding= False if group_by_length else 'max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
    ).input_ids
    
    
    targets = input_ids.clone()
    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) # 浦语里面 pad_token_id = eos_token_id
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX # <s>
        parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1  # 去除tokenizer的<s>
        target[cur_len: cur_len + temp_len] = IGNORE_INDEX
        cur_len = cur_len + temp_len
        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len: cur_len + temp_len] = IGNORE_INDEX
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len = cur_len + temp_len
        target[cur_len:] = IGNORE_INDEX

        # if False:  # Inspect and check the correctness of masking
        #     z = target.clone()
        #     z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
        #     print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. This dataset is {ds_name}.')
                sys.stdout.flush()
        
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
