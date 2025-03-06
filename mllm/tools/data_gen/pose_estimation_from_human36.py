import glob
import pickle
from typing import List
import cdflib
import cv2
import numpy as np
from tqdm import tqdm

import json
import os
import os.path as osp
import shutil
import random
import argparse


class H36mConverter():
    """Human3.6M dataset
    `Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human
    Sensing in Natural Environments' TPAMI`2014
    More details can be found in the `paper
    <http://vision.imar.ro/human3.6m/pami-h36m.pdf>`__.

    Args:
        modes (list): 'valid' or 'train' for accepted modes
        protocol (int): 1 or 2 for available protocols
        extract_img (bool): Store True to extract images into a separate
        folder. Default: False.
        mosh_dir (str, optional): Path to directory containing mosh files.
    """
    ACCEPTED_MODES = ['valid', 'train']

    def __init__(self,
                 modes: List = [],
                 protocol: int = 1,
                 extract_img: bool = False,
                 mosh_dir=None) -> None:
        # super(H36mConverter, self).__init__(modes)
        accepted_protocol = [1, 2]
        if protocol not in accepted_protocol:
            raise ValueError('Input protocol not in accepted protocol. \
                Use either 1 or 2')
        self.protocol = protocol
        self.extract_img = extract_img
        # self.get_mosh = False
        # if mosh_dir is not None and os.path.exists(mosh_dir):
        self.get_mosh = True
        self.mosh_dir = mosh_dir
        self.camera_name_to_idx = {
            '54138969': 0,
            '55011271': 1,
            '58860488': 2,
            '60457274': 3,
        }

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:

        print(f'**********{mode}***********')

        image_path_, bbox_xywh_, keypoints2d_, keypoints3d_ = [], [], [], []
        smpl = {}
        human_data = []
        smpl['body_pose'] = []
        smpl['global_orient'] = []
        smpl['betas'] = []

        # choose users ids for different set
        if mode == 'train':
            user_list = [1, 5, 6, 7, 8]
        elif mode == 'valid':
            user_list = [9, 11]

        # go over each user
        count = 0
        for user_i in tqdm(user_list, desc='user id'):
            user_name = f'S{user_i}'
            pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                     'D3_Positions_mono')
            # go over all the sequences of each user
            seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
            seq_list.sort()

            # mosh path
            if self.get_mosh:
                mosh_path = os.path.join(self.mosh_dir, user_name)

            for seq_i in tqdm(seq_list, desc='sequence id'):

                # sequence info
                seq_name = seq_i.split('/')[-1]
                action, camera, _ = seq_name.split('.')
                action_raw = action
                action = action.replace(' ', '_')
                # irrelevant sequences
                if action == '_ALL':
                    continue
                # # 3D pose file
                poses_3d = cdflib.CDF(seq_i)['Pose'][0]

                # 3D mosh file
                if self.get_mosh:
                    mosh_name = '%s_cam%s_aligned.pkl' % (
                        action_raw, self.camera_name_to_idx[camera])
                    mosh_file = os.path.join(mosh_path, mosh_name)
                    if os.path.exists(mosh_file):
                        with open(mosh_file, 'rb') as file:
                            mosh_data = pickle.load(file, encoding='latin1')
                    else:
                        print(f'mosh file {mosh_name} is missing')
                        continue
                    thetas = mosh_data['new_poses']
                    betas = mosh_data['betas']

                for frame_i in tqdm(range(poses_3d.shape[0]), desc='frame id'):

                    if frame_i % 5 == 0 and (self.protocol == 1
                                             or camera == '60457274'):
                        # image name
                        seq_id = f'{user_name}_{action}'
                        image_name = f'{seq_id}.{camera}_{frame_i + 1:06d}.jpg'
                        img_folder_name = f'{user_name}_{action}.{camera}'
                        image_path = os.path.join(dataset_path, user_name, 'images',
                                                  img_folder_name, image_name)

                        image_path_.append(image_path)

                        # get mosh data
                        if self.get_mosh:
                            pose = thetas[frame_i // 5, :]
                            R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
                            R_root = cv2.Rodrigues(pose[:3])[0]
                            new_root = R_root.dot(R_mod)
                            pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
                            smpl['body_pose'] = pose[:72]
                            json_data = {
                                "id": count,
                                "image": image_name,
                                "pose": smpl['body_pose'].tolist()
                            }
                            human_data.append(json_data)
                            count += 1

        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        print(count)

        if mode == 'train':
            if self.get_mosh:
                out_file = os.path.join(out_path, 'h36m_mosh_train.json')
            else:
                out_file = os.path.join(out_path, 'h36m_train.json')
        elif mode == 'valid':
            out_file = os.path.join(out_path,
                                    f'h36m_mosh_valid.json')
        # human_data.dump(out_file)
        with open(out_file, 'w') as output_file:
            json.dump(human_data, output_file, indent=4)


def find_and_output_substring(input_string):
    # 寻找第一个下划线的索引位置
    underscore_index = input_string.find('_')
    # 如果找到下划线，则输出从字符串开头到下划线索引位置的子字符串
    if underscore_index != -1:
        output_substring = input_string[:underscore_index]
        return output_substring


def generate_chunk_data(chunk_data, h36_root_dir, chunk_save_dir, chunk_id=None):
    '''
        chunk_data : 每份chunk_data ,
        h36_root_dir: h36数据集所在的根目录,
        chunk_save_dir : 该chunk data要保存的文件夹路径
    '''
    questions = [
        "<image> Can you predict the SMPL pose of the person in this image?",
        "<image> There is a person in the middle of the image, please output this person's SMPL pose.",
        "<image> What is the human pose in this image? Please respond with SMPL pose.",
        "<image> What is the person doing in this image? Please output SMPL pose.",
        "<image> There is a person in the middle of the image, use SMPL to describe the pose."
    ]

    answers = [
        "The SMPL pose is <POSE>.",
        "It is <POSE>.",
        "The SMPL format of this person's pose is <POSE>.",
        "Sure, it is <POSE>.",
        "Sure, the SMPL pose is <POSE>.",
        "The SMPL pose of the person is <POSE>.",
        "Sure, <POSE>."
    ]
    num_empty = 0
    dist_image_dir = osp.join(chunk_save_dir, 'images')  # images目录
    if not osp.exists(dist_image_dir):
        # 如果不存在，则递归创建目录
        os.makedirs(dist_image_dir)
        print(f"目录 '{dist_image_dir}' 创建成功")

    if chunk_id is not None:
        json_file = osp.join(chunk_save_dir, 'chunk_data.json')
    else:
        json_file = osp.join(chunk_save_dir, f'chunk_data_{chunk_id}.json')

    chunk_data_results = []
    for data in chunk_data:
        image_name = data['image']
        # 从右边找到第一个下划线的索引
        last_underscore_index = image_name.rfind('_')
        # 从左边找到第一个下划线的索引
        second_dir = find_and_output_substring(image_name)
        # image_dir = f'/datanfs/zsy/projects/mmhuman3d/data/datasets/h36m/{second_dir}/images/'
        image_dir = osp.join(h36_root_dir, second_dir, 'images')
        image_parant = image_name[:last_underscore_index]
        image_path = osp.join(image_dir, image_parant, image_name)
        dist_path = osp.join(dist_image_dir, image_name)

        if os.path.exists(image_path):
            # 将image copy到指定文件夹
            shutil.copy(image_path, dist_path)
            question = random.choice(questions)
            answer = random.choice(answers)
            conversation = [{"from": "human", "value": question},
                            {"from": "gpt", "value": answer}]
            single_data = {
                "id": data['id'],
                'image': image_name,
                "pose": data['pose'],
                "conversations": conversation
            }
            chunk_data_results.append(single_data)
        else:
            num_empty += 1

    # 保存json数据
    json.dump(chunk_data_results, open(json_file, 'w'))
    return num_empty


def save_chunk_data(all_human_data, human36_dir, posedata_save_dir, chunk_size=50000):
    # 所有图片放在同一个文件夹会太大，可以按照chunk_size分成多个chunk
    with open(all_human_data, 'r') as f:
        all_data = json.load(f)
        print("all data samples:", len(all_data))

    total_records = len(all_data)
    chunks = (total_records + chunk_size - 1) // chunk_size
    for i in range(1, chunks + 1):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, total_records)
        data_chunk_i = all_data[start_index:end_index]
        current_chunk_dir = os.path.join(posedata_save_dir, f'chunk_{i}')
        if not os.path.exists(current_chunk_dir):
            # 如果不存在，则递归创建目录
            os.makedirs(current_chunk_dir)
            print(f"目录 '{current_chunk_dir}' 创建成功")
        num_em = generate_chunk_data(data_chunk_i, human36_dir, current_chunk_dir, i)
        print(f"当前chunk图片路径不匹配的数目:{num_em}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mosh-dir', type=str, help='human3.6 mesh文件存放路径',
                        default='')
    parser.add_argument('--input-dir', type=str, help='human3.6 Images存放路径',
                        default='')
    parser.add_argument('--output-dir', type=str, help='处理后的human3.6json文件存放路径',
                        default='')
    parser.add_argument('--imagepose-data-dir', type=str, help='整理后的image2pose数据集最终存放路径',
                        default='')
    parser.add_argument('--chunk-size', type=int,
                        help='human3.6太大, 可根据chunk-size分成多个chunk',
                        default=50000)
    args = parser.parse_args()

    human36_image_dir = args.input_dir
    output_human_data = args.output_dir
    chunk_size = args.chunk_size

    # ------------process human3.6m, 生成json文件------------
    humanConv = H36mConverter()
    humanConv.protocol = 1
    humanConv.extract_img = False
    humanConv.prefix = 'h36m'
    humanConvMode = "train"  # accept mode :train or valid

    # human3.6文件存放路径
    humanConv.mosh_dir = args.hm36_mosh_dir
    humanConv.convert_by_mode(human36_image_dir, output_human_data, humanConvMode)

    # 所有图片放在同一个文件夹会太大，可以按照chunk_size分成多个chunk
    # generate chunk data
    all_human_data = os.path.join(output_human_data, 'h36m_mosh_{}.json'.format(humanConvMode))
    imagepose_data_save_dir = args.imagepose_data_dir
    save_chunk_data(all_human_data, human36_image_dir, imagepose_data_save_dir, chunk_size)