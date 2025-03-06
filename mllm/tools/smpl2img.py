import json
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose
import numpy as np
import torch
import os.path as osp
from PIL import Image
import shutil
import cv2

"""
可视化组件基于mmhuman3d，需要安装相关环境：安装步骤参考https://mmhuman3d.readthedocs.io/en/latest/install.html
为了方便对image进行灵活命名 ，安装成功后，需要修改mmhuman3d一处代码：
（1）mmhuman3d/core/renderer/torch3d_renderer/smpl_renderer.py 注释掉261-262行代码：
cv2.imwrite(
        osp.join(folder, self.out_img_format % real_idx), im)
"""




def find_white_edges(image):
    """
    找到图像的白边

    Parameters:
        image (PIL.Image.Image): PIL图像对象

    Returns:
        tuple: 包含左、上、右、下白边的元组 (left, top, right, bottom)
    """
    # 转换为灰度图像
    grayscale_image = image.convert("L")

    # 获取图像的大小
    width, height = grayscale_image.size

    # 初始化边界值
    left, top, right, bottom = width, height, 0, 0

    # 遍历图像像素，找到非白色像素的边界
    for x in range(width):
        for y in range(height):
            if grayscale_image.getpixel((x, y)) != 255:  # 非白色像素
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    return left, top, right, bottom


def crop_image_with_white_edges(image_path, save_path, padding=100):
    # 打开图像
    image = Image.open(image_path)

    # 找到白边
    left, top, right, bottom = find_white_edges(image)

    # 裁剪区域
    crop_left = max(0, left - padding)
    crop_top = max(0, top - padding)
    crop_right = min(image.width, right + padding)
    crop_bottom = min(image.height, bottom + padding)

    # 裁剪图像
    cropped_image = image.crop((crop_left, crop_top, crop_right, crop_bottom))

    # 保存裁剪后的图像
    cropped_image.save(save_path)


if __name__ == "__main__":
    file_path = '/datanfs/fengd/posegpt_dataset/posefix_data/pose_pair_data/all_data.json'
    all_data = json.load(open(file_path, 'r'))
    model_path = "support_dir"
    device = torch.device('cpu')
    body_model_config = dict(
        type='smpl',
        model_path=model_path)
    PoseA_dir = '/datanfs/fengd/posegpt_dataset/posefix_data/org_pose_images'
    output_path = '/datanfs/fengd/posegpt_dataset/posefix_data/all_data/New_Pose_pair'
    # out_img_format=f'%04d_{data_type}.png'
    i = 0
    for key, info in all_data.items():
        pose_A = info['pose_A']
        pose_A = np.array(pose_A)
        pose_A = torch.from_numpy(pose_A).to(dtype=torch.float32, device=device)
        pose_A = pose_A.reshape(-1, 72)
        # pose_B = info['pose_B']
        # pose_B = np.array(pose_B)
        # pose_B = torch.from_numpy(pose_B).to(dtype=torch.float32, device=device)
        # pose_B = pose_B.reshape(-1,72)
        # visualize_smpl_pose(
        #     poses=Pose_A,
        #     output_path= output_path,
        #     resolution=(1024, 1024),
        #     verbose= True,
        #     batch_size=1,
        #     img_format=f'{key}_B.png',
        #     overwrite = True,
        #     body_model_config=body_model_config)

        return_tensors = visualize_smpl_pose(
            poses=pose_A,
            output_path=output_path,
            resolution=(1024, 1024),
            verbose=True,
            batch_size=1,
            overwrite=True,
            return_tensor=True, # 设为True,方便自定义命名存储
            body_model_config=body_model_config)

        # save_path = osp.join(output_path, f'{key}_B.png')
        # crop_image_with_white_edges(save_path, save_path, padding=100)
        # Pose_A_scoure = osp.join(PoseA_dir, f'{key}.png')
        # Pose_A_target = osp.join(output_path, f'{key}_A.png')
        # shutil.copy(Pose_A_scoure, Pose_A_target)
        # print(f"############################### finish {key}_B.png")
        return_tensors = return_tensors.cpu().numpy()
        batch_size = return_tensors.shape[0]
        # print(return_tensors)

        if return_tensors is not None:
            # for frame_id in range(batch_size):
            im = return_tensors[0]
            cv2.imwrite(osp.join(PoseA_dir, f'{key}.png'), im)
        i += 1
        print(f"Finished {i} images")
    print("Finished")
