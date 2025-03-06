import os
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SigLipVisionTower
from .pose_encoder import PoseTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    
    if 'clip' in vision_tower.lower():
        if is_absolute_path_exists or vision_tower.startswith('openai') or vision_tower.startswith('laion'):
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "sharegpt4v" in vision_tower.lower():
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'siglip' in vision_tower.lower():
        if is_absolute_path_exists or vision_tower.startswith("google") or vision_tower.startswith('bczhou'):
            return SigLipVisionTower(vision_tower, vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    

def build_pose_tower(pose_tower_cfg, **kwargs):
    return PoseTower(pose_tower_cfg, **kwargs)