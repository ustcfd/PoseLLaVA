import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .posemodal.modeling_poseclip import PoseEncoderTower, PoseDecoderTower
from .posemodal.configuration_pose import CLIPPoseConfig

class CLIPPoseEncoderTower(nn.Module):
    def __init__(self, args, delay_load=False):
        super().__init__()
        self.cfg_only = CLIPPoseConfig()
        self.is_loaded = False
        if not delay_load:
            self.load_model()
        # elif getattr(args, 'unfreeze_mm_vision_tower', False):
        #     self.load_model()
       
            # self.cfg_only = CLIPPoseConfig.from_pretrained(self.pose_tower_name)
            
    def load_model(self):
        pose_model_path = 'llava/poseclip_checkpoints/best_sampler_model.pt'
        # print(f'Load pose encoder tower from {pose_model_path}')
        self.pose_tower = PoseEncoderTower(self.cfg_only)
        pose_tower_weights = torch.load(pose_model_path, map_location='cpu')
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        self.pose_tower.load_state_dict(get_w(pose_tower_weights,'pose_enc'))
        self.pose_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, pose_forward_outs):
        pose_features = pose_forward_outs.last_hidden_state
        return pose_features
       
    # @torch.no_grad()
    def forward(self, smpl_poses):
        if type(smpl_poses) is list:
            pose_features = []
            for pose in smpl_poses:
                pose_forward_out = self.pose_tower(pose.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                pose_feature = self.feature_select(pose_forward_out).to(pose.dtype)
                pose_features.append(pose_feature)
        else:
            pose_forward_outs = self.pose_tower(smpl_poses.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            pose_features = self.feature_select(pose_forward_outs).to(smpl_poses.dtype)

        return pose_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.pose_tower.dtype

    @property
    def device(self):
        return self.pose_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.pose_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return 1




class CLIPPoseDecoderTower(nn.Module):
    def __init__(self, delay_load=False):
        super().__init__()
        self.cfg_only = CLIPPoseConfig()
        pose_model_path = 'llava/poseclip_checkpoints/poseclip_two_declayers_model.pt'
        print(f'Load pose decoder tower from {pose_model_path}')
        self.pose_tower = PoseDecoderTower(self.cfg_only)
        pose_tower_weights = torch.load(pose_model_path, map_location='cpu')
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        self.pose_tower.load_state_dict(get_w(pose_tower_weights,'pose_dec'))
        # self.pose_tower.requires_grad_(False)
    def forward(self, pose_features):
        return self.pose_tower(pose_features)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=24):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x, start=0):
        x = x + self.pe[:, start:(start + x.size(1))]
        return x

class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, njoints, nfeats, latent_dim=256, 
                 ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
                  activation="gelu", **kargs):
        super().__init__()
        
        self.njoints = njoints
        self.nfeats = nfeats

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.latent_dim))

        self.skelEmbedding = nn.Linear(self.nfeats, self.latent_dim)
        
        # 换成可学习的位置编码
        self.pos_embed = PositionalEncoding(self.latent_dim, max_len=24)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)
        
    def forward(self, x):
        bs, njoints, nfeats  = x.shape # [B,N,C] 
        # embedding of the skeleton
        x = self.skelEmbedding(x) # [B, N, latent_dim]
        pos = self.pos_embed(x)
        # xseq = [B, N+1, latent_dim]
        cls_tokens = self.cls_token.expand(bs, -1, -1)
        cls_pos = self.cls_pos.expand(bs, -1, -1)
        xseq = torch.cat((cls_tokens, x), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)  
        x_embedding = xseq + pos
        final = self.seqTransEncoder(x_embedding)
        # mu = final[ :, :1]
        return final


class PoseTower(nn.Module):
    def __init__(self, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.pose_encoder = Encoder_TRANSFORMER(njoints=24, nfeats=6, latent_dim=512, num_layers=6)
        # self.load_model()
        # if not delay_load:
        #     self.load_model()
        # else:
        #     self.cfg_only = None
         
    def load_model(self):
        pose_model_path = '/yq/fengd/posellava_clean/llava/poseclip_checkpoints/best_model.pt'
        print(f'Load pose tower from {pose_model_path}')
        pose_tower_weights = torch.load(pose_model_path, map_location='cpu')
        def get_w(weights, keyword):
            out_dict = {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            if not out_dict:
                raise ValueError("The loading pose_tower weight is empty")
            return out_dict
        
        self.pose_encoder.load_state_dict(get_w(pose_tower_weights,'encoder'))
        self.pose_encoder.requires_grad_(False)
        self.is_loaded = True

    # def load_model(self):
    #     pose_model_path = 'llava/poseclip_checkpoints/best_poseclip_model.pt'
    #     print(f'Load pose tower from {pose_model_path}')
    #     self.pose_tower = Encoder_TRANSFORMER(njoints=24, 
    #                                           nfeats=6, 
    #                                           latent_dim=768,
    #                                           num_layers=6,
    #                                           ff_size=3072,
    #                                           num_heads=8)                                       
    #     pose_tower_weights = torch.load(pose_model_path, map_location='cpu')
    #     def get_w(weights, keyword):
    #         return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
    #     self.pose_tower.load_state_dict(get_w(pose_tower_weights,'pose_enc'))
    #     self.pose_tower.requires_grad_(False)
    #     self.is_loaded = True
    
    def forward(self, smpl_paramers):
        return self.pose_encoder(smpl_paramers)
