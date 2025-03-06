import torch
import numpy as np
from smplx import SMPL as _SMPL
# from smplx.body_models import ModelOutput
from smplx.lbs import vertices2joints
from smplx.utils import ModelOutput

import evaluation.smpl.smpl_config as config

class SMPL(_SMPL):
    """
    Extension of the official SMPL (from the smplx python package) implementation to
    support more joints.
    """
    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        J_regressor_extra = np.load(config.J_REGRESSOR_EXTRA_PATH)
        J_regressor_cocoplus = np.load(config.COCOPLUS_REGRESSOR_PATH)
        J_regressor_h36m = np.load(config.H36M_REGRESSOR_PATH)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra,
                                                               dtype=torch.float32))
        self.register_buffer('J_regressor_cocoplus', torch.tensor(J_regressor_cocoplus,
                                                                  dtype=torch.float32))
        self.register_buffer('J_regressor_h36m', torch.tensor(J_regressor_h36m,
                                                              dtype=torch.float32))

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        cocoplus_joints = vertices2joints(self.J_regressor_cocoplus, smpl_output.vertices)
        h36m_joints = vertices2joints(self.J_regressor_h36m, smpl_output.vertices)
        # print(smpl_output.joints.shape)
        # print(self.J_regressor_extra.shape, extra_joints.shape)  
        # print(self.J_regressor_cocoplus.shape,cocoplus_joints.shape)      
        # print(self.J_regressor_h36m.shape, h36m_joints.shape)     

        all_joints = torch.cat([smpl_output.joints, extra_joints, cocoplus_joints,
                                h36m_joints], dim=1)
        smpl_output.joints = all_joints
        # output = ModelOutput(vertices=smpl_output.vertices,
        #                      global_orient=smpl_output.global_orient,
        #                      body_pose=smpl_output.body_pose,
        #                      joints=all_joints,
        #                      betas=smpl_output.betas,
        #                      full_pose=smpl_output.full_pose)
        return h36m_joints

# from typing import Optional   
# import smplx
# from smplx.utils import SMPLOutput
# i
# class SMPL(smplx.SMPLLayer):
#     def __init__(self, *args, joint_regressor_extra: Optional[str] = None, update_hips: bool = False, **kwargs):
#         """
#         Extension of the official SMPL implementation to support more joints.
#         Args:
#             Same as SMPLLayer.
#             joint_regressor_extra (str): Path to extra joint regressor.
#         """
#         super(SMPL, self).__init__(*args, **kwargs)
#         smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
#                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
            
#         if joint_regressor_extra is not None:
#             self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
#         self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))
#         self.update_hips = update_hips

#     def forward(self, *args, **kwargs) -> SMPLOutput:
#         """
#         Run forward pass. Same as SMPL and also append an extra set of joints if joint_regressor_extra is specified.
#         """
#         smpl_output = super(SMPL, self).forward(*args, **kwargs)
#         joints = smpl_output.joints[:, self.joint_map, :]
#         if self.update_hips:
#             joints[:,[9,12]] = joints[:,[9,12]] + \
#                 0.25*(joints[:,[9,12]]-joints[:,[12,9]]) + \
#                 0.5*(joints[:,[8]] - 0.5*(joints[:,[9,12]] + joints[:,[12,9]]))
#         if hasattr(self, 'joint_regressor_extra'):
#             extra_joints = vertices2joints(self.joint_regressor_extra, smpl_output.vertices)
#             joints = torch.cat([joints, extra_joints], dim=1)
#         smpl_output.joints = joints
#         return smpl_output
    
