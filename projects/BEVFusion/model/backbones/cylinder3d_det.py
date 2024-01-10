# Copyright (c) OpenMMLab. All rights reserved.
r"""Modified from Cylinder3D.

Please refer to `Cylinder3D github page
<https://github.com/xinge008/Cylinder3D>`_ for details
"""

from typing import List, Optional

import numpy as np
import torch
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.ops import (SparseConv3d, SparseConvTensor, SparseInverseConv3d,
                      SubMConv3d)
from mmengine.model import BaseModule
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType

from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule


from spconv.pytorch import SparseConvTensor as spconv_SparseConvTensor
from ...model.data_preprocessors.voxelize import VoxelizationByGridShapeDet


class AsymmResBlock(BaseModule):
    """Asymmetrical Residual Block.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
        act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='LeakyReLU').
        indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 indice_key: Optional[str] = None):
        super().__init__()

        self.conv0_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act0_0 = build_activation_layer(act_cfg)
        self.bn0_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv0_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act0_1 = build_activation_layer(act_cfg)
        self.bn0_1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act1_0 = build_activation_layer(act_cfg)
        self.bn1_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act1_1 = build_activation_layer(act_cfg)
        self.bn1_1 = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv0_0(x)

        shortcut.features = self.act0_0(shortcut.features)
        shortcut.features = self.bn0_0(shortcut.features)

        shortcut = self.conv0_1(shortcut)
        shortcut.features = self.act0_1(shortcut.features)
        shortcut.features = self.bn0_1(shortcut.features)

        res = self.conv1_0(x)
        res.features = self.act1_0(res.features)
        res.features = self.bn1_0(res.features)

        res = self.conv1_1(res)
        res.features = self.act1_1(res.features)
        res.features = self.bn1_1(res.features)

        res.features = res.features + shortcut.features

        return res


class AsymmeDownBlock(BaseModule):
    """Asymmetrical DownSample Block.

    Args:
       in_channels (int): Input channels of the block.
       out_channels (int): Output channels of the block.
       norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
       act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='LeakyReLU').
       pooling (bool): Whether pooling features at the end of
           block. Defaults: True.
       height_pooling (bool): Whether pooling features at
           the height dimension. Defaults: False.
       indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 pooling: bool = True,
                 height_pooling: bool = False,
                 indice_key: Optional[str] = None):
        super().__init__()
        self.pooling = pooling

        self.conv0_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act0_0 = build_activation_layer(act_cfg)
        self.bn0_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv0_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act0_1 = build_activation_layer(act_cfg)
        self.bn0_1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act1_0 = build_activation_layer(act_cfg)
        self.bn1_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act1_1 = build_activation_layer(act_cfg)
        self.bn1_1 = build_norm_layer(norm_cfg, out_channels)[1]

        if pooling:
            if height_pooling:
                self.pool = SparseConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    indice_key=indice_key,
                    bias=False)
            else:
                self.pool = SparseConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=(2, 2, 1),
                    padding=1,
                    indice_key=indice_key,
                    bias=False)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv0_0(x)
        shortcut.features = self.act0_0(shortcut.features)
        shortcut.features = self.bn0_0(shortcut.features)

        shortcut = self.conv0_1(shortcut)
        shortcut.features = self.act0_1(shortcut.features)
        shortcut.features = self.bn0_1(shortcut.features)

        res = self.conv1_0(x)
        res.features = self.act1_0(res.features)
        res.features = self.bn1_0(res.features)

        res = self.conv1_1(res)
        res.features = self.act1_1(res.features)
        res.features = self.bn1_1(res.features)

        res.features = res.features + shortcut.features

        if self.pooling:
            pooled_res = self.pool(res)
            return pooled_res, res
        else:
            return res


class AsymmeUpBlock(BaseModule):
    """Asymmetrical UpSample Block.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for
                normalization layer.
        act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
                Defaults to dict(type='LeakyReLU').
        indice_key (str, optional): Name of indice tables. Defaults to None.
        up_key (str, optional): Name of indice tables used in
            SparseInverseConv3d. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 indice_key: Optional[str] = None,
                 up_key: Optional[str] = None):
        super().__init__()

        self.trans_conv = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'new_up')
        self.trans_act = build_activation_layer(act_cfg)
        self.trans_bn = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.act1 = build_activation_layer(act_cfg)
        self.bn1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv2 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.act2 = build_activation_layer(act_cfg)
        self.bn2 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv3 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.act3 = build_activation_layer(act_cfg)
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]

        self.up_subm = SparseInverseConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            indice_key=up_key,
            bias=False)

    def forward(self, x: SparseConvTensor,
                skip: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        x_trans = self.trans_conv(x)
        x_trans.features = self.trans_act(x_trans.features)
        x_trans.features = self.trans_bn(x_trans.features)

        # upsample
        up = self.up_subm(x_trans)

        up.features = up.features + skip.features

        up = self.conv1(up)
        up.features = self.act1(up.features)
        up.features = self.bn1(up.features)

        up = self.conv2(up)
        up.features = self.act2(up.features)
        up.features = self.bn2(up.features)

        up = self.conv3(up)
        up.features = self.act3(up.features)
        up.features = self.bn3(up.features)

        return up


class DDCMBlock(BaseModule):
    """Dimension-Decomposition based Context Modeling.

    Args:
        in_channels (int): Input channels of the block.
        out_channels (int): Output channels of the block.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
        act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='Sigmoid').
        indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='Sigmoid'),
                 indice_key: Optional[str] = None):
        super().__init__()

        self.conv1 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.bn1 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act1 = build_activation_layer(act_cfg)

        self.conv2 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 1),
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.bn2 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act2 = build_activation_layer(act_cfg)

        self.conv3 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act3 = build_activation_layer(act_cfg)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv1(x)
        shortcut.features = self.bn1(shortcut.features)
        shortcut.features = self.act1(shortcut.features)

        shortcut2 = self.conv2(x)
        shortcut2.features = self.bn2(shortcut2.features)
        shortcut2.features = self.act2(shortcut2.features)

        shortcut3 = self.conv3(x)
        shortcut3.features = self.bn3(shortcut3.features)
        shortcut3.features = self.act3(shortcut3.features)
        shortcut.features = shortcut.features + \
            shortcut2.features + shortcut3.features

        shortcut.features = shortcut.features * x.features

        return shortcut



class AsymmeZaxisDownBlock(BaseModule):
    """Asymmetrical DownSample Block.

    Args:
       in_channels (int): Input channels of the block.
       out_channels (int): Output channels of the block.
       norm_cfg (:obj:`ConfigDict` or dict): Config dict for
            normalization layer.
       act_cfg (:obj:`ConfigDict` or dict): Config dict of activation layers.
            Defaults to dict(type='LeakyReLU').
       pooling (bool): Whether pooling features at the end of
           block. Defaults: True.
       height_pooling (bool): Whether pooling features at
           the height dimension. Defaults: False.
       indice_key (str, optional): Name of indice tables. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType,
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 pooling: bool = True,
                 indice_key: Optional[str] = None):
        super().__init__()
        self.pooling = pooling

        self.conv0_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act0_0 = build_activation_layer(act_cfg)
        self.bn0_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv0_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act0_1 = build_activation_layer(act_cfg)
        self.bn0_1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_0 = SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act1_0 = build_activation_layer(act_cfg)
        self.bn1_0 = build_norm_layer(norm_cfg, out_channels)[1]

        self.conv1_1 = SubMConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 3),
            padding=1,
            bias=False,
            indice_key=indice_key + 'bef')
        self.act1_1 = build_activation_layer(act_cfg)
        self.bn1_1 = build_norm_layer(norm_cfg, out_channels)[1]

        self.pool = SparseConv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            padding=1,
            indice_key=indice_key,
            bias=False)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Forward pass."""
        shortcut = self.conv0_0(x)
        shortcut.features = self.act0_0(shortcut.features)
        shortcut.features = self.bn0_0(shortcut.features)

        shortcut = self.conv0_1(shortcut)
        shortcut.features = self.act0_1(shortcut.features)
        shortcut.features = self.bn0_1(shortcut.features)

        res = self.conv1_0(x)
        res.features = self.act1_0(res.features)
        res.features = self.bn1_0(res.features)

        res = self.conv1_1(res)
        res.features = self.act1_1(res.features)
        res.features = self.bn1_1(res.features)

        res.features = res.features + shortcut.features

        pooled_res = self.pool(res)

        return pooled_res



@MODELS.register_module()
class Asymm3DSpconvDet(BaseModule):
    """Asymmetrical 3D convolution networks.

    Args:
        grid_size (int): Size of voxel grids.
        input_channels (int): Input channels of the block.
        base_channels (int): Initial size of feature channels before
            feeding into Encoder-Decoder structure. Defaults to 16.
        backbone_depth (int): The depth of backbone. The backbone contains
            downblocks and upblocks with the number of backbone_depth.
        height_pooing (List[bool]): List indicating which downblocks perform
            height pooling.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01)).
        init_cfg (dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 grid_size: int,
                 input_channels: int,
                 base_channels: int = 16,
                 backbone_depth: int = 4,
                 height_pooing: List[bool] = [True, True, False, False],
                 norm_cfg: ConfigType = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 init_cfg=None,
                 zaxis_down_sampling_channels = [(16*2, 16*2), (16*2, 64), (64, 128), (128, 64)],
                 cyl_voxel_layer=None,
                 cart_voxel_layer=None,
                 cyl2bev=True):
        super().__init__(init_cfg=init_cfg)

        self.cyl_voxel_layer = VoxelizationByGridShapeDet(**cyl_voxel_layer)
        self.cart_voxel_layer = VoxelizationByGridShapeDet(**cart_voxel_layer)
        self.cyl2bev = cyl2bev
        self.grid_size = grid_size
        self.backbone_depth = backbone_depth
        self.down_context = AsymmResBlock(
            input_channels, base_channels, indice_key='pre', norm_cfg=norm_cfg)

        self.down_block_list = torch.nn.ModuleList()
        self.up_block_list = torch.nn.ModuleList()
        for i in range(self.backbone_depth):
            self.down_block_list.append(
                AsymmeDownBlock(
                    2**i * base_channels,
                    2**(i + 1) * base_channels,
                    height_pooling=height_pooing[i],
                    indice_key='down' + str(i),
                    norm_cfg=norm_cfg))
            if i == self.backbone_depth - 1:
                self.up_block_list.append(
                    AsymmeUpBlock(
                        2**(i + 1) * base_channels,
                        2**(i + 1) * base_channels,
                        up_key='down' + str(i),
                        indice_key='up' + str(self.backbone_depth - 1 - i),
                        norm_cfg=norm_cfg))
            else:
                self.up_block_list.append(
                    AsymmeUpBlock(
                        2**(i + 2) * base_channels,
                        2**(i + 1) * base_channels,
                        up_key='down' + str(i),
                        indice_key='up' + str(self.backbone_depth - 1 - i),
                        norm_cfg=norm_cfg))

        self.ddcm = DDCMBlock(
            2 * base_channels,
            2 * base_channels,
            indice_key='ddcm',
            norm_cfg=norm_cfg)

        # self.z_down_block_list = torch.nn.ModuleList()
        # for i, zaxis_channel in enumerate(zaxis_down_sampling_channels):
        #     self.z_down_block_list.append(
        #         AsymmeZaxisDownBlock(
        #             zaxis_channel[0],
        #             zaxis_channel[1],
        #             indice_key='down' + str(i),
        #             norm_cfg=norm_cfg))

    def forward(self, voxel_features: Tensor, coors: Tensor,
                batch_size: int) -> SparseConvTensor:
        """Forward pass."""
        coors = coors.int()
        ret = SparseConvTensor(voxel_features, coors, np.array(self.grid_size),
                               batch_size)
        ret = self.down_context(ret)

        down_skip_list = []
        down_pool = ret
        for i in range(self.backbone_depth):
            down_pool, down_skip = self.down_block_list[i](down_pool)
            down_skip_list.append(down_skip)

        up = down_pool
        for i in range(self.backbone_depth - 1, -1, -1):
            up = self.up_block_list[i](up, down_skip_list[i])

        ddcm = self.ddcm(up)
        ddcm.features = torch.cat((ddcm.features, up.features), 1)  # still is sparse feature

        # here We need to convert to 2d dense feature
        # first we need to downsample z axis
        # Very unelegant Very unelegant Very unelegant
        # ddcm = spconv_SparseConvTensor(ddcm.features, coors, np.array(self.grid_size),
        #                        batch_size)
        
        # for zaixs_down_conv in self.z_down_block_list:
        #     ddcm = zaixs_down_conv(ddcm)


        cyl_spatial_features = ddcm.dense()

        N, C, H, W, D = cyl_spatial_features.shape
        cyl_spatial_features = cyl_spatial_features.permute(0, 1, 4, 2, 3).contiguous()  # must add .contiguous()
        cyl_spatial_features = cyl_spatial_features.view(N, C * D, H, W)

        # we need to convert to 2d bev feature
        if self.cyl2bev:
            cart_H, cart_W, _ = (self.cart_voxel_layer.grid_shape).int()
            # we need to get cart voxel coords
            cart_H_tensor, cart_W_tensor = torch.meshgrid(torch.arange(cart_H), torch.arange(cart_W).T)

            cart_H_tensor = cart_H_tensor.to(cyl_spatial_features.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, cart_H, cart_W, 1)
            cart_W_tensor = cart_W_tensor.to(cyl_spatial_features.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, cart_H, cart_W, 1)

            batch_idx = torch.arange(N).to(cyl_spatial_features.device).view(N, 1, 1, 1).expand(N, cart_H, cart_W, 1)
            
            # voxel_spatical_cart_pos = torch.cat([batch_idx, cart_H_tensor, cart_W_tensor], dim=-1)

            # get real voxel coords
            tx, ty, _ = self.cart_voxel_layer.voxel_size
            cart_point_cloud_range = self.cart_voxel_layer.point_cloud_range

            cart_x_coords = cart_H_tensor * tx + cart_point_cloud_range[0] + tx / 2
            cart_y_coords = cart_W_tensor * ty + cart_point_cloud_range[1] + tx / 2

            # get the coorsponding real cyl coords cyl_spatical_voxel_pos
            rho = torch.sqrt(cart_x_coords**2 + cart_y_coords**2)
            phi = torch.atan2(cart_y_coords, cart_x_coords)

            polar_res = torch.cat([rho, phi], dim=-1)

            min_bound = polar_res.new_tensor(
                self.cyl_voxel_layer.point_cloud_range[:2])
            max_bound = polar_res.new_tensor(
                self.cyl_voxel_layer.point_cloud_range[3:5])
            
            try:  # only support PyTorch >= 1.9.0
                polar_res_clamp = torch.clamp(polar_res, min_bound,
                                                max_bound)
            except TypeError:
                polar_res_clamp = polar_res.clone()
                for coor_idx in range(2):
                    polar_res_clamp[:, coor_idx][
                        polar_res[:, coor_idx] >
                        max_bound[coor_idx]] = max_bound[coor_idx]
                    polar_res_clamp[:, coor_idx][
                        polar_res[:, coor_idx] <
                        min_bound[coor_idx]] = min_bound[coor_idx]
                    
            res_coors = torch.floor(
                (polar_res_clamp - min_bound) / polar_res_clamp.new_tensor(
                    self.cyl_voxel_layer.voxel_size[:2])).int()
            

            voxel_spatical_cyl_pos = torch.cat([batch_idx, res_coors], dim=-1).contiguous()

            # boardcast inds，for using scatter_add
            # each col are same inds
            # casue inds is too large ,we should use for ...
            cart_spatical_feature_list = []
            for i in range(N):
                # first we should generate cart voxel features
                
                # cur_cart_spatical_features = torch.zeros((cart_H, cart_W, C * D), device=cyl_spatial_features.device).view(cart_H*cart_W, C * D)

                # gt cur 
                cur_voxel_spatical_cyl_pos = voxel_spatical_cyl_pos[i]
                H, W, C = cur_voxel_spatical_cyl_pos.shape
                cur_voxel_spatical_cyl_pos = cur_voxel_spatical_cyl_pos.view(H*W, C)

                cur_cyl_spatial_features = cyl_spatial_features[i]
                C, H, W = cur_cyl_spatial_features.shape
                cur_cyl_spatial_features = cur_cyl_spatial_features.permute(1, 2, 0).view(H*W, C)

                cur_index = cur_voxel_spatical_cyl_pos[:, 1] * self.cyl_voxel_layer.grid_shape[1] + cur_voxel_spatical_cyl_pos[:, 2]

                # [cart_H*cart_W, C] -> [cart_H, cart_W, C] -> [C, cart_H, cart_W]
                cart_spatical_feature_list.append(cur_cyl_spatial_features[cur_index.long()].view(cart_H, cart_W, C).permute(2, 0, 1))

                ######################## error should reference SparseKD ########################
                # here We don't need to use scatter_add
                # cur_index = cur_index.unsqueeze(1).expand(-1, C).long()
                # cart_spatical_feature_list.append(cur_cart_spatical_features.scatter_add(0, cur_index, cur_cyl_spatial_features).view(H, W, C))
            
            cart_spatical_features = torch.stack(cart_spatical_feature_list,dim=0).contiguous()

            return cart_spatical_features
            
        else:

            return cyl_spatial_features       
