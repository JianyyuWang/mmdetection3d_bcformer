from typing import Dict, List, Optional, Sequence, Tuple, Union

from mmdet3d.models.data_preprocessors.voxelize import voxelization
from torch import nn
from torch.nn.modules.utils import _pair
import torch


# different from VoxelizationByGridShape is we using grid_shape rather than grid_shape-1
class VoxelizationByGridShapeDet(nn.Module):
    """Voxelization that allows inferring voxel size automatically based on
    grid shape.

    Please refer to `Point-Voxel CNN for Efficient 3D Deep Learning
    <https://arxiv.org/abs/1907.03739>`_ for more details.

    Args:
        point_cloud_range (list):
            [x_min, y_min, z_min, x_max, y_max, z_max]
        max_num_points (int): max number of points per voxel
        voxel_size (list): list [x, y, z] or [rho, phi, z]
            size of single voxel.
        grid_shape (list): [L, W, H], grid shape of voxelization.
        max_voxels (tuple or int): max number of voxels in
            (training, testing) time
        deterministic: bool. whether to invoke the non-deterministic
            version of hard-voxelization implementations. non-deterministic
            version is considerablly fast but is not deterministic. only
            affects hard voxelization. default True. for more information
            of this argument and the implementation insights, please refer
            to the following links:
            https://github.com/open-mmlab/mmdetection3d/issues/894
            https://github.com/open-mmlab/mmdetection3d/pull/904
            it is an experimental feature and we will appreciate it if
            you could share with us the failing cases.
    """

    def __init__(self,
                 point_cloud_range: List,
                 max_num_points: int,
                 voxel_size: List = [],
                 grid_shape: List[int] = [],
                 max_voxels: Union[tuple, int] = 20000,
                 deterministic: bool = True):
        super().__init__()
        if voxel_size and grid_shape:
            raise ValueError('voxel_size is mutually exclusive grid_shape')
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        if voxel_size:
            self.voxel_size = voxel_size
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
            grid_shape = (point_cloud_range[3:] -
                          point_cloud_range[:3]) / voxel_size
            grid_shape = torch.round(grid_shape).long().tolist()
            self.grid_shape = grid_shape
        elif grid_shape:
            grid_shape = torch.tensor(grid_shape, dtype=torch.float32)
            #######################   we modify this code ###############################################
            # voxel_size = (point_cloud_range[3:] - point_cloud_range[:3]) / (grid_shape - 1)
            voxel_size  = (point_cloud_range[3:] - point_cloud_range[:3]) / grid_shape
            voxel_size = voxel_size.tolist()
            self.voxel_size = voxel_size
            self.grid_shape = grid_shape
        else:
            raise ValueError('must assign a value to voxel_size or grid_shape')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return voxelization(input, self.voxel_size, self.point_cloud_range,
                            self.max_num_points, max_voxels,
                            self.deterministic)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'voxel_size=' + str(self.voxel_size)
        s += ', grid_shape=' + str(self.grid_shape)
        s += ', point_cloud_range=' + str(self.point_cloud_range)
        s += ', max_num_points=' + str(self.max_num_points)
        s += ', max_voxels=' + str(self.max_voxels)
        s += ', deterministic=' + str(self.deterministic)
        s += ')'
        return s


