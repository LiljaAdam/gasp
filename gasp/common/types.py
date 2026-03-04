from typing import Tuple, Union

from zod import ZodFrame, ZodSequence

PointCloudRange = Tuple[float, float, float, float, float, float]  # [x_min, y_min, zmin, x_max, y_max, zmax]
VoxelSize = Tuple[float, float, float]  # [x, y, z]
Zample = Union[ZodFrame, ZodSequence]
