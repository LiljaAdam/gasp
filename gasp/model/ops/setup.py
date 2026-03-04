import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension

requirements = ["torch"]

this_dir = os.path.dirname(os.path.abspath(__file__))

# Single setup() so "python setup.py build install" builds and installs all extensions.
# (Two setup() calls would cause only the first to be used for the install command.)
setup(
    name="gasp_ops",
    ext_modules=[
        CUDAExtension(
            name="fast_interp_cuda",
            sources=[os.path.join(this_dir, "interpolate/interpolate_kernel.cu")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
        CppExtension(
            name="fast_interp_cpu",
            sources=[os.path.join(this_dir, "interpolate/interpolate_cpu.cpp")],
            extra_compile_args={
                "cxx": ["-O3", "-fopenmp"],
                "unix": ["-fopenmp"],
            },
            extra_link_args=["-fopenmp"],
        ),
        CUDAExtension(
            name="voxel_op",
            sources=[
                os.path.join(this_dir, "voxelization/voxelization.cpp"),
                os.path.join(this_dir, "voxelization/voxelization_cpu.cpp"),
                os.path.join(this_dir, "voxelization/voxelization_cuda.cu"),
            ],
            define_macros=[("WITH_CUDA", None)],
        ),
    ],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
