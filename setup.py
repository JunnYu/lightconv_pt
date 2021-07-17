#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="lightconv_layer",
    version="0.0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            name="lightconv_cuda",
            sources=[
                "csrc/lightconv_cuda.cpp",
                "csrc/lightconv_cuda_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
