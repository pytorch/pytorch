import os
import sys
import multiprocessing
from .env import (IS_ARM, IS_DARWIN, IS_LINUX, IS_PPC, IS_WINDOWS,
                  check_env_flag, check_negative_env_flag,
                  hotpatch_build_env_vars)


hotpatch_build_env_vars()

from .build import (BLAS, BUILD_BINARY, BUILD_CAFFE2_OPS, BUILD_TEST,
                    USE_FBGEMM, USE_FFMPEG, USE_LEVELDB, USE_LMDB, USE_OPENCV,
                    USE_REDIS, USE_TENSORRT, USE_ZSTD, CUDA_NVCC_EXECUTABLE)
from .cuda import CUDA_HOME, CUDA_VERSION, USE_CUDA
from .cudnn import CUDNN_INCLUDE_DIR, CUDNN_LIB_DIR, CUDNN_LIBRARY, USE_CUDNN
from .dist_check import USE_DISTRIBUTED, USE_GLOO_IBVERBS
from .miopen import (MIOPEN_INCLUDE_DIR, MIOPEN_LIB_DIR, MIOPEN_LIBRARY,
                     USE_MIOPEN)
from .nccl import (NCCL_INCLUDE_DIR, NCCL_LIB_DIR, NCCL_ROOT_DIR,
                   NCCL_SYSTEM_LIB, USE_NCCL, USE_SYSTEM_NCCL)
from .nnpack import USE_NNPACK
from .nvtoolext import NVTOOLEXT_HOME
from .qnnpack import USE_QNNPACK
from .rocm import ROCM_HOME, ROCM_VERSION, USE_ROCM


DEBUG = check_env_flag('DEBUG')
REL_WITH_DEB_INFO = check_env_flag('REL_WITH_DEB_INFO')

BUILD_PYTORCH = check_env_flag('BUILD_PYTORCH')
# ppc64le and aarch64 do not support MKLDNN
if IS_PPC or IS_ARM:
    USE_MKLDNN = check_env_flag('USE_MKLDNN', 'OFF')
else:
    USE_MKLDNN = check_env_flag('USE_MKLDNN', 'ON')

USE_CUDA_STATIC_LINK = check_env_flag('USE_CUDA_STATIC_LINK')
RERUN_CMAKE = False

NUM_JOBS = multiprocessing.cpu_count()
max_jobs = os.getenv("MAX_JOBS")
if max_jobs is not None:
    NUM_JOBS = min(NUM_JOBS, int(max_jobs))

ONNX_NAMESPACE = os.getenv("ONNX_NAMESPACE")
if not ONNX_NAMESPACE:
    ONNX_NAMESPACE = "onnx_torch"

# Ninja
try:
    import ninja
    USE_NINJA = True
except ImportError:
    USE_NINJA = False

try:
    import numpy as np
    NUMPY_INCLUDE_DIR = np.get_include()
    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False


def get_common_env_with_flags():
    extra_flags = []
    my_env = os.environ.copy()
    my_env["PYTORCH_PYTHON"] = sys.executable
    my_env["ONNX_NAMESPACE"] = ONNX_NAMESPACE
    if BLAS:
        my_env["BLAS"] = BLAS
    if USE_SYSTEM_NCCL:
        my_env["NCCL_ROOT_DIR"] = NCCL_ROOT_DIR
        my_env["NCCL_INCLUDE_DIR"] = NCCL_INCLUDE_DIR
        my_env["NCCL_SYSTEM_LIB"] = NCCL_SYSTEM_LIB
    if USE_CUDA:
        my_env["CUDA_BIN_PATH"] = CUDA_HOME
        extra_flags += ['--use-cuda']
        if IS_WINDOWS:
            my_env["NVTOOLEXT_HOME"] = NVTOOLEXT_HOME
        if CUDA_NVCC_EXECUTABLE:
            my_env["CUDA_NVCC_EXECUTABLE"] = CUDA_NVCC_EXECUTABLE
    if USE_CUDA_STATIC_LINK:
        extra_flags += ['--cuda-static-link']
    if USE_FBGEMM:
        extra_flags += ['--use-fbgemm']
    if USE_ROCM:
        extra_flags += ['--use-rocm']
    if USE_NNPACK:
        extra_flags += ['--use-nnpack']
    if USE_CUDNN:
        my_env["CUDNN_LIB_DIR"] = CUDNN_LIB_DIR
        my_env["CUDNN_LIBRARY"] = CUDNN_LIBRARY
        my_env["CUDNN_INCLUDE_DIR"] = CUDNN_INCLUDE_DIR
    if USE_MIOPEN:
        my_env["MIOPEN_LIB_DIR"] = MIOPEN_LIB_DIR
        my_env["MIOPEN_LIBRARY"] = MIOPEN_LIBRARY
        my_env["MIOPEN_INCLUDE_DIR"] = MIOPEN_INCLUDE_DIR
    if USE_MKLDNN:
        extra_flags += ['--use-mkldnn']
    if USE_QNNPACK:
        extra_flags += ['--use-qnnpack']
    if USE_GLOO_IBVERBS:
        extra_flags += ['--use-gloo-ibverbs']
    if not RERUN_CMAKE:
        extra_flags += ['--dont-rerun-cmake']

    my_env["BUILD_TORCH"] = "ON"
    my_env["BUILD_TEST"] = "ON" if BUILD_TEST else "OFF"
    my_env["BUILD_CAFFE2_OPS"] = "ON" if BUILD_CAFFE2_OPS else "OFF"
    my_env["INSTALL_TEST"] = "ON" if BUILD_TEST else "OFF"
    my_env["USE_LEVELDB"] = "ON" if USE_LEVELDB else "OFF"
    my_env["USE_LMDB"] = "ON" if USE_LMDB else "OFF"
    my_env["USE_OPENCV"] = "ON" if USE_OPENCV else "OFF"
    my_env["USE_TENSORRT"] = "ON" if USE_TENSORRT else "OFF"
    my_env["USE_FFMPEG"] = "ON" if USE_FFMPEG else "OFF"
    my_env["USE_DISTRIBUTED"] = "ON" if USE_DISTRIBUTED else "OFF"
    my_env["USE_SYSTEM_NCCL"] = "ON" if USE_SYSTEM_NCCL else "OFF"

    return my_env, extra_flags


def get_libtorch_env_with_flags():
    my_env, extra_flags = get_common_env_with_flags()

    return my_env, extra_flags


def get_pytorch_env_with_flags():
    my_env, extra_flags = get_common_env_with_flags()
    my_env["BUILD_BINARY"] = "ON" if BUILD_BINARY else "OFF"
    my_env["BUILD_PYTHON"] = "ON"
    my_env["NUM_JOBS"] = str(NUM_JOBS)
    if not IS_WINDOWS:
        if USE_NINJA:
            my_env["CMAKE_GENERATOR"] = '-GNinja'
            my_env["CMAKE_INSTALL"] = 'ninja install'
        else:
            my_env['CMAKE_GENERATOR'] = ''
            my_env['CMAKE_INSTALL'] = 'make install'
    if USE_NUMPY:
        my_env["NUMPY_INCLUDE_DIR"] = NUMPY_INCLUDE_DIR

    return my_env, extra_flags
