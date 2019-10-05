import os
import glob
import ctypes.util

from . import which
from .env import IS_WINDOWS, IS_LINUX, IS_DARWIN, check_negative_env_flag

LINUX_HOME = '/usr/local/cuda'
WINDOWS_HOME = glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')


def find_nvcc():
    nvcc = which('nvcc')
    if nvcc is not None:
        return os.path.dirname(nvcc)
    else:
        return None


if check_negative_env_flag('USE_CUDA'):
    USE_CUDA = False
    CUDA_HOME = None
else:
    if IS_LINUX or IS_DARWIN:
        CUDA_HOME = os.getenv('CUDA_HOME', LINUX_HOME)
    else:
        CUDA_HOME = os.getenv('CUDA_PATH', '').replace('\\', '/')
        if CUDA_HOME == '' and len(WINDOWS_HOME) > 0:
            CUDA_HOME = WINDOWS_HOME[0].replace('\\', '/')
    if not os.path.exists(CUDA_HOME):
        # We use nvcc path on Linux and cudart path on macOS
        if IS_LINUX or IS_WINDOWS:
            cuda_path = find_nvcc()
        else:
            cudart_path = ctypes.util.find_library('cudart')
            if cudart_path is not None:
                cuda_path = os.path.dirname(cudart_path)
            else:
                cuda_path = None
        if cuda_path is not None:
            CUDA_HOME = os.path.dirname(cuda_path)
        else:
            CUDA_HOME = None
    USE_CUDA = CUDA_HOME is not None
