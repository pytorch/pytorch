import ctypes.util
import os
import platform

from .env import check_env_flag

if check_env_flag('NO_CUDA'):
    WITH_CUDA = False
    CUDA_HOME = None
else:
    if platform.system() == 'Windows':
        CUDA_HOME = os.getenv('CUDA_PATH', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0').replace('\\','/')

    else:
        CUDA_HOME = os.getenv('CUDA_HOME', '/usr/local/cuda')
    if not os.path.exists(CUDA_HOME):
        cudart_path = ctypes.util.find_library('cudart')
        if cudart_path is not None:
            CUDA_HOME = os.path.dirname(cudart_path)
        else:
            CUDA_HOME = None
    WITH_CUDA = CUDA_HOME is not None
