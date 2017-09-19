import os
import platform
import ctypes.util
from subprocess import Popen, PIPE

from .env import check_env_flag

LINUX_HOME = '/usr/local/cuda'
WINDOWS_HOME = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0'


def find_nvcc():
    proc = Popen(['which', 'nvcc'], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    out = out.decode().strip()
    if len(out) > 0:
        return os.path.dirname(out)
    else:
        return None


if check_env_flag('NO_CUDA'):
    WITH_CUDA = False
    CUDA_HOME = None
else:
    # We use nvcc path on Linux and cudart path on macOS
    osname = platform.system()
    if osname != 'Windows':
        CUDA_HOME = os.getenv('CUDA_HOME', LINUX_HOME)
    else:
        CUDA_HOME = os.getenv('CUDA_PATH', WINDOWS_HOME).replace('\\', '/')
    if not os.path.exists(CUDA_HOME):
        if osname == 'Linux':
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
    WITH_CUDA = CUDA_HOME is not None
