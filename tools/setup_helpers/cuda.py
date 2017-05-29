import os
from subprocess import Popen, PIPE

from .env import check_env_flag


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
    CUDA_HOME = os.getenv('CUDA_HOME', '/usr/local/cuda')
    if not os.path.exists(CUDA_HOME):
        nvcc_path = find_nvcc()
        if nvcc_path is not None:
            CUDA_HOME = os.path.dirname(nvcc_path)
        else:
            CUDA_HOME = None
    WITH_CUDA = CUDA_HOME is not None
