import os

from .env import check_env_flag

CUDA_HOME = os.getenv('CUDA_HOME', '/usr/local/cuda')
WITH_CUDA = not check_env_flag('NO_CUDA') and os.path.exists(CUDA_HOME)
if not WITH_CUDA:
    CUDA_HOME = None
