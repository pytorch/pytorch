import os
import sys
import glob
import platform
import warnings
from itertools import chain

from .env import check_env_flag
from .cuda import WITH_CUDA, CUDA_HOME


def gather_paths(env_vars):
    return list(chain(*(os.getenv(v, '').split(':') for v in env_vars)))

is_conda = 'conda' in sys.version or 'Continuum' in sys.version
conda_dir = os.path.join(os.path.dirname(sys.executable), '..')

WITH_NCCL = WITH_CUDA and platform.system() != 'Darwin'
WITH_SYSTEM_NCCL = False
NCCL_LIB_DIR = None
NCCL_INCLUDE_DIR = None
NCCL_ROOT_DIR = None
if WITH_CUDA and not check_env_flag('NO_SYSTEM_NCCL'):
    ENV_ROOT = os.getenv('NCCL_ROOT_DIR', None)
    # NCCL_ROOT_DIR takes precedence over NCCL_LIB_DIR
    lib_paths = list(filter(bool, [
        os.path.join(ENV_ROOT, 'lib') if ENV_ROOT is not None else None,
        os.path.join(ENV_ROOT, 'lib64') if ENV_ROOT is not None else None,
        os.getenv('NCCL_LIB_DIR'),
        os.path.join(CUDA_HOME, 'lib'),
        os.path.join(CUDA_HOME, 'lib64'),
        '/usr/lib/x86_64-linux-gnu/',
        '/usr/lib/powerpc64le-linux-gnu/',
        '/usr/lib/aarch64-linux-gnu/',
    ] + gather_paths([
        'LIBRARY_PATH',
    ]) + gather_paths([
        'LD_LIBRARY_PATH',
    ])))

    if os.getenv('NCCL_INCLUDE_DIR') is not None:
        warnings.warn("Ignoring environment variable NCCL_INCLUDE_DIR because "
                      "NCCL_INCLUDE_DIR is implicitly assumed as "
                      "$NCCL_ROOT_DIR/include or $NCCL_LIB_DIR/../include")
    if is_conda:
        lib_paths.append(os.path.join(conda_dir, 'lib'))
    for path in lib_paths:
        if path is None or not os.path.exists(path):
            continue
        if glob.glob(os.path.join(path, 'libnccl*')):
            if os.path.exists((os.path.join(path, '../include/nccl.h'))):
                NCCL_LIB_DIR = path
                NCCL_INCLUDE_DIR = path
                break
    if NCCL_LIB_DIR is not None:
        WITH_SYSTEM_NCCL = True
        NCCL_ROOT_DIR = os.path.abspath(os.path.join(NCCL_LIB_DIR, "../"))
