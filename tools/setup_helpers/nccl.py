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

IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')

WITH_NCCL = WITH_CUDA and not IS_DARWIN and not IS_WINDOWS
WITH_SYSTEM_NCCL = False
NCCL_LIB_DIR = None
NCCL_SYSTEM_LIB = None
NCCL_INCLUDE_DIR = None
NCCL_ROOT_DIR = None
if WITH_CUDA and not check_env_flag('NO_SYSTEM_NCCL'):
    ENV_ROOT = os.getenv('NCCL_ROOT_DIR', None)
    LIB_DIR = os.getenv('NCCL_LIB_DIR', None)
    INCLUDE_DIR = os.getenv('NCCL_INCLUDE_DIR', None)

    lib_paths = list(filter(bool, [
        LIB_DIR,
        ENV_ROOT,
        os.path.join(ENV_ROOT, 'lib') if ENV_ROOT is not None else None,
        os.path.join(ENV_ROOT, 'lib', 'x86_64-linux-gnu') if ENV_ROOT is not None else None,
        os.path.join(ENV_ROOT, 'lib64') if ENV_ROOT is not None else None,
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
    include_paths = list(filter(bool, [
        INCLUDE_DIR,
        ENV_ROOT,
        os.path.join(ENV_ROOT, 'include') if ENV_ROOT is not None else None,
        '/usr/include'
    ]))

    if is_conda:
        lib_paths.append(os.path.join(conda_dir, 'lib'))
    for path in lib_paths:
        path = os.path.expanduser(path)
        if path is None or not os.path.exists(path):
            continue
        if glob.glob(os.path.join(path, 'libnccl*')):
            NCCL_LIB_DIR = path
            # try to find an exact versioned .so/.dylib, rather than libnccl.so
            preferred_path = glob.glob(os.path.join(path, 'libnccl*[0-9]*'))
            if len(preferred_path) == 0:
                NCCL_SYSTEM_LIB = glob.glob(os.path.join(path, 'libnccl*'))[0]
            else:
                NCCL_SYSTEM_LIB = os.path.realpath(preferred_path[0])
            break
    for path in include_paths:
        path = os.path.expanduser(path)
        if path is None or not os.path.exists(path):
            continue
        if glob.glob(os.path.join(path, 'nccl.h')):
            NCCL_INCLUDE_DIR = path
            break
    if NCCL_LIB_DIR is not None and NCCL_INCLUDE_DIR is not None:
        WITH_SYSTEM_NCCL = True
        NCCL_ROOT_DIR = os.path.commonprefix((NCCL_LIB_DIR, NCCL_INCLUDE_DIR))
