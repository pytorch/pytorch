import os
import glob
import warnings
from itertools import chain

from .env import IS_WINDOWS, IS_DARWIN, IS_CONDA, CONDA_DIR, check_negative_env_flag, \
    gather_paths

from .cuda import USE_CUDA, CUDA_HOME


USE_NCCL = USE_CUDA and not IS_DARWIN and not IS_WINDOWS
USE_SYSTEM_NCCL = False
NCCL_LIB_DIR = None
NCCL_SYSTEM_LIB = None
NCCL_INCLUDE_DIR = None
NCCL_ROOT_DIR = None
USE_STATIC_NCCL = os.getenv("USE_STATIC_NCCL")
LIBNCCL_PREFIX = "libnccl"
if USE_STATIC_NCCL is not None:
    LIBNCCL_PREFIX = "libnccl_static"

if USE_CUDA and not check_negative_env_flag('USE_SYSTEM_NCCL'):
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
        '/usr/local/lib',
        '/usr/lib/x86_64-linux-gnu/',
        '/usr/lib/powerpc64le-linux-gnu/',
        '/usr/lib/aarch64-linux-gnu/',
        '/usr/lib',
    ] + gather_paths([
        'LIBRARY_PATH',
    ]) + gather_paths([
        'LD_LIBRARY_PATH',
    ])))
    include_paths = list(filter(bool, [
        INCLUDE_DIR,
        ENV_ROOT,
        os.path.join(ENV_ROOT, 'include') if ENV_ROOT is not None else None,
        os.path.join(CUDA_HOME, 'include'),
        '/usr/local/include',
        '/usr/include',
    ]))

    if IS_CONDA:
        lib_paths.append(os.path.join(CONDA_DIR, 'lib'))
    for path in lib_paths:
        path = os.path.expanduser(path)
        if path is None or not os.path.exists(path):
            continue
        if glob.glob(os.path.join(path, LIBNCCL_PREFIX + '*')):
            NCCL_LIB_DIR = path
            # try to find an exact versioned .so/.dylib, rather than libnccl.so
            preferred_path = glob.glob(os.path.join(path, LIBNCCL_PREFIX + '*[0-9]*'))
            if len(preferred_path) == 0:
                NCCL_SYSTEM_LIB = glob.glob(os.path.join(path, LIBNCCL_PREFIX + '*'))[0]
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
        USE_SYSTEM_NCCL = True
        NCCL_ROOT_DIR = os.path.commonprefix((NCCL_LIB_DIR, NCCL_INCLUDE_DIR))
