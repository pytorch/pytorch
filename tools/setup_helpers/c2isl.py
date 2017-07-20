import os
import glob
from itertools import chain

from .env import check_env_flag
from .cuda import WITH_CUDA, CUDA_HOME

WITH_C2ISL = False
C2ISL_LIB_DIR = None
C2ISL_INCLUDE_DIR = None

# TODO: Factor out common code from this and cudnn.py

# C2ISL is CUDA-only at the moment
if WITH_CUDA and not check_env_flag('NO_C2ISL'):
    lib_paths = []
    include_paths = []

    env_lib_dir = os.getenv('C2ISL_LIB_DIR')
    if env_lib_dir:
        lib_paths.append(env_lib_dir)

    env_include_dir = os.getenv('C2ISL_INCLUDE_DIR')
    if env_include_dir:
        include_paths.append(env_include_dir)

    conda_prefix = os.getenv('CONDA_PREFIX')
    if conda_prefix:
        lib_paths.append(os.path.join(conda_prefix, 'lib'))
        lib_paths.append(os.path.join(conda_prefix, 'lib64'))
        include_paths.append(os.path.join(conda_prefix, 'include'))

    for path in lib_paths:
        if not os.path.exists(path):
            continue
        if glob.glob(os.path.join(path, 'libc2isl_core*')):
            C2ISL_LIB_DIR = path
            break
    for path in include_paths:
        if not os.path.exists(path):
            continue
        if os.path.exists(os.path.join(path, 'c2isl')):
            C2ISL_INCLUDE_DIR = path
            break
    if not C2ISL_LIB_DIR or not C2ISL_INCLUDE_DIR:
        C2ISL_LIB_DIR = C2ISL_INCLUDE_DIR = None
    else:
        WITH_C2ISL = True
