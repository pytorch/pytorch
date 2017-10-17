import os
import platform

from .env import check_env_flag

WITH_NNPACK = False
NNPACK_LIB_DIR = None
NNPACK_INCLUDE_DIRS = None

if not check_env_flag('NO_NNPACK') and 'Windows' not in platform.system():
    # assume we have a flag set that determines the path
    nnpack_dir = os.getenv('NNPACK_DIR')
    if nnpack_dir is not None:
        NNPACK_LIB_DIR = os.path.join(nnpack_dir, 'lib')
        NNPACK_INCLUDE_DIRS = [os.path.join(nnpack_dir, 'include'),
                               os.path.join(nnpack_dir, 'deps', 'pthreadpool', 'include')]

        if os.path.exists(NNPACK_LIB_DIR) and all(os.path.exists(d) for d in NNPACK_INCLUDE_DIRS):
            WITH_NNPACK = True
        else:
            NNPACK_LIB_DIR = NNPACK_INCLUDE_DIRS = None
