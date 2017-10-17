import os
import platform

from .env import check_env_flag

WITH_NNPACK = False

NNPACK_IS_USER_DEFINED = False
NNPACK_LIB_DIR = []
NNPACK_INCLUDE_DIRS = []

NNPACK_IS_CONDA_DEFINED = False
NNPACK_LIB_PATHS = []
NNPACK_HEADER_PATHS = []

if not check_env_flag('NO_NNPACK') and 'Windows' not in platform.system():
    # assume we have a flag set that determines the path
    nnpack_dir = os.getenv('NNPACK_DIR')
    if nnpack_dir is not None:
        NNPACK_LIB_DIR = os.path.join(nnpack_dir, 'lib')
        NNPACK_INCLUDE_DIRS = [os.path.join(nnpack_dir, 'include'),
                               os.path.join(nnpack_dir, 'deps', 'pthreadpool', 'include')]

        if os.path.exists(NNPACK_LIB_DIR) and all(os.path.exists(d) for d in NNPACK_INCLUDE_DIRS):
            WITH_NNPACK = NNPACK_IS_USER_DEFINED = True
        else:
            NNPACK_LIB_DIR = NNPACK_INCLUDE_DIRS = None
    else:
        # Look in the Conda directory to see if we've Conda installed it.
        prefix = os.getenv('CONDA_PREFIX')
        if prefix is not None:
            lib = os.path.join(prefix, 'lib')
            include = os.path.join(prefix, 'include')

            libs = [
                os.path.join(lib, 'libnnpack.a'),
                os.path.join(lib, 'libpthreadpool.a')
            ]
            headers = [
                os.path.join(include, 'nnpack.h'),
                os.path.join(include, 'pthreadpool.h')
            ]

            if all([os.path.exists(f) for f in libs + headers]):
                WITH_NNPACK = NNPACK_IS_CONDA_DEFINED = True
                NNPACK_LIB_PATHS = libs
                NNPACK_HEADER_PATHS = headers
