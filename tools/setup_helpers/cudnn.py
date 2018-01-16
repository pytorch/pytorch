import os
import platform
import sys
import glob
from itertools import chain

from .env import check_env_flag
from .cuda import WITH_CUDA, CUDA_HOME


def gather_paths(env_vars):
    return list(chain(*(os.getenv(v, '').split(':') for v in env_vars)))


IS_WINDOWS = (platform.system() == 'Windows')
IS_CONDA = 'conda' in sys.version or 'Continuum' in sys.version
CONDA_DIR = os.path.join(os.path.dirname(sys.executable), '..')

WITH_CUDNN = False
CUDNN_LIB_DIR = None
CUDNN_INCLUDE_DIR = None
CUDNN_LIBRARY = None
if WITH_CUDA and not check_env_flag('NO_CUDNN'):
    lib_paths = list(filter(bool, [
        os.getenv('CUDNN_LIB_DIR'),
        os.path.join(CUDA_HOME, 'lib/x64'),
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
        os.getenv('CUDNN_INCLUDE_DIR'),
        os.path.join(CUDA_HOME, 'include'),
        '/usr/include/',
    ] + gather_paths([
        'CPATH',
        'C_INCLUDE_PATH',
        'CPLUS_INCLUDE_PATH',
    ])))
    if IS_CONDA:
        lib_paths.append(os.path.join(CONDA_DIR, 'lib'))
        include_paths.append(os.path.join(CONDA_DIR, 'include'))
    for path in lib_paths:
        if path is None or not os.path.exists(path):
            continue
        if IS_WINDOWS:
            library = os.path.join(path, 'cudnn.lib')
            if os.path.exists(library):
                CUDNN_LIBRARY = library
                CUDNN_LIB_DIR = path
                break
        else:
            libraries = sorted(glob.glob(os.path.join(path, 'libcudnn*')))
            if libraries:
                CUDNN_LIBRARY = libraries[0]
                CUDNN_LIB_DIR = path
                break
    for path in include_paths:
        if path is None or not os.path.exists(path):
            continue
        if os.path.exists((os.path.join(path, 'cudnn.h'))):
            CUDNN_INCLUDE_DIR = path
            break

    # Specifying the library directly will overwrite the lib directory
    library = os.getenv('CUDNN_LIBRARY')
    if library is not None and os.path.exists(library):
        CUDNN_LIBRARY = library
        CUDNN_LIB_DIR = os.path.dirname(CUDNN_LIBRARY)

    if not all([CUDNN_LIBRARY, CUDNN_LIB_DIR, CUDNN_INCLUDE_DIR]):
        CUDNN_LIBRARY = CUDNN_LIB_DIR = CUDNN_INCLUDE_DIR = None
    else:
        real_cudnn_library = os.path.realpath(CUDNN_LIBRARY)
        real_cudnn_lib_dir = os.path.realpath(CUDNN_LIB_DIR)
        assert os.path.dirname(real_cudnn_library) == real_cudnn_lib_dir, (
            'cudnn library and lib_dir must agree')
        WITH_CUDNN = True
