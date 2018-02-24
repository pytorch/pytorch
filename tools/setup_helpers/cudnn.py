import os
import glob

from .env import IS_WINDOWS, IS_CONDA, CONDA_DIR, check_env_flag, gather_paths
from .cuda import WITH_CUDA, CUDA_HOME


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
    for path in include_paths:
        if path is None or not os.path.exists(path):
            continue
        include_file_path = os.path.join(path, 'cudnn.h')
        if os.path.exists(include_file_path):
            CUDNN_INCLUDE_DIR = path
            CUDNN_INCLUDE_VERSION = -1
            with open(include_file_path) as f:
                for line in f:
                    if "#define CUDNN_MAJOR" in line:
                        CUDNN_INCLUDE_VERSION = int(line.split()[-1])
                        break
            if CUDNN_INCLUDE_VERSION == -1:
                raise AssertionError("Could not find #define CUDNN_MAJOR in " + include_file_path)
            break

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
            libraries = sorted(glob.glob(os.path.join(path, 'libcudnn*' + str(CUDNN_INCLUDE_VERSION) + "*")))
            if libraries:
                CUDNN_LIBRARY = libraries[0]
                CUDNN_LIB_DIR = path
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
