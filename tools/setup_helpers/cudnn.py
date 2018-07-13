import os
import glob

from .env import IS_WINDOWS, IS_CONDA, CONDA_DIR, check_negative_env_flag, gather_paths, lib_paths_from_base
from .cuda import USE_CUDA, CUDA_HOME


USE_CUDNN = False
CUDNN_LIB_DIR = None
CUDNN_INCLUDE_DIR = None
CUDNN_LIBRARY = None
WITH_STATIC_CUDNN = os.getenv("USE_STATIC_CUDNN")

if USE_CUDA and not check_negative_env_flag('USE_CUDNN'):
    lib_paths = list(filter(bool, [
        os.getenv('CUDNN_LIB_DIR')
    ] + lib_paths_from_base(CUDA_HOME) + [
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
    # Add CUDA related dirs to candidate list
    if IS_CONDA:
        lib_paths.append(os.path.join(CONDA_DIR, 'lib'))
        include_paths.append(os.path.join(CONDA_DIR, 'include'))
    for path in include_paths:
        if path is None or not os.path.exists(path):
            continue
        include_file_path = os.path.join(path, 'cudnn.h')
        CUDNN_INCLUDE_VERSION = None
        if os.path.exists(include_file_path):
            CUDNN_INCLUDE_DIR = path
            with open(include_file_path) as f:
                for line in f:
                    if "#define CUDNN_MAJOR" in line:
                        CUDNN_INCLUDE_VERSION = int(line.split()[-1])
                        break
            if CUDNN_INCLUDE_VERSION is None:
                raise AssertionError("Could not find #define CUDNN_MAJOR in " + include_file_path)
            break

    if CUDNN_INCLUDE_VERSION is None:
        pass

    # Check for standalone cuDNN libraries
    if CUDNN_INCLUDE_DIR is not None:
        cudnn_path = os.path.join(os.path.dirname(CUDNN_INCLUDE_DIR))
        cudnn_lib_paths = lib_paths_from_base(cudnn_path)
        lib_paths.extend(cudnn_lib_paths)

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
            if WITH_STATIC_CUDNN is not None:
                search_name = 'libcudnn_static.a'
            else:
                search_name = 'libcudnn*' + str(CUDNN_INCLUDE_VERSION) + "*"
            libraries = sorted(glob.glob(os.path.join(path, search_name)))
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
        USE_CUDNN = True
