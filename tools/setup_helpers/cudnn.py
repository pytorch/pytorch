import os
import sys
import glob
from itertools import chain
import re

from .env import check_env_flag
from .cuda import WITH_CUDA, CUDA_HOME

try:  # python 3+
    from itertools import zip_longest as zip_longest
except:  # python 2.7+
    from itertools import izip_longest as zip_longest


def gather_paths(env_vars):
    return list(chain(*(os.getenv(v, '').split(':') for v in env_vars)))


def find_cudnn_version(cudnn_lib_dir):
    candidate_names = list(glob.glob(os.path.join(cudnn_lib_dir, 'libcudnn*')))
    candidate_names = [os.path.basename(c) for c in candidate_names]

    # suppose lib file is is libcudnn.so.MAJOR.MINOR.PATCH, all numbers
    version_regex = re.compile('.so.(\d+\.\d+\.\d+)$')
    candidates = [c.group(1) for c in map(version_regex.search, candidate_names) if c]

    # libcudnn.so.MAJOR.MINOR
    version_regex = re.compile('.so.(\d+\.\d+)$')
    candidates += [c.group(1) for c in map(version_regex.search, candidate_names) if c]

    # libcudnn.so.MAJOR
    version_regex = re.compile('.so.(\d+)$')
    candidates += [c.group(1) for c in map(version_regex.search, candidate_names) if c]

    if len(candidates) == 0:
        return 'unknown'

    # Each candidate represented as list, eg 6.0.21 -> [6, 0, 21]
    candidates = [[int(x) for x in c.split('.')] for c in candidates]

    # From candidates, take the most recent, then most detailed version string
    def version_cmp(a, b):
        for (x, y) in zip_longest(a, b, fillvalue=-1):
            diff = x - y
            if diff != 0:
                return diff
        return 0

    version = candidates[0]
    for candidate in candidates:
        result = version_cmp(version, candidate)
        if result < 0:  # version < candidate
            version = candidate
    return '.'.join([str(v) for v in version])


def check_cudnn_version(cudnn_version_string):
    if cudnn_version_string is 'unknown':
        return  # Assume version is OK and let compilation continue

    cudnn_min_version = 6
    cudnn_version = int(cudnn_version_string.split('.')[0])
    if cudnn_version < cudnn_min_version:
        raise RuntimeError(
            'CuDNN v%s found, but need at least CuDNN v%s. '
            'You can get the latest version of CuDNN from '
            'https://developer.nvidia.com/cudnn or disable '
            'CuDNN with NO_CUDNN=1' %
            (cudnn_version_string, cudnn_min_version))


is_conda = 'conda' in sys.version or 'Continuum' in sys.version
conda_dir = os.path.join(os.path.dirname(sys.executable), '..')

WITH_CUDNN = False
CUDNN_LIB_DIR = None
CUDNN_INCLUDE_DIR = None
CUDNN_VERSION = None
if WITH_CUDA and not check_env_flag('NO_CUDNN'):
    lib_paths = list(filter(bool, [
        os.getenv('CUDNN_LIB_DIR'),
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
    if is_conda:
        lib_paths.append(os.path.join(conda_dir, 'lib'))
        include_paths.append(os.path.join(conda_dir, 'include'))
    for path in lib_paths:
        if path is None or not os.path.exists(path):
            continue
        if glob.glob(os.path.join(path, 'libcudnn*')):
            CUDNN_LIB_DIR = path
            break
    for path in include_paths:
        if path is None or not os.path.exists(path):
            continue
        if os.path.exists((os.path.join(path, 'cudnn.h'))):
            CUDNN_INCLUDE_DIR = path
            break
    if not CUDNN_LIB_DIR or not CUDNN_INCLUDE_DIR:
        CUDNN_LIB_DIR = CUDNN_INCLUDE_DIR = None
    else:
        CUDNN_VERSION = find_cudnn_version(CUDNN_LIB_DIR)
        check_cudnn_version(CUDNN_VERSION)
        WITH_CUDNN = True
