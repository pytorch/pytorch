import os
import glob
import re
import platform
import ctypes.util
from subprocess import Popen, PIPE

from .env import check_env_flag


def find_nvcc():
    proc = Popen(['which', 'nvcc'], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    out = out.decode().strip()
    if len(out) > 0:
        return os.path.dirname(out)
    else:
        return None


def find_cuda_version(cuda_home):
    if cuda_home is None:
        return None
    # get CUDA lib folder
    cuda_lib_dirs = ['lib64', 'lib']
    for lib_dir in cuda_lib_dirs:
        cuda_lib_path = os.path.join(cuda_home, lib_dir)
        if os.path.exists(cuda_lib_path):
            break
    # get a list of candidates for the version number
    # which are files containing cudart
    candidate_names = list(glob.glob(os.path.join(cuda_lib_path, '*cudart*')))
    candidate_names = [os.path.basename(c) for c in candidate_names]
    # suppose version is MAJOR.MINOR.PATCH, all numbers
    version_regex = re.compile('[0-9]+\.[0-9]+\.[0-9]+')
    candidates = [c.group() for c in map(version_regex.search, candidate_names) if c]
    if len(candidates) > 0:
        # normally only one will be retrieved, take the first result
        return candidates[0]
    # if no candidates were found, try MAJOR.MINOR
    version_regex = re.compile('[0-9]+\.[0-9]+')
    candidates = [c.group() for c in map(version_regex.search, candidate_names) if c]
    if len(candidates) > 0:
        return candidates[0]


if check_env_flag('NO_CUDA'):
    WITH_CUDA = False
    CUDA_HOME = None
    CUDA_VERSION = None
else:
    CUDA_HOME = os.getenv('CUDA_HOME', '/usr/local/cuda')
    if not os.path.exists(CUDA_HOME):
        # We use nvcc path on Linux and cudart path on macOS
        osname = platform.system()
        if osname == 'Linux':
            cuda_path = find_nvcc()
        else:
            cudart_path = ctypes.util.find_library('cudart')
            if cudart_path is not None:
                cuda_path = os.path.dirname(cudart_path)
            else:
                cuda_path = None
        if cuda_path is not None:
            CUDA_HOME = os.path.dirname(cuda_path)
        else:
            CUDA_HOME = None
    CUDA_VERSION = find_cuda_version(CUDA_HOME)
    WITH_CUDA = CUDA_HOME is not None
