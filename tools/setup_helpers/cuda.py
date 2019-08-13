import os
import glob
import re
import ctypes.util
from subprocess import Popen, PIPE

from . import escape_path, which
from .env import IS_WINDOWS, IS_LINUX, IS_DARWIN, check_env_flag, check_negative_env_flag

LINUX_HOME = '/usr/local/cuda'
WINDOWS_HOME = glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')


def find_nvcc():
    nvcc = which('nvcc')
    if nvcc is not None:
        nvcc = escape_path(nvcc)
        return os.path.dirname(nvcc)
    else:
        return None


def find_cuda_version(cuda_home):
    if cuda_home is None:
        return None
    if IS_WINDOWS:
        candidate_names = [os.path.basename(cuda_home)]
    else:
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
        # if we didn't find any cudart, ask nvcc
        if len(candidate_names) == 0:
            proc = Popen(['nvcc', '--version'], stdout=PIPE, stderr=PIPE)
            out, err = proc.communicate()
            candidate_names = [out.decode().rsplit('V')[-1]]

    # suppose version is MAJOR.MINOR.PATCH, all numbers
    version_regex = re.compile(r'[0-9]+\.[0-9]+\.[0-9]+')
    candidates = [c.group() for c in map(version_regex.search, candidate_names) if c]
    if len(candidates) > 0:
        # normally only one will be retrieved, take the first result
        return candidates[0]
    # if no candidates were found, try MAJOR.MINOR
    version_regex = re.compile(r'[0-9]+\.[0-9]+')
    candidates = [c.group() for c in map(version_regex.search, candidate_names) if c]
    if len(candidates) > 0:
        return candidates[0]

if check_negative_env_flag('USE_CUDA') or check_env_flag('USE_ROCM'):
    USE_CUDA = False
    CUDA_HOME = None
    CUDA_VERSION = None
else:
    if IS_LINUX or IS_DARWIN:
        CUDA_HOME = os.getenv('CUDA_HOME', LINUX_HOME)
    else:
        CUDA_HOME = os.getenv('CUDA_PATH', '').replace('\\', '/')
        if CUDA_HOME == '' and len(WINDOWS_HOME) > 0:
            CUDA_HOME = WINDOWS_HOME[0].replace('\\', '/')
    if not os.path.exists(CUDA_HOME):
        # We use nvcc path on Linux and cudart path on macOS
        if IS_LINUX or IS_WINDOWS:
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
    USE_CUDA = CUDA_HOME is not None
