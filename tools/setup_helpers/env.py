import os
import platform
import sys
from itertools import chain


IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')
IS_PPC = (platform.machine() == 'ppc64le')
IS_ARM = (platform.machine() == 'aarch64')

IS_CONDA = 'conda' in sys.version or 'Continuum' in sys.version or any([x.startswith('CONDA') for x in os.environ])
CONDA_DIR = os.path.join(os.path.dirname(sys.executable), '..')


def check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def check_negative_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['OFF', '0', 'NO', 'FALSE', 'N']


def gather_paths(env_vars):
    return list(chain(*(os.getenv(v, '').split(os.pathsep) for v in env_vars)))


def lib_paths_from_base(base_path):
    return [os.path.join(base_path, s) for s in ['lib/x64', 'lib', 'lib64']]


def hotpatch_var(var, prefix='USE_'):
    if check_env_flag('NO_' + var):
        os.environ[prefix + var] = '0'
    elif check_negative_env_flag('NO_' + var):
        os.environ[prefix + var] = '1'
    elif check_env_flag('WITH_' + var):
        os.environ[prefix + var] = '1'
    elif check_negative_env_flag('WITH_' + var):
        os.environ[prefix + var] = '0'


def hotpatch_build_env_vars():
    # Before we run the setup_helpers, let's look for NO_* and WITH_*
    # variables and hotpatch environment with the USE_* equivalent
    use_env_vars = ['CUDA', 'CUDNN', 'FBGEMM', 'MIOPEN', 'MKLDNN', 'NNPACK', 'DISTRIBUTED',
                    'OPENCV', 'TENSORRT', 'QNNPACK', 'FFMPEG', 'SYSTEM_NCCL',
                    'GLOO_IBVERBS']
    list(map(hotpatch_var, use_env_vars))

    # Also hotpatch a few with BUILD_* equivalent
    build_env_vars = ['BINARY', 'TEST', 'CAFFE2_OPS']
    [hotpatch_var(v, 'BUILD_') for v in build_env_vars]

hotpatch_build_env_vars()

DEBUG = check_env_flag('DEBUG')
REL_WITH_DEB_INFO = check_env_flag('REL_WITH_DEB_INFO')
USE_MKLDNN = check_env_flag('USE_MKLDNN', 'OFF' if IS_PPC or IS_ARM else 'ON')
