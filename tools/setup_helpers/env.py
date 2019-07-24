import os
import platform
import struct
import sys
from itertools import chain


IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')

IS_CONDA = 'conda' in sys.version or 'Continuum' in sys.version or any([x.startswith('CONDA') for x in os.environ])
CONDA_DIR = os.path.join(os.path.dirname(sys.executable), '..')

IS_64BIT = (struct.calcsize("P") == 8)


def check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def check_negative_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['OFF', '0', 'NO', 'FALSE', 'N']


def gather_paths(env_vars):
    return list(chain(*(os.getenv(v, '').split(os.pathsep) for v in env_vars)))


def lib_paths_from_base(base_path):
    return [os.path.join(base_path, s) for s in ['lib/x64', 'lib', 'lib64']]


def hotpatch_var(var, prefix='USE_'):
    def print_warning(good_prefix, bad_prefix, var):
        print(("The use of {bad_prefix}{var} is deprecated and will be removed on Feb 20, 2020."
               "Please use {good_prefix}{var} instead.").format(
                   good_prefix=good_prefix, bad_prefix=bad_prefix, var=var))

    if check_env_flag('NO_' + var):
        print_warning(prefix, 'NO_', var)
        os.environ[prefix + var] = '0'
    elif check_negative_env_flag('NO_' + var):
        print_warning(prefix, 'NO_', var)
        os.environ[prefix + var] = '1'
    elif check_env_flag('WITH_' + var):
        print_warning(prefix, 'WITH_', var)
        os.environ[prefix + var] = '1'
    elif check_negative_env_flag('WITH_' + var):
        print_warning(prefix, 'WITH_', var)
        os.environ[prefix + var] = '0'


def hotpatch_build_env_vars():
    # Before we run the setup_helpers, let's look for NO_* and WITH_* variables and hotpatch environment with the USE_*
    # equivalent The use of NO_* and WITH_* is deprecated and will be removed in Feb 20, 2020.
    use_env_vars = ['CUDA', 'CUDNN', 'FBGEMM', 'MKLDNN', 'NNPACK', 'DISTRIBUTED',
                    'OPENCV', 'TENSORRT', 'QNNPACK', 'FFMPEG', 'SYSTEM_NCCL',
                    'GLOO_IBVERBS']
    list(map(hotpatch_var, use_env_vars))

    # Also hotpatch a few with BUILD_* equivalent
    build_env_vars = ['BINARY', 'TEST', 'CAFFE2_OPS']
    [hotpatch_var(v, 'BUILD_') for v in build_env_vars]

hotpatch_build_env_vars()


class BuildType(object):
    """Checks build type. This avoids checking os.environ['CMAKE_BUILD_TYPE'] directly, which is error-prone.

    Args:
        cmake_build_type_env (str): The value of os.environ['CMAKE_BUILD_TYPE'].
    """

    def __init__(self, cmake_build_type_env):
        self.build_type_string = cmake_build_type_env

    def is_debug(self):
        "Checks Debug build."
        return self.build_type_string == 'Debug'

    def is_rel_with_deb_info(self):
        "Checks RelWithDebInfo build."
        return self.build_type_string == 'RelWithDebInfo'

    def is_release(self):
        "Checks Release build."
        return self.build_type_string == 'Release'


# Determining build type. 'CMAKE_BUILD_TYPE' always prevails over DEBUG or REL_WITH_DEB_INFO.
if 'CMAKE_BUILD_TYPE' not in os.environ:  # hotpatch environment variable 'CMAKE_BUILD_TYPE'
    if check_env_flag('DEBUG'):
        os.environ['CMAKE_BUILD_TYPE'] = 'Debug'
    elif check_env_flag('REL_WITH_DEB_INFO'):
        os.environ['CMAKE_BUILD_TYPE'] = 'RelWithDebInfo'
    else:
        os.environ['CMAKE_BUILD_TYPE'] = 'Release'

build_type = BuildType(os.environ['CMAKE_BUILD_TYPE'])
