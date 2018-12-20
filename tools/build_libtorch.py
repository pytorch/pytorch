# This module can be run either as a module, or as a standalone script.
# cd <pytorch_root>; python -m tools.setup.py  # module
# python ./.../build_libtorch.py             # standalone script

from __future__ import print_function

import argparse
import os
import platform
import shlex
import subprocess
import sys

dirname = os.path.dirname
pytorch_root = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(pytorch_root)

from tools.setup_helpers.env import check_env_flag, hotpatch_build_env_vars

hotpatch_build_env_vars()

from tools.setup_helpers.cuda import USE_CUDA
from tools.setup_helpers.dist_check import USE_DISTRIBUTED, USE_GLOO_IBVERBS, IS_LINUX


if __name__ == '__main__':
    # Placeholder for future interface. For now just gives a nice -h.
    parser = argparse.ArgumentParser(
        description='Build libtorch. Please add platform specific'
                    ' environment variables on your own.')
    options = parser.parse_args()

    IS_WINDOWS = (platform.system() == 'Windows')
    IS_DARWIN = (platform.system() == 'Darwin')

    MAC_REQUIRED_VARS = ['CC', 'CXX', 'MACOSX_DEPLOYMENT_TARGET']
    WINDOWS_REQUIRED_VARS = ['VS150COMNTOOLS', 'CMAKE_GENERATOR',
                             'DISTUTILS_USE_SDK', 'MSSdk', 'FORCE_PY27_BUILD']

    # TODO: test on Windows platform
    if IS_WINDOWS and any((var not in os.environ) for var in IS_WINDOWS):
        print("WARNING. You may have omitted some environment variables"
              " that are required:", ", ".join(IS_WINDOWS))
    if IS_DARWIN and any((var not in os.environ) for var in MAC_REQUIRED_VARS):
        print("WARNING. You may have omitted some environment variables"
              " that are required:", ", ".join(MAC_REQUIRED_VARS))

    os.environ['BUILD_TORCH'] = 'ON'
    os.environ['BUILD_TEST'] = 'ON'
    os.environ['ONNX_NAMESPACE'] = 'onnx_torch'
    os.environ['PYTORCH_PYTHON'] = sys.executable

    tools_path = os.path.join(pytorch_root, 'tools')
    build_path = os.path.join(pytorch_root, 'build')

    if IS_WINDOWS:
        # TODO: handle cwd if needed
        kwargs = {}
        build_pytorch_libs = os.path.join(tools_path, 'build_pytorch_libs.bat')
    else:
        os.makedirs(build_path, exist_ok=True)
        kwargs = {'cwd': build_path}
        build_pytorch_libs = os.path.join(tools_path, 'build_pytorch_libs.sh')

    command = [build_pytorch_libs, '--use-nnpack']
    USE_MKLDNN = check_env_flag('USE_MKLDNN', 'ON')
    if USE_MKLDNN:
        command.append('--use-mkldnn')
    if USE_CUDA:
        command.append('--use-cuda')
        if os.environ.get('USE_CUDA_STATIC_LINK', False):
            command.append('--cuda-static-link')
    if USE_DISTRIBUTED and IS_LINUX:
        if USE_GLOO_IBVERBS:
            command.append('--use-gloo-ibverbs')
        command.append('--use-distributed')

    command.append('caffe2')

    sys.stdout.flush()
    sys.stderr.flush()

    subprocess.check_call(command, universal_newlines=True, **kwargs)
