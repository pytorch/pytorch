import argparse
import os
import shlex
import subprocess
import sys

from setup_helpers.cuda import USE_CUDA
from setup_helpers.env import check_env_flag

if __name__ == '__main__':
    # Placeholder for future interface. For now just gives a nice -h.
    parser = argparse.ArgumentParser(description='Build libtorch')
    options = parser.parse_args()

    os.environ['BUILD_TORCH'] = 'ON'
    os.environ['BUILD_TEST'] = 'ON'
    os.environ['ONNX_NAMESPACE'] = 'onnx_torch'
    os.environ['PYTORCH_PYTHON'] = sys.executable

    tools_path = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == 'win32':
        build_pytorch_libs = os.path.join(tools_path, 'build_pytorch_libs.bat')
    else:
        build_pytorch_libs = os.path.join(tools_path, 'build_pytorch_libs.sh')

    command = [build_pytorch_libs, '--use-nnpack']
    USE_MKLDNN = check_env_flag('USE_MKLDNN', 'ON')
    if USE_MKLDNN:
        command.append('--use-mkldnn')
    if USE_CUDA:
        command.append('--use-cuda')
        if os.environ.get('USE_CUDA_STATIC_LINK', False):
            command.append('--cuda-static-link')
    command.append('caffe2')

    sys.stdout.flush()
    sys.stderr.flush()
    subprocess.check_call(command, universal_newlines=True)
