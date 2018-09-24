import argparse
import os
import shlex
import subprocess
import sys

from setup_helpers.cuda import USE_CUDA

if __name__ == '__main__':
    # Placeholder for future interface. For now just gives a nice -h.
    parser = argparse.ArgumentParser(description='Build libtorch')
    parser.add_argument('--use-cereal', action='store_true')
    options = parser.parse_args()

    os.environ['BUILD_TORCH'] = 'ON'
    os.environ['BUILD_TEST'] = 'ON'
    os.environ['ONNX_NAMESPACE'] = 'onnx_torch'
    os.environ['PYTORCH_PYTHON'] = sys.executable

    tools_path = os.path.dirname(os.path.abspath(__file__))
    build_pytorch_libs = os.path.join(tools_path, 'build_pytorch_libs.sh')

    command = [build_pytorch_libs, '--use-nnpack']
    if USE_CUDA:
        command.append('--use-cuda')
        if os.environ.get('USE_CUDA_STATIC_LINK', False):
            command.append('--cuda-static-link')
    if options.use_cereal:
        command.append('--use-cereal')
    command.append('caffe2')

    sys.stdout.flush()
    sys.stderr.flush()
    subprocess.check_call(command, universal_newlines=True)
