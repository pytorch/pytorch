import argparse
import os
import shlex
import subprocess
import sys

# Placeholder for future interface. For now just gives a nice -h.
parser = argparse.ArgumentParser(description='Build libtorch')
parser.parse_args()

os.environ['BUILD_TORCH'] = 'ON'
os.environ['ONNX_NAMESPACE'] = 'onnx_torch'

tools_path = os.path.dirname(os.path.abspath(__file__))
build_pytorch_libs = os.path.join(tools_path, 'build_pytorch_libs.sh')
command = '{} --use-nnpack caffe2'.format(build_pytorch_libs)

sys.stdout.flush()
sys.stderr.flush()
subprocess.call(shlex.split(command), universal_newlines=True)
