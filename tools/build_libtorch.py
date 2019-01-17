import argparse
import os
from os.path import dirname, abspath
import shlex
import subprocess
import sys

# By appending pytorch_root to sys.path, this module can import other torch
# modules even when run as a standalone script. i.e., it's okay either you
# do `python build_libtorch.py` or `python -m tools.build_libtorch`.
pytorch_root = dirname(dirname(abspath(__file__)))
sys.path.append(pytorch_root)

# If you want to modify flags or environmental variables that is set when
# building torch, you should do it in tools/setup_helpers/configure.py.
# Please don't add it here unless it's only used in LibTorch.
from tools.setup_helpers.configure import get_libtorch_env_with_flags, IS_WINDOWS

if __name__ == '__main__':
    # Placeholder for future interface. For now just gives a nice -h.
    parser = argparse.ArgumentParser(description='Build libtorch')
    options = parser.parse_args()

    tools_path = os.path.dirname(os.path.abspath(__file__))
    if IS_WINDOWS:
        build_pytorch_libs = os.path.join(tools_path, 'build_pytorch_libs.bat')
    else:
        build_pytorch_libs = os.path.join(tools_path, 'build_pytorch_libs.sh')

    command = [build_pytorch_libs]
    my_env, extra_flags = get_libtorch_env_with_flags()
    command.extend(extra_flags)
    command.append('caffe2')

    sys.stdout.flush()
    sys.stderr.flush()
    subprocess.check_call(command, universal_newlines=True, env=my_env)
