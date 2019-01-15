import argparse
import os
import shlex
import subprocess
import sys

# If you want to modify flags or environmental variables that is set when
# building torch, you should do it in tools/setup_helpers/configure.py.
# Please don't add it here unless it's only used in LibTorch.
from setup_helpers.configure import get_libtorch_env_with_flags

if __name__ == '__main__':
    # Placeholder for future interface. For now just gives a nice -h.
    parser = argparse.ArgumentParser(description='Build libtorch')
    options = parser.parse_args()

    tools_path = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == 'win32':
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
