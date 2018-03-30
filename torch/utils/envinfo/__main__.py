import re
import subprocess
import sys
from collections import namedtuple

import torch

PY3 = sys.version_info >= (3, 0)

# System Environment Information
SystemEnv = namedtuple('SystemEnv', [
    'torch_version',
    'is_debug_build',
    'os',
    'python_version',
    'is_cuda_available',
    'cuda_compiled_version',
    'cuda_runtime_version',
    'nvidia_driver_version',
    'nvidia_gpu_models',
    'pip_version',  # 'pip' or 'pip3'
    'pip_packages',
    'gcc_version',
    'cmake_version',
])

# General scheme is that None represents an error in data collection
# and NOT_APPLICABLE represents something like not having gpu models on a CPU-only machine
NOT_APPLICABLE = 'N/A'


def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    if PY3:
        output = output.decode("ascii")
        err = err.decode("ascii")
    return rc, output, err


def get_gpu_info():
    if not torch.cuda.is_available():
        return NOT_APPLICABLE
    rc, out, _ = run('nvidia-smi -L')
    if rc is not 0:
        return None
    return out


def run_and_parse_first_match(command, regex):
    rc, out, _ = run(command)
    if rc is not 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def get_running_cuda_version():
    return run_and_parse_first_match('nvcc --version', r'V(.*)$')


def get_gcc_version():
    return run_and_parse_first_match('gcc --version', r'gcc (.*)')


def get_cmake_version():
    return run_and_parse_first_match('cmake --version', r'cmake version (.*)')


def get_nvidia_driver_version():
    return run_and_parse_first_match('nvidia-smi', r'Driver Version: (.*?) ')


def get_pip_packages():
    # People generally have `pip` as `pip` or `pip3`
    def run_with_pip(pip):
        rc, out, _ = run(pip + ' list --format=legacy | grep torch')
        if rc is 0:
            return out
        return None

    if not PY3:
        return 'pip', run_with_pip('pip')

    # Try to figure out if the user is running pip or pip3.
    out2 = run_with_pip('pip')
    out3 = run_with_pip('pip3')

    num_pips = len([x for x in [out2, out3] if x is not None])
    if num_pips is 0:
        return 'pip', out2

    if num_pips == 1:
        if out2 is not None:
            return 'pip', out2
        return 'pip3', out3

    # num_pips is 2. Return pip3 by default b/c that most likely
    # is the one associated with Python 3
    return 'pip3', out3


def get_env_info():
    pip_version, pip_list_output = get_pip_packages()

    return SystemEnv(
        torch_version=torch.__version__,
        is_debug_build=torch.version.debug,
        python_version='{}.{}'.format(sys.version_info[0], sys.version_info[1]),
        is_cuda_available=torch.cuda.is_available(),
        cuda_compiled_version=torch.version.cuda,
        cuda_runtime_version=get_running_cuda_version(),
        nvidia_gpu_models=get_gpu_info(),
        nvidia_driver_version=get_nvidia_driver_version(),
        pip_version=pip_version,
        pip_packages=pip_list_output,
        os=None,
        gcc_version=get_gcc_version(),
        cmake_version=get_cmake_version(),
    )


def main():
    print(get_env_info())

if __name__ == '__main__':
    main()
