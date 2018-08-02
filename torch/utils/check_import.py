# This script tries to figure out the reason of
# `ImportError` on Windows.
# Run it with `python check_import.py`.

import ctypes
import glob
import os
import sys
import subprocess

PY3 = sys.version_info >= (3, 0)

TORCH_ROOT = os.path.dirname(os.path.dirname(__file__))
PY_DLL_PATH = os.path.join(os.path.dirname(sys.executable), 'Library\\bin')
TORCH_DLL_PATH = os.path.join(TORCH_ROOT, 'lib')
NVTOOLEXT_HOME = os.getenv(
    'NVTOOLSEXT_PATH', 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt')
NV_ROOT = os.path.dirname(NVTOOLEXT_HOME)

IS_CUDA = len(glob.glob(TORCH_ROOT + '\\_nvrtc*.pyd')) > 0
IS_CONDA = 'conda' in sys.version or 'Continuum' in sys.version or any(
    [x.startswith('CONDA') for x in os.environ])

VC_LIBS = ['msvcp140.dll']
MKL_LIBS = ['mkl_rt.dll']
INTEL_OPENMP_LIBS = ['libiomp5md.dll']
CUDA_LIBS = ['nvcuda.dll',
             'nvToolsExt64_1.dll',
             'nvfatbinaryLoader.dll']
TORCH_LIBS = ['shm.dll']


def add_paths(paths):
    """Add paths to `PATH`"""
    for path in paths:
        os.environ['PATH'] = path + ';' + os.environ['PATH']


def get_output(command):
    """Returns stdout if rc is not 0 else None"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    output, _ = p.communicate()
    rc = p.returncode
    if PY3:
        output = output.decode("ascii")
    if rc is not 0:
        return None
    return output.strip()


def get_file_path(filename):
    """Returns the path of a file in `PATH`"""
    out = get_output('where ' + filename)
    if out is None:
        return out
    else:
        return out.split('\r\n')[0]


def detect_reason(raw_message):
    message = ''
    if raw_message == 'No module named torch':
        # detect pip python path mismatch
        mismatch, pip_path, python_path = detect_install_import_mismatch()
        if mismatch:
            message += 'Probably you installed torch in one environment '
            message += 'but imported in another one.\n'
            message += 'Detected pip path: %s\n' % pip_path
            message += 'Detected python path: %s\n' % python_path
        else:
            message += 'It seems that torch is not installed.\n'
            message += 'Please refer to https://pytorch.org for installation.\n'
    elif raw_message.startswith('DLL load failed'):
        add_paths([NVTOOLEXT_HOME, PY_DLL_PATH])
        message += check_dependents(TORCH_LIBS, 'PyTorch', [
                                    '1. Please change your current directory.', '2. Please reinstall torch.'])
        message += check_dependents(VC_LIBS, 'VC Redist',
                                    'Please refer to https://aka.ms/vs/15/release/VC_redist.x64.exe for installation.')
        message += check_dependents(MKL_LIBS, 'MKL',
                                    '`conda install mkl` or `pip install mkl`')
        message += check_dependents(INTEL_OPENMP_LIBS, 'intel-openmp',
                                    '`conda install intel-openmp` or `pip install intel-openmp`')
        if IS_CUDA:
            if os.path.exists(NV_ROOT):
                message += check_dependents(
                    CUDA_LIBS, 'CUDA', 'Please refer to https://developer.nvidia.com/cuda-downloads for installation.')
            else:
                message += check_dependents(
                    CUDA_LIBS, 'CUDA', 'It seems that you don\'t have NV cards. Please use CPU version instead.')
        
        if message == '':
            message += 'It seems `import torch` should work.'
            message += 'You may try to add `%s` to the environment variable `PATH`.\n' % PY_DLL_PATH
            message += 'And make sure you restart the command prompt when you apply any changes to the environment.\n'

    return message


def detect_install_import_mismatch():
    pip_path = get_file_path('pip.exe')
    python_path = sys.executable
    if pip_path is None or python_path is None:
        return False
    pip_dir = os.path.dirname(pip_path)
    python_dir = os.path.dirname(python_path)
    pip_parent_path = os.path.normpath(os.path.dirname(pip_dir))
    python_path = os.path.normpath(python_dir)
    mismatch = pip_parent_path != python_path
    return mismatch, pip_path, python_path


def check_dependents(dependents, name, solution):
    """Checks dependencies loading and prints name and solution"""
    message = ''
    for dll in dependents:
        try:
            _ = ctypes.CDLL(dll)
        except Exception as e:
            message += 'DLL loading %s failed\n' % dll
            message += 'Original error message:\n'
            message += str(e)
            if name is not None:
                message += 'It is a component of %s\n' % name
            if solution:
                message += 'Possible solution:\n'
                if isinstance(solution, list):
                    message += '\n'.join(solution)
                else:
                    message += solution
                message += '\n'

    return message


def main():
    try:
        import torch
        print('`import torch` works perfectly.')
    except ImportError as e:
        message = detect_reason(str(e))
        print(message)


if __name__ == '__main__':
    main()
