import os
from os.path import exists
import subprocess
import sys
import contextlib


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


subprocess.call(str(os.environ['SCRIPT_HELPERS_DIR']) + '\setup_pytorch_env.py', shell=True)

subprocess.run(['git', 'submodule', 'update', '--init', '--recursive', '--jobs',\
 '0', 'third_party/pybind11'])

os.chdir('test\\custom_backend')

# Build the custom backend library.
os.mkdir('build')

with pushd('build'):

    subprocess.run(['echo', 'Executing CMake for custom_backend test...'])

    # Note: Caffe2 does not support MSVC + CUDA + Debug mode (has to be Release mode)
    try:
        subprocess.run(['cmake', '-DCMAKE_PREFIX_PATH=' + str(os.environ['TMP_DIR_WIN']) +\
         '\\build\\torch', '-DCMAKE_BUILD_TYPE=Release', '-GNinja', '..'])

        subprocess.run(['echo', 'Executing Ninja for custom_backend test...'])
        subprocess.run(['ninja', '-v'])

        subprocess.run(['echo', 'Ninja succeeded for custom_backend test.'])

    except Exception as e:

        subprocess.run(['echo', 'custom_backend cmake test failed'])
        subprocess.run(['echo', e])
        sys.exit()


try:

    # Run tests Python-side and export a script module.
    subprocess.run(['python', 'test_custom_backend.py', '-v'])
    subprocess.run(['python', 'backend.py', '--export-module-to="build/model.pt"'])

    # Run tests C++-side and load the exported script module.
    os.chdir('build')
    os.environ['PATH']='C:\\Program Files\\NVIDIA Corporation\\NvToolsExt\\bin\\x64;'\
     + str(os.environ['TMP_DIR_WIN']) + '\\build\\torch\\lib;' + str(os.environ['PATH'])

    subprocess.run(['test_custom_backend.exe', 'model.pt'])

except Exception as e:

    subprocess.run(['echo', 'test_custom_backend failed'])
    subprocess.run(['echo', e])
    sys.exit()
