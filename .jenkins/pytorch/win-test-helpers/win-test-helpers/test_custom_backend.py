import os
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


subprocess.run('python ' + os.environ['SCRIPT_HELPERS_DIR'] + '\\setup_pytorch_env.py', shell=True)

subprocess.run('git submodule update --init --recursive --jobs 0 third_party/pybind11', shell=True)

os.chdir('test\\custom_backend')

# Build the custom backend library.
os.mkdir('build')

with pushd('build'):

    subprocess.run('echo Executing CMake for custom_backend test...', shell=True)

    # Note: Caffe2 does not support MSVC + CUDA + Debug mode (has to be Release mode)
    try:
        result = subprocess.run('cmake -DCMAKE_PREFIX_PATH=' + str(os.environ['TMP_DIR_WIN']) +
            '\\build\\torch -DCMAKE_BUILD_TYPE=Release -GNinja ..', shell=True)
        result.check_returncode()

        result = subprocess.run('echo Executing Ninja for custom_backend test...', shell=True)
        result.check_returncode()

        result = subprocess.run('ninja -v', shell=True)
        result.check_returncode()

        result = subprocess.run('echo Ninja succeeded for custom_backend test.', shell=True)
        result.check_returncode()

    except Exception as e:

        subprocess.run('echo custom_backend cmake test failed', shell=True)
        subprocess.run('echo ' + str(e), shell=True)
        sys.exit()


try:

    # Run tests Python-side and export a script module.
    result = subprocess.run('conda install -n test_env python test_custom_backend.py -v', shell=True)
    result.check_returncode()

    result = subprocess.run('conda install -n test_env python backend.py --export-module-to="build/model.pt"', shell=True)
    result.check_returncode()

    # Run tests C++-side and load the exported script module.
    os.chdir('build')
    os.environ['PATH'] = 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt\\bin\\x64;'\
        + str(os.environ['TMP_DIR_WIN']) + '\\build\\torch\\lib;' + str(os.environ['PATH'])

    result = subprocess.run('test_custom_backend.exe model.pt', shell=True)
    result.check_returncode()

except Exception as e:

    subprocess.run('echo test_custom_backend failed', shell=True)
    subprocess.run('echo ' + str(e), shell=True)
    sys.exit()
