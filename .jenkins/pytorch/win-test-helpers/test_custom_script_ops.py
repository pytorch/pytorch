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


subprocess.call('python ' + os.environ['SCRIPT_HELPERS_DIR'] + '\\setup_pytorch_env.py', shell=True)

subprocess.run(['git', 'submodule', 'update', '--init', '--recursive', '--jobs',
    '0', 'third_party/pybind11'])

os.chdir('test\\custom_operator')

# Build the custom operator library.
os.mkdir('build')

with pushd('build'):

    try:
        # Note: Caffe2 does not support MSVC + CUDA + Debug mode (has to be Release mode)
        subprocess.run(['echo', 'Executing CMake for custom_operator test...'])
        subprocess.run(['cmake', '-DCMAKE_PREFIX_PATH=' + str(os.environ['TMP_DIR_WIN']) +
            '\\build\\torch', '-DCMAKE_BUILD_TYPE=Release', '-GNinja', '..'])

        subprocess.run(['echo', 'Executing Ninja for custom_operator test...'])
        subprocess.run(['ninja', '-v'])

        subprocess.run(['echo', 'Ninja succeeded for custom_operator test.'])

    except Exception as e:

        subprocess.run(['echo', 'custom_operator test failed'])
        subprocess.run(['echo', e])
        sys.exit()


try:
    # Run tests Python-side and export a script module.
    subprocess.run(['conda', 'install', '-n', 'test_env', 'python', 'test_custom_ops.py', '-v'])

    # TODO: fix and re-enable this test
    # See https://github.com/pytorch/pytorch/issues/25155
    # subprocess.run(['python', 'test_custom_classes.py', '-v'])

    subprocess.run(['conda', 'install', '-n', 'test_env', 'python', 'module.py', '--export-script-module="build/model.pt"'])

    # Run tests C++-side and load the exported script module.
    os.chdir('build')
    os.environ['PATH'] = 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt\\bin\\x64;'\
        + str(os.environ['TMP_DIR_WIN']) + '\\build\\torch\\lib;' + str(os.environ['PATH'])

    subprocess.run(['test_custom_ops.exe', 'model.pt'])

except Exception as e:

    subprocess.run(['echo', 'test_custom_ops failed'])
    subprocess.run(['echo', e])
    sys.exit()
