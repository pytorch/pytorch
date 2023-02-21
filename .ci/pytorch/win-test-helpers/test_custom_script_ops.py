import os
import subprocess
import sys
import contextlib


subprocess.run('python ' + os.environ['SCRIPT_HELPERS_DIR'] + '\\setup_pytorch_env.py', shell=True)

subprocess.run('git submodule update --init --recursive --jobs 0 third_party/pybind11', shell=True)

os.chdir('test\\custom_operator')

# Build the custom operator library.
os.mkdir('build')


try:
    # Note: Caffe2 does not support MSVC + CUDA + Debug mode (has to be Release mode)
    subprocess.run('echo Executing CMake for custom_operator test...', shell=True, check=True, cwd='build')

    subprocess.run('cmake -DCMAKE_PREFIX_PATH=' + str(os.environ['TMP_DIR_WIN']) +
        '\\build\\torch -DCMAKE_BUILD_TYPE=Release -GNinja ..', shell=True, check=True, cwd='build')

    subprocess.run('echo Executing Ninja for custom_operator test...', shell=True, check=True, cwd='build')

    subprocess.run('ninja -v', shell=True, check=True, cwd='build')

    subprocess.run('echo Ninja succeeded for custom_operator test.', shell=True, check=True, cwd='build')

except Exception as e:

    subprocess.run('echo custom_operator test failed', shell=True, cwd='build')
    subprocess.run('echo ' + str(e), shell=True, cwd='build')
    sys.exit()


try:
    # Run tests Python-side and export a script module.
    subprocess.run('conda install -n test_env python test_custom_ops.py -v', shell=True, check=True)

    # TODO: fix and re-enable this test
    # See https://github.com/pytorch/pytorch/issues/25155
    # subprocess.run(['python', 'test_custom_classes.py', '-v'])

    subprocess.run('conda install -n test_env python module.py --export-script-module="build/model.pt"', shell=True, check=True)

    # Run tests C++-side and load the exported script module.
    os.chdir('build')
    os.environ['PATH'] = 'C:\\Program Files\\NVIDIA Corporation\\NvToolsExt\\bin\\x64;'\
        + str(os.environ['TMP_DIR_WIN']) + '\\build\\torch\\lib;' + str(os.environ['PATH'])

    subprocess.run('test_custom_ops.exe model.pt', shell=True, check=True)

except Exception as e:

    subprocess.run('echo test_custom_ops failed', shell=True)
    subprocess.run('echo ' + str(e), shell=True)
    sys.exit()
