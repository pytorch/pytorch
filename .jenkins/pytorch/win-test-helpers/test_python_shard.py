import os
from os.path import exists
import subprocess
import sys
import contextlib


shard_number = os.environ['SHARD_NUMBER']


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


try:

    subprocess.call('python ' + os.environ['SCRIPT_HELPERS_DIR'] + '\\setup_pytorch_env.py', shell=True)

except Exception as e:

    subprocess.call('echo setup pytorch env failed', shell=True)
    subprocess.call('echo ' + e, shell=True)
    sys.exit()


with pushd('test'):

    gflags_exe = "C:\\Program Files (x86)\\Windows Kits\\10\\Debuggers\\x64\\gflags.exe"
    os.environ['GFLAGS_EXE'] = gflags_exe


    if shard_number == "1" and exists(gflags_exe):

        subprocess.call('echo Some smoke tests', shell=True)

        try:
            subprocess.call(gflags_exe + ' /i python.exe +sls', shell=True)
            subprocess.call('conda run -n test_env' + ' python ' + os.environ['SCRIPT_HELPERS_DIR'] + '\\run_python_nn_smoketests.py', shell=True)
            subprocess.call(gflags_exe + ' /i python.exe -sls', shell=True)

        except Exception as e:

            subprocess.call('echo shard dmoke test failed', shell=True)
            subprocess.call('echo ' + e, shell=True)
            sys.exit()


    subprocess.call('echo Copying over test times file', shell=True)
    subprocess.call('copy /Y ' + str(os.environ['PYTORCH_FINAL_PACKAGE_DIR_WIN']) +
        '\\.pytorch-test-times.json ' + str(os.environ['PROJECT_DIR_WIN']), shell=True)


    subprocess.call('echo Run nn tests', shell=True)

    try:
        subprocess.call('conda install -n test_env python run_test.py --exclude-jit-executor ' +
            '--exclude-distributed-tests --shard ' + shard_number + ' ' + str(os.environ['NUM_TEST_SHARDS']) +
                ' --verbose', shell=True)

    except Exception as e:

        subprocess.call('echo shard nn tests failed', shell=True)
        subprocess.call('echo ' + e, shell=True)
        sys.exit()
