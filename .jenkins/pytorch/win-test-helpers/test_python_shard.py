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

    subprocess.run(['echo', 'setup pytorch env failed'])
    subprocess.run(['echo', e])
    sys.exit()


with pushd('test'):

    gflags_exe = "C:\\Program Files (x86)\\Windows Kits\\10\\Debuggers\\x64\\gflags.exe"
    os.environ['GFLAGS_EXE'] = gflags_exe


    if shard_number == "1" and exists(gflags_exe):

        subprocess.run(['echo', 'Some smoke tests'])

        try:
            subprocess.run([gflags_exe, '/i', 'python.exe', '+sls'])
            subprocess.call('conda run -n test_env' + ' python ' + os.environ['SCRIPT_HELPERS_DIR'] + '\\run_python_nn_smoketests.py', shell=True)
            subprocess.run([gflags_exe, '/i', 'python.exe', '-sls'])

        except Exception as e:

            subprocess.run(['echo', 'shard dmoke test failed'])
            subprocess.run(['echo', e])
            sys.exit()


    subprocess.run(['echo', 'Copying over test times file'])
    subprocess.run(['copy', '/Y', str(os.environ['PYTORCH_FINAL_PACKAGE_DIR_WIN']) +
        '\\.pytorch-test-times.json', str(os.environ['PROJECT_DIR_WIN'])])


    subprocess.run(['echo', 'Run nn tests'])

    try:
        subprocess.run([*'conda run -n test_env'.split(), 'python', 'run_test.py', '--exclude-jit-executor',
            '--exclude-distributed-tests', '--shard', shard_number, str(os.environ['NUM_TEST_SHARDS']),
                '--verbose'])

    except Exception as e:

        subprocess.run(['echo', 'shard nn tests failed'])
        subprocess.run(['echo', e])
        sys.exit()
