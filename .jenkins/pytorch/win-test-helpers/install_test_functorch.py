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


try:

    result = subprocess.run('python ' + os.environ['SCRIPT_HELPERS_DIR'] + '\\setup_pytorch_env.py', shell=True)
    result.check_returncode()

except Exception as e:

    subprocess.run('echo setup pytorch env failed', shell=True)
    subprocess.run('echo ' + str(e), shell=True)
    sys.exit()


subprocess.run('echo Installing test dependencies', shell=True)

try:
    result = subprocess.run('conda install -n test_env pip install networkx', shell=True)
    result.check_returncode()

except Exception as e:

    subprocess.run('echo install networkx failed', shell=True)
    subprocess.run('echo ' + str(e), shell=True)
    sys.exit()


subprocess.run('echo Test functorch', shell=True)

try:

    with pushd('test'):
        result = subprocess.run('conda install -n test_env python run_test.py --functorch --shard ' +
            os.environ['SHARD_NUMBER'] + ' ' + os.environ['NUM_TEST_SHARDS'] + ' --verbose', shell=True)
        result.check_returncode()

except Exception as e:

    subprocess.run('echo run_test functorch shard failed', shell=True)
    subprocess.run('echo ' + str(e), shell=True)
    sys.exit()
