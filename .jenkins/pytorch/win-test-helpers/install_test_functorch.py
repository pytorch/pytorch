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

    subprocess.check_call('python ' + os.environ['SCRIPT_HELPERS_DIR'] + '\\setup_pytorch_env.py', shell=True)

except Exception as e:

    subprocess.check_call('echo setup pytorch env failed', shell=True)
    subprocess.check_call('echo ' + str(e), shell=True)
    sys.exit()


subprocess.check_call('echo Installing test dependencies', shell=True)

try:
    subprocess.check_call('conda install -n test_env pip install networkx', shell=True)

except Exception as e:

    subprocess.check_call('echo install networkx failed', shell=True)
    subprocess.check_call('echo ' + str(e), shell=True)
    sys.exit()


subprocess.check_call('echo Test functorch', shell=True)

try:

    with pushd('test'):
        subprocess.check_call('conda install -n test_env python run_test.py --functorch --shard ' +
            os.environ['SHARD_NUMBER'] + ' ' + os.environ['NUM_TEST_SHARDS'] + ' --verbose', shell=True)

except Exception as e:

    subprocess.check_call('echo run_test functorch shard failed', shell=True)
    subprocess.check_call('echo ' + str(e), shell=True)
    sys.exit()
