import os
import subprocess
import shutil
import contextlib


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


subprocess.call('python ' + os.environ['SCRIPT_HELPERS_DIR'] + "\\setup_pytorch_env.py", shell=True)

subprocess.run(['echo', 'Copying over test times file'])
shutil.copy(str(os.environ['PYTORCH_FINAL_PACKAGE_DIR_WIN']) + "\\.pytorch-test-times.json",
    os.environ['PROJECT_DIR_WIN'])

with pushd('test'):
    try:
        subprocess.run(['echo', 'Run', 'jit_profiling', 'tests'])
        subprocess.run(['conda', 'install', '-n', 'test_env', 'python', 'run_test.py', '--include', 'test_jit_legacy',
            'test_jit_fuser_legacy', '--verbose'])
    except Exception as e:
        pass
