import os
import subprocess
import shutil
import contextlib


subprocess.run('python ' + os.environ['SCRIPT_HELPERS_DIR'] + "\\setup_pytorch_env.py", shell=True)

subprocess.run('echo Copying over test times file', shell=True)
shutil.copy(str(os.environ['PYTORCH_FINAL_PACKAGE_DIR_WIN']) + "\\.pytorch-test-times.json",
    os.environ['PROJECT_DIR_WIN'])

try:
    subprocess.run('echo Run jit_profiling tests', shell=True, check=True)

    subprocess.run('conda install -n test_env python run_test.py --include test_jit_legacy ' +
        'test_jit_fuser_legacy --verbose', shell=True, check=True, cwd='test')

except Exception as e:
    pass
