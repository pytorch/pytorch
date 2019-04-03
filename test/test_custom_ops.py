from subprocess import check_call
from os.path import dirname as dir, join, abspath
import os

root = dir(dir(__file__))
print(root)
CUSTOM_OP_TEST = abspath(join(root, 'test/custom_operator'))
CUSTOM_OP_BUILD = join(CUSTOM_OP_TEST, 'build')
env = os.environ.copy()
env['CMAKE_PREFIX_PATH'] = abspath(join(root, "torch"))
check_call(['mkdir', '-p', CUSTOM_OP_BUILD])
check_call(['cmake', CUSTOM_OP_TEST], cwd=CUSTOM_OP_BUILD, env=env)
check_call(['make', 'VERBOSE=1', '-j8'], cwd=CUSTOM_OP_BUILD)
check_call(['python', 'test_custom_ops.py', '-v'], cwd=CUSTOM_OP_TEST)
check_call(['python', 'model.py', '--export-script-module=model.pt'], cwd=CUSTOM_OP_TEST)
check_call(['build/test_custom_ops', './model.pt'], cwd=CUSTOM_OP_TEST)
