import sys
sys.argv.append("--ge_config=simple")
from test_jit import *

if __name__ == '__main__':
    run_tests()
    if not PY2:
        import test_jit_py3
        suite = unittest.findTestCases(test_jit_py3)
        unittest.TextTestRunner().run(suite)
