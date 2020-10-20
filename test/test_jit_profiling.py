import sys
sys.argv.append("--jit_executor=profiling")
from test_jit import *

if __name__ == '__main__':
    run_tests()
    import test_jit_py3
    suite = unittest.findTestCases(test_jit_py3)
    unittest.TextTestRunner().run(suite)
