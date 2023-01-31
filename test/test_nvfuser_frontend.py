# Owner(s): ["module: nvfuser"]

try:
    from _nvfuser.test_python_frontend import *  # noqa: F403
except ImportError:
    pass

if __name__ == '__main__':
    run_tests()
