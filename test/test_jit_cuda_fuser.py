# Owner(s): ["module: nvfuser"]

try:
    from _nvfuser.test_torchscript import *  # noqa: F403,F401
except ImportError:
    def run_tests():
        return
    pass

if __name__ == '__main__':
    run_tests()
