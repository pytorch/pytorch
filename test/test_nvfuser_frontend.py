# Owner(s): ["module: nvfuser"]

try:
    from _nvfuser.test_python_frontend import run_tests  # noqa: F403
except ImportError:
    def run_tests():
        return
    pass

if __name__ == '__main__':
    run_tests()
