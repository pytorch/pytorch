def load_tests(loader, standard_tests, pattern):
    """Load all tests from `test/pacakge/`
    """
    if pattern is None:
        # Use the default pattern if none is specified by the test loader.
        pattern = "test*.py"
    package_tests = loader.discover("package", pattern=pattern)
    standard_tests.addTests(package_tests)
    return standard_tests


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests
    run_tests()
