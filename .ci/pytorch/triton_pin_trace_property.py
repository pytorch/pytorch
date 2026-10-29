"""Record the original exception hidden by a property's expected-failure wrapper."""

import sys
import traceback

import pytest


class TraceExpectedFailure:
    def pytest_runtest_call(self, item):
        instance = getattr(item.obj, '__self__', None)
        original = getattr(instance, '_run_with_expected_failure', None)
        if original is None:
            return

        def wrapper(device, operation, backend, kind, dtype, test):
            def trace():
                try:
                    return test()
                except (AssertionError, RuntimeError):
                    print('PIN_ORIGINAL_EXPECTED_EXCEPTION', operation, kind, dtype, flush=True)
                    traceback.print_exc(file=sys.stdout)
                    raise

            return original(device, operation, backend, kind, dtype, trace)

        instance._run_with_expected_failure = wrapper


if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:], plugins=[TraceExpectedFailure()]))
