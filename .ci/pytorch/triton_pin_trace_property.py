"""Record the original exception hidden by a property's expected-failure wrapper."""

import sys
import traceback

import pytest


class TraceExpectedFailure:
    def __init__(self, remove_rocm_sub_xfail=False):
        self.remove_rocm_sub_xfail = remove_rocm_sub_xfail

    def pytest_runtest_call(self, item):
        if self.remove_rocm_sub_xfail:
            assert item.name.startswith('test_eager_equivalence_sub_backend_inductor_numerics_')
            item.module.ROCM_EAGER_EQUIV_XFAILS['inductor_numerics'].pop('sub', None)
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
    arguments = sys.argv[1:]
    remove_sub = '--remove-rocm-sub-xfail' in arguments
    if remove_sub:
        arguments.remove('--remove-rocm-sub-xfail')
    sys.exit(pytest.main(arguments, plugins=[TraceExpectedFailure(remove_sub)]))
