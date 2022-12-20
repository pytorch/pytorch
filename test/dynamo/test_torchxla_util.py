# Owner(s): ["module: dynamo"]
import functools
import unittest


@functools.lru_cache(None)
def should_run_torchxla_tests():
    """
    Run the tests if torch_xla is available and xla_device can be init.
    """
    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        return False
    try:
        device = xm.xla_device()
    except RuntimeError:
        return False
    return True


def maybe_skip_torchxla_test(test_case):
    return unittest.skipIf(
        not should_run_torchxla_tests(),
        "Skip the tests since torch_xla is not available or XLA devices are not specified",
    )(test_case)
