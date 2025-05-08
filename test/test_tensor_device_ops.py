# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# Owner(s): ["module: tests"]

from torch.testing._internal.common_utils import TestCase

# TODO: these empty classes are temporarily instantiated for XLA compatibility
#   once XLA updates their test suite it should be removed
class TestTensorDeviceOps(TestCase):
    pass

if __name__ == '__main__':
    from torch.testing._internal.common_utils import run_tests
    run_tests()
