import unittest
from common_utils import TestCase, run_tests
import torch


def namedtensor_enabled():
    return '-DNAMEDTENSOR_ENABLED' in torch.__config__.show()

skipIfNamedTensorDisabled = \
    unittest.skipIf(not namedtensor_enabled(),
                    'PyTorch not compiled with namedtensor support')

class TestNamedTensor(TestCase):
    @skipIfNamedTensorDisabled
    def test_trivial(self):
        pass


if __name__ == '__main__':
    run_tests()
