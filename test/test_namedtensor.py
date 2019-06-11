import unittest
from common_utils import TestCase, run_tests
from common_cuda import TEST_CUDA
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

    def _test_factory(self, factory, device):
        x = factory([], device=device)
        self.assertEqual(x.names, ())

        x = factory(1, 2, 3, device=device)
        self.assertEqual(x.names, (None, None, None))

        x = factory(1, 2, 3, names=None, device=device)
        self.assertEqual(x.names, (None, None, None))

        x = factory(1, 2, 3, names=('N', 'T', 'D'), device=device)
        self.assertEqual(x.names, ('N', 'T', 'D'))

        x = factory(1, 2, 3, names=('N', None, 'D'), device=device)
        self.assertEqual(x.names, ('N', None, 'D'))

        with self.assertRaisesRegex(RuntimeError,
                                    'must contain alphabetical characters and/or underscore'):
            x = factory(2, names=('?',), device=device)

        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            x = factory(2, 1, names=('N',), device=device)

        with self.assertRaisesRegex(TypeError, 'invalid combination of arguments'):
            x = factory(2, 1, names='N', device=device)


    @skipIfNamedTensorDisabled
    def test_empty(self):
        self._test_factory(torch.empty, 'cpu')

    @skipIfNamedTensorDisabled
    @unittest.skipIf(not TEST_CUDA, 'no CUDA')
    def test_empty_cuda(self):
        self._test_factory(torch.empty, 'cuda')


if __name__ == '__main__':
    run_tests()
