import unittest
from common_utils import TestCase, run_tests
from common_cuda import TEST_CUDA
import torch
import torch.testing
import sys
from collections import namedtuple
import itertools


def namedtensor_enabled():
    return '-DNAMEDTENSOR_ENABLED' in torch.__config__.show()

skipIfNamedTensorDisabled = \
    unittest.skipIf(not namedtensor_enabled(),
                    'PyTorch not compiled with namedtensor support')

def pass_name_to_python_arg_parser(name):
    x = torch.empty(2, names=(name,))


class TestNamedTensor(TestCase):
    def test_trivial(self):
        pass

    def _test_create_tensor_with_names(self, factory, device):
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

        with self.assertRaisesRegex(RuntimeError, 'construct a tensor with duplicate names'):
            x = factory(2, 1, 1, names=('N', 'C', 'N'), device=device)

        # Tests for tagged names
        x = factory(2, 3, 1, names=('C.in', 'H', 'C.out'), device=device)
        self.assertEqual(x.names, ('C.in', 'H', 'C.out'))

        with self.assertRaisesRegex(RuntimeError, 'construct a tensor with duplicate names'):
            x = factory(2, 1, 1, names=('C.in', 'H', 'C.in'), device=device)

        with self.assertRaisesRegex(
                RuntimeError,
                'with duplicate names unless they are tagged and have different tags'):
            x = factory(2, 1, 1, names=('C.in', 'H', 'C'), device=device)


    def test_create_tensor_with_names(self):
        # All the creation methods go through the same codepath,
        # so let's just test one for all edge cases.
        for device in torch.testing.get_all_device_types():
            self._test_create_tensor_with_names(torch.empty, device)

    def test_creation_functions_use_names(self):
        Test = namedtuple('Test', ['expected_names', 'func'])
        names = ('N', 'D')
        tests = [
            Test(names, lambda dev: torch.empty(1, 1, names=names, device=dev)),
            Test(names, lambda dev: torch.ones(1, 1, names=names, device=dev)),
            Test(names, lambda dev: torch.zeros(1, 1, names=names, device=dev)),
            Test(names, lambda dev: torch.rand(1, 1, names=names, device=dev)),
            Test(names, lambda dev: torch.rand(1, 1, generator=torch.default_generator, names=names, device=dev)),
            Test(names, lambda dev: torch.randn(1, 1, names=names, device=dev)),
            Test(names, lambda dev: torch.randn(1, 1, generator=torch.default_generator, names=names, device=dev)),
        ]

        for test, device in itertools.product(tests, torch.testing.get_all_device_types()):
            self.assertEqual(test.func(device).names, test.expected_names)

    def test_using_seen_interned_string_doesnt_bump_refcount(self):
        def see_name():
            seen_name = 'N'
            pass_name_to_python_arg_parser(seen_name)

        see_name()
        seen_name = 'N'
        old_refcnt = sys.getrefcount(seen_name)

        pass_name_to_python_arg_parser(seen_name)

        new_refcnt = sys.getrefcount(seen_name)
        self.assertEqual(new_refcnt, old_refcnt)

    def test_using_unseen_interned_string_bumps_refcount_permanently(self):
        # Please don't use this as a name in a different test.
        unseen_name = 'abcdefghi'
        old_refcnt = sys.getrefcount(unseen_name)

        pass_name_to_python_arg_parser(unseen_name)

        new_refcnt = sys.getrefcount(unseen_name)
        self.assertEqual(new_refcnt, old_refcnt + 1)

    def test_using_unseen_uninterned_string_refcounts(self):
        # Please don't use this as a name in a different test.
        # non-compile-time constants are not interned
        unseen_name = ''.join(['abc', 'def', 'ghi', 'jkl'])
        interned_unseen_name = 'abcdefghijkl'
        self.assertFalse(unseen_name is interned_unseen_name)

        old_uninterned_refcnt = sys.getrefcount(unseen_name)
        old_interned_refcnt = sys.getrefcount(interned_unseen_name)

        pass_name_to_python_arg_parser(unseen_name)

        new_uninterned_refcnt = sys.getrefcount(unseen_name)
        new_interned_refcnt = sys.getrefcount(interned_unseen_name)

        # Internally, PyTorch should not hold a reference to the uninterned string
        self.assertEqual(new_uninterned_refcnt, old_uninterned_refcnt)

        # Instead, we should hold a new reference to the interned version.
        self.assertEqual(new_interned_refcnt, old_interned_refcnt + 1)

    def _test_select(self, device):
        x = torch.empty(2, 3, 4, 5, names=('N', 'C', 'H', 'W'), device=device)
        y = x.select(1, 1)
        self.assertEqual(y.names, ('N', 'H', 'W'))

        y = x.select('C', 1)
        self.assertEqual(y.names, ('N', 'H', 'W'))

        with self.assertRaisesRegex(
                RuntimeError, 'Please look up dimensions by name'):
            y = x.select(None, 1)

        with self.assertRaisesRegex(
                RuntimeError, 'Name \'C.in\' not found in'):
            y = x.select('C.in', 1)

        x = torch.empty(2, 3, 4, 5, names=('N', 'C.in', 'H', 'W'), device=device)
        y = x.select('C', 1)
        self.assertEqual(y.names, ('N', 'H', 'W'))

        x = torch.empty(2, 3, 4, 5, names=('C.out', 'C.in', 'H', 'W'), device=device)
        y = x.select('C.in', 1)
        self.assertEqual(y.names, ('C.out', 'H', 'W'))

        with self.assertRaisesRegex(
                RuntimeError, 'Name \'C\' could refer to multiple dimensions'):
            y = x.select('C', 1)


    def test_select(self):
        self._test_select('cpu')

    @unittest.skipIf(not TEST_CUDA, 'no CUDA')
    def test_select_cuda(self):
        self._test_select('cuda')

    def _test_as_strided(self, device):
        x = torch.empty(2, 3, 4, 5, names=('N', 'C', 'H', 'W'), device=device)
        y = x.as_strided([2 * 3 * 4 * 5], [1])
        self.assertEqual(y.names, (None,))

    def test_as_strided(self):
        self._test_as_strided('cpu')

    @unittest.skipIf(not TEST_CUDA, 'no CUDA')
    def test_as_strided_cuda(self):
        self._test_as_strided('cuda')

# Disable all tests if named tensor is not available.
for attr in dir(TestNamedTensor):
    if attr.startswith('test_'):
        new_test = skipIfNamedTensorDisabled(getattr(TestNamedTensor, attr))
        setattr(TestNamedTensor, attr, new_test)

if __name__ == '__main__':
    run_tests()
