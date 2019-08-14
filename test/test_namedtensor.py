import unittest
from common_utils import TestCase, run_tests
from common_cuda import TEST_CUDA
from collections import namedtuple
import itertools
import torch
import torch.nn.functional as F
import sys


skipIfNamedTensorDisabled = \
    unittest.skipIf(not torch._C._BUILD_NAMEDTENSOR,
                    'PyTorch not compiled with namedtensor support')

def pass_name_to_python_arg_parser(name):
    x = torch.empty(2, names=(name,))


def flatten(lst):
    return [item for sublist in lst for item in sublist]


Function = namedtuple('TestCase', ['name', 'lambd'])


class TestNamedTensor(TestCase):
    def test_trivial(self):
        pass

    # TODO(rzou): Some form of this check should be added to self.assertEqual.
    # Right now I don't know what it should look like.
    def assertTensorDataAndNamesEqual(self, x, y):
        self.assertEqual(x.names, y.names)
        unnamed_x = x.view_names(None)
        unnamed_y = y.view_names(None)
        self.assertEqual(unnamed_x, unnamed_y)

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

    def test_has_names(self):
        unnamed = torch.empty(2, 3)
        none_named = torch.empty(2, 3, names=(None, None))
        partially_named = torch.empty(2, 3, names=('N', None))
        fully_named = torch.empty(2, 3, names=('N', 'C'))

        self.assertFalse(unnamed.has_names())
        self.assertFalse(none_named.has_names())
        self.assertTrue(partially_named.has_names())
        self.assertTrue(fully_named.has_names())

    def test_repr(self):
        named_tensor = torch.zeros(2, 3).names_('N', 'C')
        expected = "tensor([[0., 0., 0.],\n        [0., 0., 0.]], names=('N', 'C'))"
        self.assertEqual(repr(named_tensor), expected)

        unnamed_tensor = torch.zeros(2, 3)
        expected = "tensor([[0., 0., 0.],\n        [0., 0., 0.]])"
        self.assertEqual(repr(unnamed_tensor), expected)

        none_named_tensor = torch.zeros(2, 3).names_(None, None)
        self.assertEqual(repr(none_named_tensor), expected)

    def test_noncontig_contiguous(self):
        # This type of contiguous is special-cased and therefore needs its own test
        for device in torch.testing.get_all_device_types():
            x = torch.randn(2, 3, device=device).t().names_('N', 'C')
            self.assertEqual(x.contiguous().names, ('N', 'C'))

    def test_copy_transpose(self):
        # This type of copy is special-cased and therefore needs its own test
        def _test(self_names, other_names, expected_names):
            x = torch.empty(2, 5, names=self_names)
            y = torch.empty(5, 2).t().names_(*other_names)
            x.copy_(y)
            self.assertEqual(x.names, expected_names)

        _test(('N', 'C'), ('N', 'C'), ('N', 'C'))
        _test(None, ('N', 'C'), ('N', 'C'))

    def test_names_(self):
        tensor = torch.empty(1, 1, names=('N', 'C'))
        self.assertEqual(tensor.names_(None).names, (None, None))
        self.assertEqual(tensor.names_('H', 'W').names, ('H', 'W'))
        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.names_('N', 'C', 'W')
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.names_('N', 'N')

    def test_view_names(self):
        tensor = torch.empty(1, 1, names=('N', 'C'))

        self.assertEqual(tensor.view_names(None).names, (None, None))
        self.assertEqual(tensor.view_names('H', 'W').names, ('H', 'W'))

        # Check that we didn't modify tensor.names
        self.assertEqual(tensor.names, ('N', 'C'))

        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.view_names('N', 'C', 'W')
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.view_names('N', 'N')

        with self.assertRaisesRegex(RuntimeError, 'either positional args or keyword args'):
            tensor.view_names(None, N='batch')

        # view_names returns a view on the tensor
        self.assertEqual(tensor.view_names('H', 'W').data_ptr(), tensor.data_ptr())
        self.assertEqual(tensor.view_names(None).data_ptr(), tensor.data_ptr())

    def test_view_names_globber(self):
        scalar = torch.randn([])
        unnamed_tensor = torch.empty(1, 1, 1, 1)
        named_tensor = torch.empty(1, 1, 1, 1, names=('N', 'C', 'H', 'W'))

        self.assertEqual(scalar.view_names(None).names, [])
        self.assertEqual(scalar.view_names('*').names, [])

        # Check that it works with unnamed tensors
        self.assertEqual(unnamed_tensor.view_names('*').names, unnamed_tensor.names)
        self.assertEqual(unnamed_tensor.view_names('*', 'H', 'W').names,
                         [None, None, 'H', 'W'])
        self.assertEqual(unnamed_tensor.view_names('N', '*', 'W').names,
                         ['N', None, None, 'W'])
        self.assertEqual(unnamed_tensor.view_names('N', 'C', '*').names,
                         ['N', 'C', None, None])

        # Check that it works with named tensors
        self.assertEqual(named_tensor.view_names('*').names, named_tensor.names)
        self.assertEqual(named_tensor.view_names('*', 'width').names,
                         ['N', 'C', 'H', 'width'])
        self.assertEqual(named_tensor.view_names('batch', 'channels', '*', 'width').names,
                         ['batch', 'channels', 'H', 'width'])
        self.assertEqual(named_tensor.view_names('batch', '*').names,
                         ['batch', 'C', 'H', 'W'])

        # Test empty glob
        self.assertEqual(unnamed_tensor.view_names('*', None, None, None, None).names,
                         [None, None, None, None])
        self.assertEqual(named_tensor.view_names('N', 'C', 'H', '*', 'W').names,
                         ['N', 'C', 'H', 'W'])

        # Multiple globs throw
        with self.assertRaisesRegex(RuntimeError, 'More than one '):
            named_tensor.view_names('*', 'channels', '*')

    def test_view_names_rename_map(self):
        scalar = torch.randn([])
        unnamed_tensor = torch.empty(1, 1, 1, 1)
        named_tensor = torch.empty(1, 1, 1, 1, names=('N', 'C', 'H', 'W'))

        with self.assertRaisesRegex(RuntimeError, "dim 'N' does not exist"):
            scalar.view_names(N='batch')
        with self.assertRaisesRegex(RuntimeError, "dim 'N' does not exist"):
            unnamed_tensor.view_names(N='batch')
        with self.assertRaisesRegex(RuntimeError, "dim 'B' does not exist"):
            named_tensor.view_names(B='batch')
        with self.assertRaisesRegex(RuntimeError, "dim 'B' does not exist"):
            named_tensor.view_names(H='height', B='batch')

        self.assertEqual(named_tensor.view_names(N='batch').data_ptr(),
                         named_tensor.data_ptr())
        self.assertEqual(named_tensor.view_names(N='batch').names,
                         ['batch', 'C', 'H', 'W'])
        self.assertEqual(named_tensor.view_names(N='batch', H='height').names,
                         ['batch', 'C', 'height', 'W'])

    def test_set_names_property(self):
        tensor = torch.empty(1, 1, names=('N', 'C'))

        tensor.names = None
        self.assertEqual(tensor.names, (None, None))

        tensor.names = ('N', 'W')
        self.assertEqual(tensor.names, ('N', 'W'))

        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.names = ['N', 'C', 'W']
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.names = ['N', 'N']

    def test_factory_edge_cases(self):
        for device in torch.testing.get_all_device_types():
            self._test_factory(torch.empty, device)

    def test_factory_coverage(self):
        def _test(factory, device):
            names = ('N', 'T', 'D')

            torch.manual_seed(0)
            result = factory(1, 2, 3, names=names, device=device)

            torch.manual_seed(0)
            expected = factory(1, 2, 3, device=device).names_(*names)

            self.assertTensorDataAndNamesEqual(result, expected)

        supported = [
            torch.ones,
            torch.rand,
            torch.randn,
            torch.zeros,
        ]

        for op, device in itertools.product(supported, torch.testing.get_all_device_types()):
            _test(op, device)

        # Test torch.full
        for device in torch.testing.get_all_device_types():
            names = ('N', 'T', 'D')
            result = torch.full([1, 2, 3], 2, names=names, device=device)
            expected = torch.full([1, 2, 3], 2, device=device).names_(*names)
            self.assertTensorDataAndNamesEqual(result, expected)

    def test_size(self):
        t = torch.empty(2, 3, 5, names=('N', None, 'C'))
        self.assertEqual(t.size('N'), 2)
        self.assertEqual(t.size('C'), 5)
        with self.assertRaisesRegex(RuntimeError, 'Please look up dimensions by name*'):
            t.size(None)
        with self.assertRaisesRegex(RuntimeError, 'Name \'channels\' not found in '):
            t.size('channels')
        with self.assertRaisesRegex(RuntimeError, 'Name \'N\' not found in '):
            torch.empty(2, 3, 4).size('N')

    def test_stride(self):
        t = torch.empty(2, 3, 5, names=('N', None, 'C'))
        self.assertEqual(t.stride('N'), 3 * 5)
        self.assertEqual(t.stride('C'), 1)
        with self.assertRaisesRegex(RuntimeError, 'Please look up dimensions by name'):
            t.stride(None)
        with self.assertRaisesRegex(RuntimeError, 'Name \'channels\' not found in '):
            t.stride('channels')
        with self.assertRaisesRegex(RuntimeError, 'Name \'N\' not found in '):
            torch.empty(2, 3, 4).stride('N')

    def test_info_smoke(self):
        # Smoke test for info functions / methods / attributes on named tensors.
        tensor = torch.empty(1, 1, names=('N', 'D'))

        tensor.device
        tensor.dtype
        tensor.get_device()
        tensor.is_complex()
        tensor.is_floating_point()
        tensor.is_nonzero()
        torch.is_same_size(tensor, tensor)
        torch.is_signed(tensor)
        tensor.layout
        tensor.numel()
        tensor.dim()
        tensor.element_size()
        tensor.is_contiguous()
        tensor.is_cuda
        tensor.is_leaf
        tensor.is_pinned()
        tensor.is_shared()
        tensor.is_sparse
        tensor.ndimension()
        tensor.nelement()
        tensor.shape
        tensor.size()
        tensor.size(1)
        tensor.storage()
        tensor.storage_offset()
        tensor.storage_type()
        tensor.stride()
        tensor.stride(1)
        tensor.data
        tensor.data_ptr()
        tensor.ndim
        tensor.item()
        tensor.type()

    def test_split_fns_propagates_names(self):
        fns = [
            lambda x: x.split(1, 0),
            lambda x: x.split([1, 1], 1),
            lambda x: x.chunk(2, 0),
        ]

        for device in torch.testing.get_all_device_types():
            orig_tensor = torch.empty(2, 2, names=('N', 'D'), device=device)
            for fn in fns:
                splits = fn(orig_tensor)
                for split in splits:
                    self.assertEqual(split.names, orig_tensor.names)

    def test_binary_ops(self):
        def test_basic(op):
            a = torch.empty(2, 3, names=('N', 'C'))
            b = torch.empty(2, 3, names=('C', 'N'))
            c = torch.empty(3, names=('C',))
            d = torch.empty(3, names=('W',))

            self.assertEqual(op(a, a).names, ('N', 'C'))
            self.assertEqual(op(a, c).names, ('N', 'C'))

            with self.assertRaisesRegex(RuntimeError, "do not match"):
                op(a, d)
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                op(a, b)

        def test_wildcard(op):
            a = torch.empty(2, 3, names=('N', 'C'))
            c = torch.empty(2, 3, names=(None, 'C'))
            self.assertEqual(op(a, c).names, ('N', 'C'))

            b = torch.empty(2, 3)
            self.assertEqual(op(a, b).names, ('N', 'C'))

            d = torch.empty(2, 3, names=('C', None))
            with self.assertRaisesRegex(RuntimeError, "misaligned"):
                op(d, c)

        def method(name, *args, **kwargs):
            return [Function(name, lambda a, b: getattr(a, name)(b, *args, **kwargs))]

        def out_function(name, *args, **kwargs):
            out_fn = getattr(torch, name)

            def fn(a, b):
                result = a.new_empty([0])
                out_fn(a, b, *args, out=result, **kwargs)
                return result

            return [Function(name, fn)]

        def fn_method_and_inplace(name, *args, **kwargs):
            return (
                method(name, *args, **kwargs) +
                method(name + '_', *args, **kwargs) +
                out_function(name, *args, **kwargs)
            )

        tests = [
            fn_method_and_inplace('add'),
            fn_method_and_inplace('div'),
            fn_method_and_inplace('mul'),
            fn_method_and_inplace('sub'),
            method('copy_'),
        ]
        tests = flatten(tests)

        for _, op in tests:
            test_basic(op)
            test_wildcard(op)

    def test_out_fn_semantics(self):
        out_fn = torch.abs
        unnamed_tensor = torch.randn(3, 2)
        none_named_tensor = torch.randn(3, 2, names=(None, None))
        named_tensor = torch.randn(3, 2, names=('N', 'C'))
        partially_named_tensor = torch.randn(3, 2, names=('N', None))

        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            out_fn(partially_named_tensor, out=named_tensor)
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            out_fn(named_tensor, out=partially_named_tensor)
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            out_fn(none_named_tensor, out=named_tensor)
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            out_fn(unnamed_tensor, out=named_tensor)

        output = torch.randn(3, 2)
        out_fn(unnamed_tensor, out=output)
        self.assertFalse(output.has_names())

        output = torch.randn(3, 2, names=(None, None))
        out_fn(named_tensor, out=output)
        self.assertEqual(output.names, named_tensor.names)

        output = torch.randn(3, 2)
        out_fn(named_tensor, out=output)
        self.assertEqual(output.names, named_tensor.names)

        output = torch.randn(3, 2, names=(None, None))
        out_fn(unnamed_tensor, out=output)
        self.assertFalse(output.has_names())

    def test_unary_propagate_names_fns(self):
        def _test(testcase, names=('N', 'D'), device='cpu'):
            sizes = [2] * len(names)
            tensor = torch.empty(sizes, names=names, device=device)
            out = testcase.lambd(tensor)
            self.assertEqual(out.names, tensor.names,
                             message=testcase.name)

        def fn(name, *args, **kwargs):
            return [Function(name, lambda t: getattr(torch, name)(t, *args, **kwargs))]

        def method(name, *args, **kwargs):
            return [Function(name, lambda t: getattr(t, name)(*args, **kwargs))]

        def out_function(name, *args, **kwargs):
            out_fn = getattr(torch, name)

            def fn(tensor):
                result = tensor.new_empty([0])
                out_fn(tensor, *args, out=result, **kwargs)
                return result

            return [Function(name + '_out', fn)]

        def fn_method_and_inplace(name, *args, **kwargs):
            return (
                method(name, *args, **kwargs) +
                method(name + '_', *args, **kwargs) +
                out_function(name, *args, **kwargs)
            )

        # All of these operate on 2x2 tensors.
        tests = [
            # unary pointwise
            fn_method_and_inplace('abs'),
            fn_method_and_inplace('acos'),
            fn_method_and_inplace('asin'),
            fn_method_and_inplace('atan'),
            fn_method_and_inplace('ceil'),
            fn_method_and_inplace('clamp', -1, 1),
            fn_method_and_inplace('clamp_min', -2),
            fn_method_and_inplace('clamp_max', 2),
            method('cauchy_'),
            method('clone'),
            method('contiguous'),
            fn_method_and_inplace('cos'),
            fn_method_and_inplace('cosh'),
            fn_method_and_inplace('digamma'),
            fn_method_and_inplace('erf'),
            fn_method_and_inplace('erfc'),
            fn_method_and_inplace('erfinv'),
            fn_method_and_inplace('exp'),
            fn_method_and_inplace('expm1'),
            method('exponential_'),
            fn_method_and_inplace('floor'),
            fn_method_and_inplace('frac'),
            method('geometric_', p=0.5),
            fn_method_and_inplace('lgamma'),
            fn_method_and_inplace('log'),
            fn_method_and_inplace('log10'),
            fn_method_and_inplace('log1p'),
            fn_method_and_inplace('log2'),
            method('log_normal_'),
            fn_method_and_inplace('neg'),
            method('normal_'),
            [Function('polygamma', lambda t: torch.polygamma(1, t))],
            method('polygamma_', 1),
            fn_method_and_inplace('reciprocal'),
            method('random_', 0, 1),
            method('random_', 1),
            method('random_'),
            fn_method_and_inplace('round'),
            fn_method_and_inplace('rsqrt'),
            fn_method_and_inplace('sigmoid'),
            fn_method_and_inplace('sign'),
            fn_method_and_inplace('sin'),
            fn_method_and_inplace('sinh'),
            fn_method_and_inplace('sqrt'),
            fn_method_and_inplace('tan'),
            fn_method_and_inplace('tanh'),
            fn_method_and_inplace('trunc'),
            method('uniform_'),
            method('zero_'),
            method('fill_', 1),
            method('fill_', torch.tensor(3.14)),

            # conversions
            method('to', dtype=torch.long),
            method('to', device='cpu'),
            method('to', torch.empty([])),
            method('bool'),
            method('byte'),
            method('char'),
            method('cpu'),
            method('double'),
            method('float'),
            method('long'),
            method('half'),
            method('int'),
            method('short'),
            method('type', dtype=torch.long),

            # views
            method('narrow', 0, 0, 1),

            # creation functions
            fn('empty_like'),

            # bernoulli variants
            method('bernoulli_', 0.5),
            method('bernoulli_', torch.tensor(0.5)),

            [Function('F.dropout(inplace)', lambda t: F.dropout(t, p=0.5, inplace=True))],
            [Function('F.dropout(outplace)', lambda t: F.dropout(t, p=0.5, inplace=False))],
        ]
        tests = flatten(tests)

        for testcase, device in itertools.product(tests, torch.testing.get_all_device_types()):
            _test(testcase, device=device)

    def test_bernoulli(self):
        for device in torch.testing.get_all_device_types():
            names = ('N', 'D')
            tensor = torch.rand(2, 3, names=names)
            result = torch.empty(0)
            self.assertEqual(tensor.bernoulli().names, names)

            torch.bernoulli(tensor, out=result)
            self.assertEqual(result.names, names)

    def test_reduction_fns(self):
        def test_simple_reduce(op_name, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            op = getattr(torch.Tensor, op_name)
            self.assertEqual(op(t, 1).names, ['N', 'L'])
            self.assertEqual(op(t, 'C').names, ['N', 'L'])
            with self.assertRaisesRegex(RuntimeError, 'Please look up dimensions by name'):
                op(t, None)
            with self.assertRaisesRegex(RuntimeError, 'Name \'H\' not found'):
                op(t, 'H')

        def test_complete_reduce(op_name, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            op = getattr(torch.Tensor, op_name)
            self.assertEqual(op(t).names, [])

        def test_multidim_reduce(op_name, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            op = getattr(torch.Tensor, op_name)

            self.assertEqual(op(t, [1, 2]).names, ['N'])
            self.assertEqual(op(t, ['C', 'L']).names, ['N'])
            with self.assertRaisesRegex(RuntimeError, 'Please look up dimensions by name'):
                op(t, [None, 'C'])

        def test_out_variant(op_name, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            out = t.new_empty([0])
            getattr(torch, op_name)(t, 'C', out=out)
            self.assertEqual(out.names, ['N', 'L'])

        def test_keepdim(op_name, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            op = getattr(torch.Tensor, op_name)
            self.assertEqual(op(t, 'C', keepdim=True).names, ['N', 'C', 'L'])

        Case = namedtuple('Case', [
            'op_name',
            'supports_complete_reduce',
            'supports_multidim_reduce',
        ])

        tests = [
            Case(op_name='sum', supports_complete_reduce=True, supports_multidim_reduce=True),
            Case(op_name='prod', supports_complete_reduce=True, supports_multidim_reduce=False),
        ]

        for testcase, device in itertools.product(tests, torch.testing.get_all_device_types()):
            op_name = testcase.op_name
            test_simple_reduce(op_name, device)
            test_keepdim(op_name, device)
            test_out_variant(op_name, device)

            if testcase.supports_complete_reduce:
                test_complete_reduce(op_name, device)
            if testcase.supports_multidim_reduce:
                test_multidim_reduce(op_name, device)

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

    def test_align_to(self):
        def _test(tensor_namedshape, align_names, expected_sizes, expected_error):
            tensor_names, tensor_sizes = tensor_namedshape
            tensor = torch.empty(*tensor_sizes, names=tensor_names)
            if expected_error is not None:
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    tensor.align_to(align_names)
                return

            output = tensor.align_to(align_names)
            self.assertEqual(output.shape, expected_sizes)
            self.assertEqual(output.names, align_names)

        Case = namedtuple('Case', [
            'tensor_namedshape',
            'align_names',
            'expected_sizes',
            'expected_error',
        ])

        tests = [
            # basic tests
            Case(tensor_namedshape=(['C'], [2]),
                 align_names=['C'],
                 expected_sizes=[2],
                 expected_error=None),
            Case(tensor_namedshape=(['C'], [2]),
                 align_names=['D'],
                 expected_sizes=None,
                 expected_error='not a subsequence'),
            Case(tensor_namedshape=(['N', 'C'], [2, 3]),
                 align_names=['C'],
                 expected_sizes=None,
                 expected_error='shorter list of dims'),

            # single-dim alignment test
            Case(tensor_namedshape=(['C'], [2]),
                 align_names=['N', 'C'],
                 expected_sizes=[1, 2],
                 expected_error=None),
            Case(tensor_namedshape=[['N'], [2]],
                 align_names=['N', 'C'],
                 expected_sizes=[2, 1],
                 expected_error=None),

            # multiple dim alignment test
            Case(tensor_namedshape=[['N', 'C'], [2, 3]],
                 align_names=['N', 'H', 'C', 'W'],
                 expected_sizes=[2, 1, 3, 1],
                 expected_error=None),
            Case(tensor_namedshape=[['N', 'C'], [2, 3]],
                 align_names=['C', 'H', 'N', 'W'],
                 expected_sizes=None,
                 expected_error='not a subsequence'),

            # scalar tensor tests
            Case(tensor_namedshape=[None, [[]]],
                 align_names=['N', 'C'],
                 expected_sizes=[1, 1],
                 expected_error=None),
            Case(tensor_namedshape=[[], [[]]],
                 align_names=[None, None],
                 expected_sizes=[1, 1],
                 expected_error=None),

            # unnamed tensor tests
            Case(tensor_namedshape=[None, [2, 3]],
                 align_names=[None],
                 expected_sizes=None,
                 expected_error='shorter list'),
            Case(tensor_namedshape=[None, [2, 3]],
                 align_names=[None, None],
                 expected_sizes=[2, 3],
                 expected_error=None),
            Case(tensor_namedshape=[None, [2, 3]],
                 align_names=[None, None, None],
                 expected_sizes=[1, 2, 3],
                 expected_error=None),
            Case(tensor_namedshape=[None, [2]],
                 align_names=['N'],
                 expected_sizes=None,
                 expected_error='not a subsequence'),

            # unnamed dim alignment tests
            Case(tensor_namedshape=[[None], [2]],
                 align_names=['N', None],
                 expected_sizes=[1, 2],
                 expected_error=None),
            Case(tensor_namedshape=[[None], [2]],
                 align_names=['N', None, None, None],
                 expected_sizes=[1, 1, 1, 2],
                 expected_error=None),
            Case(tensor_namedshape=[['N'], [2]],
                 align_names=['N', None, None, None],
                 expected_sizes=[2, 1, 1, 1],
                 expected_error=None),
            Case(tensor_namedshape=[[None, 'N', None], [2, 3, 5]],
                 align_names=[None, None, 'N', None],
                 expected_sizes=[1, 2, 3, 5],
                 expected_error=None),
            Case(tensor_namedshape=[[None], [2]],
                 align_names=[None, 'N'],
                 expected_sizes=None,
                 expected_error='absolute position from the right'),
            Case(tensor_namedshape=[None, [2]],
                 align_names=[None, 'N'],
                 expected_sizes=None,
                 expected_error='absolute position from the right'),
            Case(tensor_namedshape=[[None, 'N'], [2, 3]],
                 align_names=[None, 'C', 'N'],
                 expected_sizes=None,
                 expected_error='absolute position from the right'),
        ]

        for test in tests:
            _test(*test)

    def test_align_tensors(self):
        # align_tensors shares code with align_to. test_align_to already tests
        # the alignment rules, so we don't do that again here.
        def reference_fn(*tensors):
            longest_names = tensors[0].names
            for tensor in tensors:
                if len(tensor.names) > len(longest_names):
                    longest_names = tensor.names
            return [tensor.align_to(longest_names) for tensor in tensors]

        x = torch.empty(1, 1, names=('N', 'H'))
        y = torch.empty(2, 3, 5, names=('N', 'C', 'H'))
        z = torch.empty(2, names=('N',))
        output = torch.align_tensors(x, y, z)
        expected_tensors = reference_fn(x, y, z)
        for tensor, expected in zip(output, expected_tensors):
            self.assertTensorDataAndNamesEqual(tensor, expected)

# Disable all tests if named tensor is not available.
for attr in dir(TestNamedTensor):
    if attr.startswith('test_'):
        new_test = skipIfNamedTensorDisabled(getattr(TestNamedTensor, attr))
        setattr(TestNamedTensor, attr, new_test)

if __name__ == '__main__':
    run_tests()
