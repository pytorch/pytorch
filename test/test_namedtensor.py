# Owner(s): ["module: named tensor"]
# ruff: noqa: F841
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_NUMPY
from torch.testing._internal.common_utils import skipIfTorchDynamo
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import get_all_device_types
from collections import namedtuple, OrderedDict
import itertools
import functools
import torch
from torch import Tensor
import torch.nn.functional as F
from multiprocessing.reduction import ForkingPickler
import pickle
import io
import sys
import warnings


def pass_name_to_python_arg_parser(name):
    x = torch.empty(2, names=(name,))


def flatten(lst):
    return [item for sublist in lst for item in sublist]


Function = namedtuple('TestCase', ['name', 'lambd'])


def parse_compressed_namedshape(string):
    # This is a metalanguage for describing a shape of a tensor compactly.
    # 'N:3,C:2' -> size = [3, 2], names: ['N', 'C']
    # 'None:3,None:2' -> size = [3, 2], names: ['None', 'None']
    # '3,2' -> size = [3, 2], names=None passed to ctor.
    def parse_name(maybe_name):
        maybe_name = maybe_name.strip()
        if maybe_name == 'None':
            return None
        return maybe_name

    string = string.strip()

    # '' -> size: [], names:None
    if len(string) == 0:
        return None, []

    # '3, 2' -> size = [3, 2], None names.
    if ':' not in string:
        return None, [int(size) for size in string.split(',')]

    dims = string.split(',')
    tuples = [dim.split(':') for dim in dims]
    return zip(*[(parse_name(name), int(size)) for name, size in tuples])


def create(namedshape, factory=torch.randn):
    # namedshape: str
    names, shape = parse_compressed_namedshape(namedshape)
    return factory(shape, names=names)


def out_fn(operator):
    @functools.wraps(operator)
    def fn(*inputs):
        return operator(*inputs[1:], out=inputs[0])
    return fn


class TestNamedTensor(TestCase):
    def test_aaa_must_run_first_check_experimental_warning(self):
        # TODO(rzou): It would be nice for this to be a "real" python warning.
        # Right now this error message only prints once and doesn't respect
        # warnings.simplefilter behavior (where python users can control whether
        # or not to display warnings once, all the time, or never).
        with warnings.catch_warnings(record=True) as warns:
            x = torch.randn(3, 3, names=('N', 'C'))
            self.assertEqual(len(warns), 1)
            self.assertTrue(str(warns[0].message).startswith(
                'Named tensors and all their associated APIs are an experimental feature'))

    def test_trivial(self):
        pass

    def _test_name_inference(self, op, args=(), expected_names=(), device='cpu',
                             maybe_raises_regex=None):
        casted_args = [arg.to(device) if isinstance(arg, torch.Tensor) else arg
                       for arg in args]
        if maybe_raises_regex is not None:
            with self.assertRaisesRegex(RuntimeError, maybe_raises_regex):
                result = op(*args)
            return
        result = op(*args)
        self.assertEqual(result.names, expected_names,
                         msg=f'Name inference for {op.__name__} on device {device} failed')

    # TODO(rzou): Some form of this check should be added to self.assertEqual.
    # Right now I don't know what it should look like.
    def assertTensorDataAndNamesEqual(self, x, y):
        self.assertEqual(x.names, y.names)
        unnamed_x = x.rename(None)
        unnamed_y = y.rename(None)
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

        x = factory(1, 2, 3, names=('_1', 'batch9', 'BATCH_5'), device=device)
        self.assertEqual(x.names, ('_1', 'batch9', 'BATCH_5'))

        with self.assertRaisesRegex(RuntimeError,
                                    'a valid identifier contains only'):
            x = factory(2, names=('1',), device=device)

        with self.assertRaisesRegex(RuntimeError,
                                    'a valid identifier contains only'):
            x = factory(2, names=('?',), device=device)

        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            x = factory(2, 1, names=('N',), device=device)

        with self.assertRaisesRegex(TypeError, 'invalid combination of arguments'):
            x = factory(2, 1, names='N', device=device)

        with self.assertRaisesRegex(RuntimeError, 'construct a tensor with duplicate names'):
            x = factory(2, 1, 1, names=('N', 'C', 'N'), device=device)

        names64 = ['A' * i for i in range(1, 65)]
        x = factory([1] * 64, names=names64, device=device)
        self.assertEqual(x.names, names64)

        with self.assertRaisesRegex(
                RuntimeError,
                'only support up to 64 dims'):
            names65 = ['A' * i for i in range(1, 66)]
            x = factory([1] * 65, names=names64, device=device)

    @skipIfTorchDynamo("not a bug: Dynamo causes the refcounts to be different")
    def test_none_names_refcount(self, N=10):
        def scope():
            unnamed = torch.empty(2, 3)
            unnamed.names  # materialize [None, None]

        prev_none_refcnt = sys.getrefcount(None)
        # Ran it N times to reduce flakiness
        [scope() for i in range(N)]
        after_none_refcnt = sys.getrefcount(None)
        self.assertTrue(after_none_refcnt - prev_none_refcnt < N / 2,
                        msg='Using tensor.names should not change '
                            'the refcount of Py_None')

    def test_has_names(self):
        unnamed = torch.empty(2, 3)
        none_named = torch.empty(2, 3, names=(None, None))
        partially_named = torch.empty(2, 3, names=('N', None))
        fully_named = torch.empty(2, 3, names=('N', 'C'))

        self.assertFalse(unnamed.has_names())
        self.assertFalse(none_named.has_names())
        self.assertTrue(partially_named.has_names())
        self.assertTrue(fully_named.has_names())

    def test_py3_ellipsis(self):
        tensor = torch.randn(2, 3, 5, 7)
        output = tensor.refine_names('N', ..., 'C')
        self.assertEqual(output.names, ['N', None, None, 'C'])

    def test_refine_names(self):
        # Unnamed tensor -> Unnamed tensor
        self._test_name_inference(Tensor.refine_names,
                                  [create('None:1,None:2,None:3'), 'N', 'C', 'H'],
                                  ['N', 'C', 'H'])

        # Named tensor -> Named tensor
        self._test_name_inference(Tensor.refine_names,
                                  [create('N:1,C:2,H:3'), 'N', 'C', 'H'],
                                  ['N', 'C', 'H'])

        # Partially named tensor -> named tensor
        self._test_name_inference(Tensor.refine_names,
                                  [create('None:1,C:2,None:3'), None, 'C', 'H'],
                                  [None, 'C', 'H'])

        # Too few names
        self._test_name_inference(Tensor.refine_names,
                                  [create('None:2,None:3'), 'N', 'C', 'H'],
                                  maybe_raises_regex="different number of dims")

        # Cannot change Tensor[D] to Tensor[N]
        self._test_name_inference(Tensor.refine_names,
                                  [create('D:3'), 'N'],
                                  maybe_raises_regex="is different from")

        # Cannot change Tensor[D] to Tensor[None]
        self._test_name_inference(Tensor.refine_names,
                                  [create('D:3'), None],
                                  maybe_raises_regex="'D' is more specific than None")

        # globbing behavior exists
        self._test_name_inference(Tensor.refine_names,
                                  [create('None:1,None:1,None:2,None:3'), '...', 'C', 'H'],
                                  [None, None, 'C', 'H'])

    def test_detach(self):
        names = ['N']
        self._test_name_inference(
            Tensor.detach_,
            [torch.randn(3, requires_grad=True, names=names)],
            names)
        self._test_name_inference(
            Tensor.detach,
            [torch.randn(3, requires_grad=True, names=names)],
            names)

    def test_index_fill(self):
        for device in get_all_device_types():
            expected_names = ('N', 'C')
            x = torch.randn(3, 5, device=device, names=expected_names)

            output = x.index_fill_('C', torch.tensor([0, 1], device=device), 5)
            self.assertEqual(output.names, expected_names)

            output = x.index_fill_('C', torch.tensor([0, 1], device=device), torch.tensor(4.))
            self.assertEqual(output.names, expected_names)

            output = x.index_fill('C', torch.tensor([0, 1], device=device), 5)
            self.assertEqual(output.names, expected_names)

            output = x.index_fill('C', torch.tensor([0, 1], device=device), torch.tensor(4.))
            self.assertEqual(output.names, expected_names)

    def test_equal(self):
        for device in get_all_device_types():
            tensor = torch.randn(2, 3, device=device)
            other = tensor.clone()

            self.assertTrue(torch.equal(tensor.rename('N', 'C'), other.rename('N', 'C')))
            self.assertFalse(torch.equal(tensor.rename('M', 'C'), other.rename('N', 'C')))
            self.assertFalse(torch.equal(tensor.rename(None, 'C'), other.rename('N', 'C')))

    def test_squeeze(self):
        x = create('N:3,C:1,H:1,W:1')
        output = x.squeeze('C')
        self.assertEqual(output.names, ['N', 'H', 'W'])

        output = x.squeeze()
        self.assertEqual(output.names, ['N'])

    def test_repr(self):
        named_tensor = torch.zeros(2, 3).rename_('N', 'C')
        expected = "tensor([[0., 0., 0.],\n        [0., 0., 0.]], names=('N', 'C'))"
        self.assertEqual(repr(named_tensor), expected)

        unnamed_tensor = torch.zeros(2, 3)
        expected = "tensor([[0., 0., 0.],\n        [0., 0., 0.]])"
        self.assertEqual(repr(unnamed_tensor), expected)

        none_named_tensor = torch.zeros(2, 3).rename_(None, None)
        self.assertEqual(repr(none_named_tensor), expected)

    def test_diagonal(self):
        named_tensor = torch.zeros(2, 3, 5, 7, names=list('ABCD'))
        self.assertEqual(named_tensor.diagonal().names, ['C', 'D', None])
        self.assertEqual(named_tensor.diagonal(1, 3).names, ['A', 'C', None])

        self.assertEqual(named_tensor.diagonal(outdim='E', dim1='B', dim2='D').names,
                         ['A', 'C', 'E'])

    def test_max_pooling(self):
        def check_tuple_return(op, inputs, expected_names):
            values, indices = op(*inputs)
            self.assertEqual(values.names, expected_names)
            self.assertEqual(indices.names, expected_names)

        for device in get_all_device_types():

            named_tensor_1d = torch.zeros(2, 3, 5, device=device, names=list('ABC'))
            named_tensor_2d = torch.zeros(2, 3, 5, 7, device=device, names=list('ABCD'))
            named_tensor_3d = torch.zeros(2, 3, 5, 7, 9, device=device, names=list('ABCDE'))

            self.assertEqual(F.max_pool1d(named_tensor_1d, 2).names, named_tensor_1d.names)
            self.assertEqual(F.max_pool2d(named_tensor_2d, [2, 2]).names, named_tensor_2d.names)
            self.assertEqual(F.max_pool3d(named_tensor_3d, [2, 2, 2]).names, named_tensor_3d.names)

            check_tuple_return(F.max_pool1d_with_indices, [named_tensor_1d, 2], named_tensor_1d.names)
            check_tuple_return(F.max_pool2d_with_indices, [named_tensor_2d, [2, 2]], named_tensor_2d.names)
            check_tuple_return(F.max_pool3d_with_indices, [named_tensor_3d, [2, 2, 2]], named_tensor_3d.names)

    def test_max_pooling_without_names_does_not_warn(self):
        for device in get_all_device_types():
            tensor_2d = torch.zeros(2, 3, 5, 7, device=device, requires_grad=True)
            with warnings.catch_warnings(record=True) as warns:
                warnings.simplefilter("always")
                result = F.max_pool2d(tensor_2d, [2, 2])
                result.sum().backward()
                self.assertEqual(len(warns), 0)

    def test_no_save_support(self):
        named_tensor = torch.zeros(2, 3, names=('N', 'C'))
        buf = io.BytesIO()
        with self.assertRaisesRegex(RuntimeError, "NYI"):
            torch.save(named_tensor, buf)

    def test_no_pickle_support(self):
        named_tensor = torch.zeros(2, 3, names=('N', 'C'))
        with self.assertRaisesRegex(RuntimeError, "NYI"):
            serialized = pickle.dumps(named_tensor)

    def test_no_multiprocessing_support(self):
        named_tensor = torch.zeros(2, 3, names=('N', 'C'))
        buf = io.BytesIO()
        with self.assertRaisesRegex(RuntimeError, "NYI"):
            ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(named_tensor)

    def test_big_tensor_repr_has_names(self):
        def check_repr(named_tensor):
            unnamed_tensor = named_tensor.rename(None)
            names_tag = f'names={named_tensor.names}'
            self.assertIn(names_tag, repr(named_tensor))

        check_repr(torch.randn(128, 3, 64, 64, names=('N', 'C', 'H', 'W')))

    def test_noncontig_contiguous(self):
        # This type of contiguous is special-cased and therefore needs its own test
        for device in get_all_device_types():
            x = torch.randn(2, 3, device=device).t().rename_('N', 'C')
            self.assertEqual(x.contiguous().names, ('N', 'C'))

    def test_copy_transpose(self):
        # This type of copy is special-cased and therefore needs its own test
        def _test(self_names, other_names, expected_names):
            x = torch.empty(2, 5, names=self_names)
            y = torch.empty(5, 2).t().rename_(*other_names)
            x.copy_(y)
            self.assertEqual(x.names, expected_names)

        _test(('N', 'C'), ('N', 'C'), ('N', 'C'))
        _test(None, ('N', 'C'), ('N', 'C'))

    def test_rename_(self):
        tensor = torch.empty(1, 1, names=('N', 'C'))
        self.assertEqual(tensor.rename_(None).names, (None, None))
        self.assertEqual(tensor.rename_('H', 'W').names, ('H', 'W'))
        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.rename_('N', 'C', 'W')
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.rename_('N', 'N')

    def test_rename(self):
        tensor = torch.empty(1, 1, names=('N', 'C'))

        self.assertEqual(tensor.rename(None).names, (None, None))
        self.assertEqual(tensor.rename('H', 'W').names, ('H', 'W'))

        # Check that we didn't modify tensor.names
        self.assertEqual(tensor.names, ('N', 'C'))

        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.rename('N', 'C', 'W')
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.rename('N', 'N')

        with self.assertRaisesRegex(RuntimeError, 'either positional args or keyword args'):
            tensor.rename(None, N='batch')

        # rename returns a view on the tensor
        self.assertEqual(tensor.rename('H', 'W').data_ptr(), tensor.data_ptr())
        self.assertEqual(tensor.rename(None).data_ptr(), tensor.data_ptr())

    def test_rename_globber(self):
        scalar = torch.randn([])
        unnamed_tensor = torch.empty(1, 1, 1, 1)
        named_tensor = torch.empty(1, 1, 1, 1, names=('N', 'C', 'H', 'W'))

        self.assertEqual(scalar.rename(None).names, [])
        self.assertEqual(scalar.rename('...').names, [])

        # Check that it works with unnamed tensors
        self.assertEqual(unnamed_tensor.rename('...').names, unnamed_tensor.names)
        self.assertEqual(unnamed_tensor.rename('...', 'H', 'W').names,
                         [None, None, 'H', 'W'])
        self.assertEqual(unnamed_tensor.rename('N', '...', 'W').names,
                         ['N', None, None, 'W'])
        self.assertEqual(unnamed_tensor.rename('N', 'C', '...').names,
                         ['N', 'C', None, None])

        # Check that it works with named tensors
        self.assertEqual(named_tensor.rename('...').names, named_tensor.names)
        self.assertEqual(named_tensor.rename('...', 'width').names,
                         ['N', 'C', 'H', 'width'])
        self.assertEqual(named_tensor.rename('batch', 'channels', '...', 'width').names,
                         ['batch', 'channels', 'H', 'width'])
        self.assertEqual(named_tensor.rename('batch', '...').names,
                         ['batch', 'C', 'H', 'W'])

        # Test empty glob
        self.assertEqual(unnamed_tensor.rename('...', None, None, None, None).names,
                         [None, None, None, None])
        self.assertEqual(named_tensor.rename('N', 'C', 'H', '...', 'W').names,
                         ['N', 'C', 'H', 'W'])

        # Multiple globs throw
        with self.assertRaisesRegex(RuntimeError, 'More than one '):
            named_tensor.rename('...', 'channels', '...')

    def test_rename_rename_map(self):
        scalar = torch.randn([])
        unnamed_tensor = torch.empty(1, 1, 1, 1)
        named_tensor = torch.empty(1, 1, 1, 1, names=('N', 'C', 'H', 'W'))

        with self.assertRaisesRegex(RuntimeError, "dim 'N' does not exist"):
            scalar.rename(N='batch')
        with self.assertRaisesRegex(RuntimeError, "dim 'N' does not exist"):
            unnamed_tensor.rename(N='batch')
        with self.assertRaisesRegex(RuntimeError, "dim 'B' does not exist"):
            named_tensor.rename(B='batch')
        with self.assertRaisesRegex(RuntimeError, "dim 'B' does not exist"):
            named_tensor.rename(H='height', B='batch')

        self.assertEqual(named_tensor.rename(N='batch').data_ptr(),
                         named_tensor.data_ptr())
        self.assertEqual(named_tensor.rename(N='batch').names,
                         ['batch', 'C', 'H', 'W'])
        self.assertEqual(named_tensor.rename(N='batch', H='height').names,
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
        for device in get_all_device_types():
            self._test_factory(torch.empty, device)

    def test_factory_coverage(self):
        def _test(factory, device):
            names = ('N', 'T', 'D')

            torch.manual_seed(0)
            result = factory(1, 2, 3, names=names, device=device)

            torch.manual_seed(0)
            expected = factory(1, 2, 3, device=device).rename_(*names)

            self.assertTensorDataAndNamesEqual(result, expected)

        supported = [
            torch.ones,
            torch.rand,
            torch.randn,
            torch.zeros,
        ]

        for op, device in itertools.product(supported, get_all_device_types()):
            _test(op, device)

        # Test torch.full
        for device in get_all_device_types():
            names = ('N', 'T', 'D')
            result = torch.full([1, 2, 3], 2., names=names, device=device)
            expected = torch.full([1, 2, 3], 2., device=device).rename_(*names)
            self.assertTensorDataAndNamesEqual(result, expected)

    def test_tensor_from_lists(self):
        names = ('N', 'C')
        tensor = torch.tensor([[1]], names=names)
        self.assertEqual(tensor.names, names)

        names = ('N',)
        tensor = torch.tensor([1], names=names)
        self.assertEqual(tensor.names, names)

        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            names = ('N', 'C')
            tensor = torch.tensor([1], names=names)

    @unittest.skipIf(not TEST_NUMPY, "no numpy")
    def test_tensor_from_numpy(self):
        import numpy as np
        arr = np.array([[1]])
        names = ('N', 'C')
        tensor = torch.tensor([[1]], names=names)
        self.assertEqual(tensor.names, names)

    def test_tensor_from_tensor(self):
        x = torch.randn(1, 1)
        names = ('N', 'C')
        tensor = torch.tensor(x, names=names)
        self.assertEqual(tensor.names, names)

    def test_tensor_from_named_tensor(self):
        x = torch.randn(1, 1, names=('N', 'D'))
        tensor = torch.tensor(x)
        self.assertEqual(tensor.names, ('N', 'D'))

        # there's no way to distinguish between names=None and not passing in names.
        # If the user passes in names=None they are asking for trouble.
        x = torch.randn(1, 1, names=('N', 'D'))
        tensor = torch.tensor(x, names=None)
        self.assertEqual(tensor.names, ('N', 'D'))

        x = torch.randn(1, 1, names=('N', 'D'))
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            tensor = torch.tensor(x, names=('N', 'C'))

    def test_size(self):
        t = torch.empty(2, 3, 5, names=('N', None, 'C'))
        self.assertEqual(t.size('N'), 2)
        self.assertEqual(t.size('C'), 5)
        with self.assertRaisesRegex(RuntimeError, 'Name \'channels\' not found in '):
            t.size('channels')
        with self.assertRaisesRegex(RuntimeError, 'Name \'N\' not found in '):
            torch.empty(2, 3, 4).size('N')

    def test_stride(self):
        t = torch.empty(2, 3, 5, names=('N', None, 'C'))
        self.assertEqual(t.stride('N'), 3 * 5)
        self.assertEqual(t.stride('C'), 1)
        with self.assertRaisesRegex(RuntimeError, 'Name \'channels\' not found in '):
            t.stride('channels')
        with self.assertRaisesRegex(RuntimeError, 'Name \'N\' not found in '):
            torch.empty(2, 3, 4).stride('N')

    def test_transpose_variants(self):
        t = torch.randn(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
        self.assertEqual(t.transpose('N', 'C').names, ['C', 'N', 'H', 'W'])
        self.assertEqual(t.transpose(1, 3).names, ['N', 'W', 'H', 'C'])

        t = torch.randn(2, 3, names=('N', 'C'))
        self.assertEqual(t.t().names, ['C', 'N'])

    def test_resize(self):
        for device in get_all_device_types():
            named = torch.randn(2, names=('N',), device=device)
            named.resize_([2])
            self.assertEqual(named.names, ['N'])

            with self.assertRaisesRegex(RuntimeError, "Cannot resize named tensor"):
                named.resize_([3])

            other_named = torch.randn(2, names=('N',), device=device)
            named.resize_as_(other_named)
            self.assertEqual(other_named.names, ['N'])

            unnamed = torch.randn(2, device=device)
            with self.assertRaisesRegex(
                    RuntimeError, r'names .* are not the same as the computed output names'):
                named.resize_as_(unnamed)

            unnamed = torch.randn(1, device=device)
            unnamed.resize_as_(named)
            self.assertEqual(unnamed.names, ['N'])

    def test_cdist(self):
        for device in get_all_device_types():
            tensor = torch.randn(3, 1, 2, 7, names=('M', 'N', 'first_group', 'features'),
                                 device=device)
            other = torch.randn(5, 11, 7, names=('N', 'second_group', 'features'),
                                device=device)
            result = torch.cdist(tensor, other)
            self.assertEqual(result.names, ['M', 'N', 'first_group', 'second_group'])

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
        tensor.is_shared()
        tensor.is_signed()

    def test_autograd_smoke(self):
        x = torch.randn(3, 3, names=('N', 'D'), requires_grad=True)

        y = x.clone()
        y.retain_grad()
        y.register_hook(lambda x: x)

        y.sum().backward()

        # autograd related attributes
        tensor = torch.empty(1, 1, names=('N', 'D'), requires_grad=True)
        tensor = tensor.relu()
        tensor.output_nr
        tensor.grad_fn
        tensor.requires_grad

    def test_split_fns_propagates_names(self):
        fns = [
            lambda x: x.split(1, 0),
            lambda x: x.split([1, 1], 1),
            lambda x: x.chunk(2, 0),
        ]

        for device in get_all_device_types():
            orig_tensor = torch.empty(2, 2, names=('N', 'D'), device=device)
            for fn in fns:
                splits = fn(orig_tensor)
                for split in splits:
                    self.assertEqual(split.names, orig_tensor.names)

    def test_any_all(self):
        for device in get_all_device_types():
            x = torch.zeros(3, dtype=torch.bool, device=device, names=('C',))
            self.assertEqual(x.any().names, [])
            self.assertEqual(x.all().names, [])

    def test_addcmul_addcdiv(self):
        for device in get_all_device_types():
            names = ['N']
            a = torch.rand(3, device=device, names=names)
            b = torch.rand(3, device=device, names=names)
            # avoid division by 0
            c = torch.rand(3, device=device, names=names).clamp_min_(0.1)
            out = torch.randn(3, device=device, names=names)

            self.assertEqual(torch.addcmul(a, b, c).names, names)
            self.assertEqual(torch.addcmul(a, b, c, out=out).names, names)
            self.assertEqual(a.addcmul_(b, c).names, names)

            self.assertEqual(torch.addcdiv(a, b, c).names, names)
            self.assertEqual(torch.addcdiv(a, b, c, out=out).names, names)
            self.assertEqual(a.addcdiv_(b, c).names, names)

    def test_binary_ops(self):
        def test_basic(op):
            a = torch.empty(2, 3, names=('N', 'C'))
            b = torch.empty(3, 2, names=('C', 'N'))
            c = torch.empty(3, names=('C',))
            d = torch.empty(5, names=('W',))

            self.assertEqual(op(a, a).names, ('N', 'C'))
            self.assertEqual(op(a, c).names, ('N', 'C'))
            # TODO: dynamo will throw a slightly different
            # error message because it's adding fake tensors
            # `must match the size of` portion is the dynamo error
            with self.assertRaisesRegex(RuntimeError, "do not match|must match the size of"):
                op(a, d)
            with self.assertRaisesRegex(RuntimeError, "do not match|must match the size of"):
                op(a, b)

        def test_wildcard(op):
            a = torch.empty(2, 3, names=('N', 'C'))
            c = torch.empty(2, 3, names=(None, 'C'))
            self.assertEqual(op(a, c).names, ('N', 'C'))

            b = torch.empty(2, 3)
            self.assertEqual(op(a, b).names, ('N', 'C'))

            d = torch.empty(2, 3, names=('C', None))
            with self.assertRaisesRegex(RuntimeError, "Misaligned"):
                op(d, c)

        def test_mixed_unnamed_named(op, is_inplace):
            named2 = torch.randn(1, 1, names=('N', 'C'))
            unnamed1 = torch.randn(1)
            unnamed2 = torch.randn(1, 1)
            unnamed3 = torch.randn(1, 1, 1)

            def compute_expected_names(tensor, other):
                assert tensor.has_names() ^ other.has_names()
                named = tensor if tensor.has_names() else other
                unnamed = other if tensor.has_names() else tensor
                unnamed_dim = unnamed.dim()
                if unnamed_dim > named.dim():
                    return [None] * (unnamed_dim - named.dim()) + list(named.names)
                else:
                    return named.names

            inputs = itertools.chain(
                itertools.product([named2], [unnamed1, unnamed2, unnamed3]),
                itertools.product([unnamed1, unnamed2, unnamed3], [named2]),
            )
            if is_inplace:
                # In-place ops have the constraint that they must not change shape.
                inputs = [(a, b) for (a, b) in inputs if a.dim() >= b.dim()]

            for tensor, other in inputs:
                expected_names = compute_expected_names(tensor, other)
                self.assertEqual(op(tensor, other).names, expected_names)

        def method(name, *args, **kwargs):
            return [Function(name, lambda a, b: getattr(a, name)(b, *args, **kwargs))]

        def function(name, *args, **kwargs):
            return [Function(name, lambda a, b: getattr(torch, name)(a, b, *args, **kwargs))]

        def out_function(name, *args, **kwargs):
            out_fn = getattr(torch, name)

            def fn(a, b):
                result = torch.empty([0], dtype=a.dtype, device=a.device)
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
            fn_method_and_inplace('pow'),
            fn_method_and_inplace('atan2'),
            method('copy_'),
            function('floor_divide'),
            function('true_divide'),
        ]
        tests = flatten(tests)

        for name, op in tests:
            test_basic(op)
            test_wildcard(op)
            test_mixed_unnamed_named(op, is_inplace=name.endswith('_'))

    def test_logical_ops(self):
        # Implemented via TensorIterator, so just check that each version
        # (out-of-place, inplace, out=) propagates names.
        def zeros(*args, **kwargs):
            return torch.zeros(*args, dtype=torch.bool, **kwargs)

        for op in ('logical_xor', 'logical_and', 'logical_or'):
            self._test_name_inference(
                getattr(torch, op),
                (create('N:2,C:3', zeros), create('N:2,C:3', zeros)),
                expected_names=['N', 'C'])

            self._test_name_inference(
                getattr(Tensor, op + '_'),
                (create('N:2,C:3', zeros), create('N:2,C:3', zeros)),
                expected_names=['N', 'C'])

            self._test_name_inference(
                lambda out, x, y: getattr(torch, op)(x, y, out=out),
                (create('0', zeros), create('N:2,C:3', zeros), create('N:2,C:3', zeros)),
                expected_names=['N', 'C'])

    def test_pow_special(self):
        # There are a few pow cases that don't go through TensorIterator.
        # Test them here.
        for device in get_all_device_types():
            named = torch.randn(2, 3, names=('N', 'C'), device=device)
            unnamed = torch.randn([0], device=device)

            result = torch.pow(named, 0, out=unnamed.clone())
            self.assertEqual(result.names, named.names)

            result = torch.pow(named, 1, out=unnamed.clone())
            self.assertEqual(result.names, named.names)

            result = torch.pow(1, named, out=unnamed.clone())
            self.assertEqual(result.names, named.names)

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
            try:
                out = testcase.lambd(tensor)
            except RuntimeError as err:
                # Get a better error message by catching the error and asserting.
                raise RuntimeError(f'{testcase.name}: {err}') from err
            self.assertEqual(out.names, tensor.names,
                             msg=testcase.name)

        def fn(name, *args, **kwargs):
            return [Function(name, lambda t: getattr(torch, name)(t, *args, **kwargs))]

        def method(name, *args, **kwargs):
            return [Function(name, lambda t: getattr(t, name)(*args, **kwargs))]

        def out_function(name, *args, **kwargs):
            out_fn = getattr(torch, name)

            def fn(tensor):
                result = torch.empty([0], dtype=tensor.dtype, device=tensor.device)
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
            method('relu_'),
            method('requires_grad_'),
            method('relu'),
            fn_method_and_inplace('round'),
            fn_method_and_inplace('rsqrt'),
            fn_method_and_inplace('sigmoid'),
            fn_method_and_inplace('sign'),
            fn_method_and_inplace('sin'),
            fn_method_and_inplace('sinh'),
            fn_method_and_inplace('sqrt'),
            fn_method_and_inplace('tan'),
            fn_method_and_inplace('tanh'),
            fn('threshold', 0, 1),
            fn('threshold_', 0, 1),
            out_function('threshold', 0, 1),
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

            # cumsum and cumprod
            fn('cumsum', 0),
            fn('cumsum', 'D'),
            out_function('cumsum', 'D'),
            fn('cumprod', 0),
            fn('cumprod', 'D'),
            out_function('cumprod', 'D'),

            # views
            method('narrow', 0, 0, 1),

            # creation functions
            fn('empty_like'),
            fn('zeros_like'),
            fn('ones_like'),
            fn('full_like', 3.14),
            fn('rand_like'),
            fn('randn_like'),

            # bernoulli variants
            method('bernoulli_', 0.5),
            method('bernoulli_', torch.tensor(0.5)),

            method('softmax', dim=1),
            method('softmax', dim='D'),
            method('log_softmax', dim=1),
            method('log_softmax', dim='D'),

            [Function('F.dropout(inplace)', lambda t: F.dropout(t, p=0.5, inplace=True))],
            [Function('F.dropout(outplace)', lambda t: F.dropout(t, p=0.5, inplace=False))],
        ]
        tests = flatten(tests)

        for testcase, device in itertools.product(tests, get_all_device_types()):
            _test(testcase, device=device)

    def test_cummax_cummin(self):
        def test_ops(op):
            for device in get_all_device_types():
                names = ('N', 'D')
                tensor = torch.rand(2, 3, names=names)
                result = op(tensor, 0)
                self.assertEqual(result[0].names, names)
                self.assertEqual(result[1].names, names)
        test_ops(torch.cummax)
        test_ops(torch.cummin)

    def test_logcumsumexp(self):
        for device in get_all_device_types():
            names = ('N', 'D')
            tensor = torch.rand(2, 3, names=names)
            result = torch.logcumsumexp(tensor, 'D')
            self.assertEqual(result.names, names)

    def test_bitwise_not(self):
        for device in get_all_device_types():
            names = ('N', 'D')
            tensor = torch.zeros(2, 3, names=names, dtype=torch.bool)
            result = torch.empty(0, dtype=torch.bool)

            self.assertEqual(tensor.bitwise_not().names, names)
            self.assertEqual(torch.bitwise_not(tensor, out=result).names, names)
            self.assertEqual(tensor.bitwise_not_().names, names)

    def test_logical_not(self):
        for device in get_all_device_types():
            names = ('N', 'D')
            tensor = torch.zeros(2, 3, names=names, dtype=torch.bool)
            result = torch.empty(0, dtype=torch.bool)

            self.assertEqual(tensor.logical_not().names, names)
            self.assertEqual(torch.logical_not(tensor, out=result).names, names)
            self.assertEqual(tensor.logical_not_().names, names)

    def test_bernoulli(self):
        for device in get_all_device_types():
            names = ('N', 'D')
            tensor = torch.rand(2, 3, names=names)
            result = torch.empty(0)
            self.assertEqual(tensor.bernoulli().names, names)

            torch.bernoulli(tensor, out=result)
            self.assertEqual(result.names, names)

    def test_flatten(self):
        tensor = torch.randn(2, 3, 5, 7, 11, names=('N', 'C', 'D', 'H', 'W'))

        # basic
        out = tensor.flatten('D', 'W', 'features')
        self.assertEqual(out.names, ['N', 'C', 'features'])
        self.assertEqual(out.rename(None), tensor.rename(None).view(2, 3, -1))

        # int overload
        out = tensor.flatten(2, 4, 'features')
        self.assertEqual(out.names, ['N', 'C', 'features'])
        self.assertEqual(out.rename(None), tensor.rename(None).view(2, 3, -1))

        # list overload
        out = tensor.flatten(['D', 'H', 'W'], 'features')
        self.assertEqual(out.names, ['N', 'C', 'features'])
        self.assertEqual(out.rename(None), tensor.rename(None).view(2, 3, -1))

        # Non-contiguous flatten: N and H are not "adjacent" in memory.
        sentences = torch.randn(2, 3, 5, 7, names=('N', 'T', 'H', 'D'))
        sentences = sentences.transpose('T', 'H')
        out = sentences.flatten('N', 'H', 'N_H')
        self.assertEqual(out.names, ['N_H', 'T', 'D'])

        with self.assertRaisesRegex(RuntimeError, "Name 'L' not found in"):
            tensor.flatten(['D', 'L'], 'features')

        with self.assertRaisesRegex(RuntimeError, "must be consecutive in"):
            tensor.flatten(['D', 'W'], 'features')

        with self.assertRaisesRegex(RuntimeError, "must be consecutive in"):
            tensor.flatten(['H', 'D', 'W'], 'features')

    def test_flatten_nodims(self):
        tensor = torch.empty((2, 3))
        with self.assertRaisesRegex(RuntimeError, "cannot be empty"):
            tensor.flatten((), 'abcd')

    def test_flatten_index_error(self):
        tensor = torch.randn(1, 2)
        with self.assertRaisesRegex(IndexError,
                                    r"Dimension out of range \(expected to be in range of \[-2, 1\], but got 2\)"):
            tensor.flatten(0, 2)
        with self.assertRaisesRegex(IndexError,
                                    r"Dimension out of range \(expected to be in range of \[-2, 1\], but got 2\)"):
            tensor.flatten(0, 2, 'N')
        with self.assertRaisesRegex(RuntimeError,
                                    r"flatten\(\) has invalid args: start_dim cannot come after end_dim"):
            tensor.flatten(1, 0)
        with self.assertRaisesRegex(RuntimeError,
                                    r"flatten\(\) has invalid args: start_dim cannot come after end_dim"):
            tensor.flatten(1, 0, 'N')

    def test_unflatten(self):
        # test args: tensor, int, namedshape
        self.assertTrue(torch.equal(
            torch.ones(4, names=('A',)).unflatten('A', (('A', 2), ('B', 2))),
            torch.ones(2, 2, names=('A', 'B'))))
        self.assertTrue(torch.equal(
            torch.ones(4, names=('A',)).unflatten('A', [('A', 2), ('B', 2)]),
            torch.ones(2, 2, names=('A', 'B'))))
        self.assertTrue(torch.equal(
            torch.ones(4, names=('A',)).unflatten('A', (['A', 2], ['B', 2])),
            torch.ones(2, 2, names=('A', 'B'))))
        self.assertTrue(torch.equal(
            torch.ones(2, 10, names=('A', 'B')).unflatten('B', (['B1', -1],)),
            torch.ones(2, 10, names=('A', 'B1'))))
        self.assertTrue(torch.equal(
            torch.ones(2, 3 * 4 * 5 * 6, names=('A', 'B'))
                 .unflatten('B', (['B1', 3], ['B2', 4], ['B3', -1], ['B4', 6])),
            torch.ones(2, 3, 4, 5, 6, names=('A', 'B1', 'B2', 'B3', 'B4'))))
        self.assertTrue(torch.equal(
            torch.ones(2, 0, names=('A', 'B'))
                 .unflatten('B', (['B1', 3], ['B2', -1], ['B3', 4])),
            torch.ones(2, 3, 0, 4, names=('A', 'B1', 'B2', 'B3'))))

        # test args: namedtensor, str, namedshape
        self.assertTrue(torch.equal(
            torch.ones(2, 4, names=('A', 'B')).unflatten('B', (('B1', 2), ('B2', 2))),
            torch.ones(2, 2, 2, names=('A', 'B1', 'B2'))))

        # test invalid args: namedtensor, str, sizes
        with self.assertRaisesRegex(TypeError, r"unflatten\(\): argument 'dim' \(position 1\) must be int, not str"):
            torch.tensor([1], names=('A',)).unflatten('A', (1, 1))

        # test invalid args: namedtensor, int, sizes
        with self.assertRaisesRegex(RuntimeError, r"input is a named tensor but no names were given for unflattened sizes"):
            torch.tensor([1], names=("A",)).unflatten(0, (1, 1))

        with self.assertRaisesRegex(RuntimeError,
                                    r"Provided sizes \[3, -1\] don't multiply up to the "
                                    r"size of dim 1 \('B': 4\) in Tensor\['A', 'B'\]"):
            torch.ones(2, 4, names=('A', 'B')).unflatten('B', (('B1', 3), ('B2', -1)))

        with self.assertRaisesRegex(RuntimeError,
                                    r"the unspecified dimension size -1 can be any value and is ambiguous"):
            torch.ones(2, 0, names=('A', 'B')).unflatten('B', (('B1', 0), ('B2', -1)))

        tensor = torch.randn(7, 2 * 3 * 5, 11, names=('N', 'D', 'K'))

        # accepts OrderedDict
        out = tensor.unflatten('D', OrderedDict((('C', 2), ('H', 3), ('W', 5))))
        self.assertEqual(out.names, ('N', 'C', 'H', 'W', 'K'))
        self.assertEqual(out.shape, (7, 2, 3, 5, 11))

        # Unflatten left-most
        out = tensor.unflatten('N', (('N', 7), ('H', 1)))
        self.assertEqual(out.names, ('N', 'H', 'D', 'K'))
        self.assertEqual(out.shape, (7, 1, 2 * 3 * 5, 11))

        # Unflatten right-most
        out = tensor.unflatten('K', (('K', 11), ('H', 1)))
        self.assertEqual(out.names, ('N', 'D', 'K', 'H'))
        self.assertEqual(out.shape, (7, 2 * 3 * 5, 11, 1))

        with self.assertRaisesRegex(RuntimeError, "don't multiply up to"):
            tensor.unflatten('D', (('H', 3), ('W', 5)))

        with self.assertRaisesRegex(RuntimeError, 'sizes must be non-empty'):
            tensor.unflatten('D', None)

        with self.assertRaisesRegex(RuntimeError, 'non-empty'):
            tensor.unflatten('D', OrderedDict())

    def test_unsupported_op_error_msg(self):
        named = torch.randn(3, 3, names=('N', 'C'))
        with self.assertRaisesRegex(
                RuntimeError, r"pdist.+is not yet supported with named tensors"):
            torch.pdist(named)
        with self.assertRaisesRegex(
                RuntimeError, r"as_strided_.+is not yet supported with named tensors"):
            named.as_strided_((3, 3), (3, 1))

    def test_reduction_fns(self):
        def check_output(output, expected_names):
            if isinstance(output, torch.Tensor):
                self.assertEqual(output.names, expected_names)
                return
            for out in output:
                self.assertEqual(out.names, expected_names)

        def sum_all_outputs(output):
            if isinstance(output, torch.Tensor):
                return output.sum()
            result = 0
            for out in output:
                result = out + result
            return result.sum()

        def test_simple_reduce(op, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            check_output(op(t, 1), ['N', 'L'])
            check_output(op(t, -1), ['N', 'C'])
            check_output(op(t, 'C'), ['N', 'L'])
            ops_support_dim_none = [
                'sum',
                'mean',
                'std',
                'var',
                'std_mean',
                'var_mean',
                'nanmean',
                'nansum',
            ]
            if op.__name__ in ops_support_dim_none:
                check_output(op(t, None), [])
            else:
                with self.assertRaisesRegex(RuntimeError, 'Please look up dimensions by name'):
                    op(t, None)
            with self.assertRaisesRegex(RuntimeError, 'Name \'H\' not found'):
                op(t, 'H')

        def test_autograd_supports_dimname_overload(op, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device, requires_grad=True)
            sum_all_outputs(op(t, 'C')).backward()
            self.assertIsNotNone(t.grad)

        def test_complete_reduce(op, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            check_output(op(t), [])

        def test_multidim_reduce(op, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)

            check_output(op(t, [1, 2]), ['N'])
            check_output(op(t, [0, -1]), ['C'])
            check_output(op(t, ['C', 'L']), ['N'])
            with self.assertRaisesRegex(RuntimeError, 'Please look up dimensions by name'):
                op(t, [None, 'C'])

        def test_out_variant(op, output_lambda, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            if output_lambda:
                out = output_lambda(t)
            else:
                out = torch.empty([0], device=device)
            op(t, 'C', out=out)
            check_output(out, ['N', 'L'])

        def test_keepdim(op, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            check_output(op(t, 'C', keepdim=True), ['N', 'C', 'L'])

        def values_and_indices(t):
            return (torch.empty([0], device=t.device),
                    torch.empty([0], device=t.device, dtype=torch.long))

        def kthvalue_wrapper(tensor, *args, **kwargs):
            # Return the 0-th value
            return torch.kthvalue(tensor, 1, *args, **kwargs)

        Case = namedtuple('Case', [
            'op',
            'supports_complete_reduce',
            'supports_multidim_reduce',
            'supports_out_variant',
            'supports_keepdim',
            'output_lambda',
        ])

        tests = [
            Case(torch.sum, True, True, True, True, None),
            Case(torch.prod, True, False, True, True, None),
            Case(torch.mean, True, True, True, True, None),
            Case(torch.var, True, True, True, True, None),
            Case(torch.std, True, True, True, True, None),
            Case(torch.std_mean, True, True, False, True, None),
            Case(torch.var_mean, True, True, False, True, None),
            Case(torch.min, True, False, True, True, values_and_indices),
            Case(torch.max, True, False, True, True, values_and_indices),
            Case(torch.unbind, False, False, False, False, None),
            Case(torch.logsumexp, False, True, True, True, None),
            Case(torch.mode, False, False, True, True, values_and_indices),
            Case(kthvalue_wrapper, False, False, True, True, values_and_indices),
            Case(torch.median, True, False, True, True, values_and_indices),
            Case(torch.nanmedian, True, False, True, True, values_and_indices),
        ]

        for testcase, device in itertools.product(tests, get_all_device_types()):
            op = testcase.op
            test_simple_reduce(op, device)
            test_autograd_supports_dimname_overload(op, device)

            if testcase.supports_keepdim:
                test_keepdim(op, device)
            if testcase.supports_out_variant:
                test_out_variant(op, testcase.output_lambda, device)
            if testcase.supports_complete_reduce:
                test_complete_reduce(op, device)
            if testcase.supports_multidim_reduce:
                test_multidim_reduce(op, device)

    def test_masked_select(self):
        # simple
        self._test_name_inference(
            torch.masked_select,
            (create('N:2,C:3'), (create('2,3') > 0).rename('N', 'C')),
            expected_names=[None])

        # left broadcast
        self._test_name_inference(
            torch.masked_select,
            (create('C:3'), (create('2,3') > 0).rename('N', 'C')),
            expected_names=[None])

        # right broadcast
        self._test_name_inference(
            torch.masked_select,
            (create('N:2,C:3'), (create('3') > 0).rename('C')),
            expected_names=[None])

        # error
        self._test_name_inference(
            torch.masked_select,
            (create('N:2,C:3'), (create('3') > 0).rename('D')),
            maybe_raises_regex='do not match')

        # out=
        self._test_name_inference(
            out_fn(torch.masked_select),
            (create('0'), create('N:2,C:3'), (create('2,3') > 0).rename('N', 'C')),
            expected_names=[None])

    def test_cat(self):
        # simple
        self._test_name_inference(
            torch.cat,
            [[create('N:2,C:3'), create('N:2,C:3')]],
            expected_names=['N', 'C'])

        # error: zero dim
        self._test_name_inference(
            torch.cat,
            [[create(''), create('')]],
            maybe_raises_regex='zero-dim')

        # error: names don't match
        self._test_name_inference(
            torch.cat,
            [[create('N:2,C:3'), create('C:3,N:2')]],
            maybe_raises_regex='do not match')

        # error: different number of dims
        self._test_name_inference(
            torch.cat,
            [[create('N:2,C:3'), create('C:3')]],
            maybe_raises_regex='must have same number of dimensions')

        # out=
        self._test_name_inference(
            out_fn(torch.cat),
            [create('0'), [create('N:2,C:3'), create('N:2,C:3')]],
            expected_names=['N', 'C'])

    def test_masked_fill(self):
        # simple
        self._test_name_inference(
            Tensor.masked_fill,
            (create('N:2,C:3'), (create('2,3') > 0).rename('N', 'C'), 3.14),
            expected_names=['N', 'C'])

        # left broadcast
        self._test_name_inference(
            Tensor.masked_fill,
            (create('C:3'), (create('2,3') > 0).rename('N', 'C'), 3.14),
            maybe_raises_regex="must be less than or equal to")

        # right broadcast
        self._test_name_inference(
            Tensor.masked_fill,
            (create('N:2,C:3'), (create('3') > 0).rename('C'), 3.14),
            expected_names=['N', 'C'])

        # error
        self._test_name_inference(
            Tensor.masked_fill,
            (create('N:2,C:3'), (create('3') > 0).rename('D'), 3.14),
            maybe_raises_regex='do not match')

        # inplace
        self._test_name_inference(
            Tensor.masked_fill_,
            (create('N:2,C:3'), (create('2,3') > 0).rename('N', 'C'), 3.14),
            expected_names=['N', 'C'])

        # inplace, computed names don't match output tensor names
        self._test_name_inference(
            Tensor.masked_fill_,
            (create('N:2,None:3'), (create('2,3') > 0).rename('N', 'C'), 3.14),
            maybe_raises_regex="not the same as the computed output names")


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

    # This test is failing on Python 3.12: https://github.com/pytorch/pytorch/issues/119464
    @unittest.skipIf(sys.version_info >= (3, 12), "Failing on python 3.12+")
    def test_using_unseen_interned_string_bumps_refcount_permanently(self):
        # Please don't use this as a name in a different test.
        unseen_name = 'abcdefghi'
        old_refcnt = sys.getrefcount(unseen_name)

        pass_name_to_python_arg_parser(unseen_name)

        new_refcnt = sys.getrefcount(unseen_name)
        self.assertEqual(new_refcnt, old_refcnt + 1)

    # This test is failing on Python 3.12: https://github.com/pytorch/pytorch/issues/119464
    @unittest.skipIf(sys.version_info >= (3, 12), "Failing on python 3.12+")
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

    def test_no_jit_tracer_support(self):
        def foo(x):
            return torch.full(x.shape, 2., names=('N',))

        with self.assertRaisesRegex(RuntimeError, 'not supported with the tracer'):
            x = torch.randn(3)
            torch.jit.trace(foo, example_inputs=x)

        def bar(x):
            return x.select('N', 1)

        with self.assertRaisesRegex(RuntimeError, 'not supported with the tracer'):
            x = torch.randn(3)
            torch.jit.trace(bar, example_inputs=x)

    def test_no_jit_script_support(self):
        @torch.jit.script
        def foo(x):
            return x + 1

        with self.assertRaisesRegex(RuntimeError, 'NYI'):
            foo(torch.randn(2, 3, names=('N', 'C')))

        @torch.jit.ignore
        def add_names(x):
            x.names = ('N', 'C')

        @torch.jit.script
        def return_named_tensor(input):
            add_names(input)
            return input

        with self.assertRaisesRegex(RuntimeError, "NYI"):
            return_named_tensor(torch.randn(1, 1))

    def test_align_to(self):
        # trivial
        tensor = create('N:3')
        output = tensor.align_to('N')
        self.assertEqual(output.names, ['N'])
        self.assertEqual(output.shape, [3])

        # unsqueeze behavior
        tensor = create('N:3')
        output = tensor.align_to('N', 'D')
        self.assertEqual(output.names, ['N', 'D'])
        self.assertEqual(output.shape, [3, 1])

        # transpose behavior
        tensor = create('N:3,C:2')
        output = tensor.align_to('C', 'N')
        self.assertEqual(output.names, ['C', 'N'])
        self.assertEqual(output.shape, [2, 3])

        # unsqueeze / transpose
        tensor = create('C:2,N:3,H:5')
        output = tensor.align_to('N', 'H', 'W', 'C')
        self.assertEqual(output.names, ['N', 'H', 'W', 'C'])
        self.assertEqual(output.shape, [3, 5, 1, 2])

        # All input dimensions must be named
        with self.assertRaisesRegex(RuntimeError, "All input dims must be named. Found unnamed dim at index 0"):
            create('None:2,C:3').align_to('N', 'C')

        # not enough names
        with self.assertRaisesRegex(RuntimeError, "Cannot find dim 'N'"):
            create('N:2,C:3').align_to('C')

        # names not found
        with self.assertRaisesRegex(RuntimeError, "Cannot find dim 'C'"):
            create('N:2,C:3').align_to('D', 'N')

    def test_align_to_ellipsis(self):
        tensor = create('N:7,H:3,W:5,C:2')

        # ... = ['N', 'H', 'W', 'C']
        output = tensor.align_to('...')
        self.assertEqual(output.names, ['N', 'H', 'W', 'C'])
        self.assertEqual(output.shape, [7, 3, 5, 2])

        # ... = ['H', 'C']
        output = tensor.align_to('...', 'W', 'N')
        self.assertEqual(output.names, ['H', 'C', 'W', 'N'])
        self.assertEqual(output.shape, [3, 2, 5, 7])

        # ... = ['N', 'W']
        output = tensor.align_to('H', 'C', '...')
        self.assertEqual(output.names, ['H', 'C', 'N', 'W'])
        self.assertEqual(output.shape, [3, 2, 7, 5])

        # ... = ['H', 'C']
        output = tensor.align_to('W', '...', 'N')
        self.assertEqual(output.names, ['W', 'H', 'C', 'N'])
        self.assertEqual(output.shape, [5, 3, 2, 7])

        # ... = []
        output = tensor.align_to('N', '...', 'C', 'D', 'H', 'W')
        self.assertEqual(output.names, ['N', 'C', 'D', 'H', 'W'])
        self.assertEqual(output.shape, [7, 2, 1, 3, 5])

        # Input tensor partially named
        partially_named = create('None:2,None:3,None:5,C:7')
        output = partially_named.align_to('C', '...')
        self.assertEqual(output.names, ['C', None, None, None])
        self.assertEqual(output.shape, [7, 2, 3, 5])

        with self.assertRaisesRegex(RuntimeError, "order of dimensions cannot contain a None"):
            partially_named.align_to('C', None, '...')

        # Input order partially named
        with self.assertRaisesRegex(RuntimeError, "cannot contain a None name"):
            tensor.align_to('...', 'N', None)

        # Input order duplicate names
        with self.assertRaisesRegex(RuntimeError, "duplicate names"):
            tensor.align_to('...', 'N', 'N')

    def test_align_as(self):
        # align_as calls align_to internally. align_to has pretty substantial tests,
        # so just test some basic things here.
        tensor = create('C:2,N:3,H:5')
        other = create('N:1,H:1,W:1,C:1')
        output = tensor.align_as(other)
        self.assertEqual(output.names, ['N', 'H', 'W', 'C'])
        self.assertEqual(output.shape, [3, 5, 1, 2])

    @unittest.skip("Not implemented yet")
    def test_align_tensors_two_inputs(self):
        def _test(tensor_namedshape, align_names, expected_sizes, expected_error):
            tensor_names, tensor_sizes = tensor_namedshape
            tensor = torch.empty(*tensor_sizes, names=tensor_names)
            other = torch.empty([1] * len(align_names), names=align_names)
            if expected_error is not None:
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    torch.align_tensors(tensor, other)
                return

            output, _ = torch.align_tensors(tensor, other)
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

    @unittest.skip("Not implemented yet")
    def test_align_tensors(self):
        def reference_fn(*tensors):
            longest_names = tensors[0].names
            for tensor in tensors:
                if len(tensor.names) > len(longest_names):
                    longest_names = tensor.names
            return [tensor.align_to(*longest_names) for tensor in tensors]

        x = torch.empty(1, 1, names=('N', 'H'))
        y = torch.empty(2, 3, 5, names=('N', 'C', 'H'))
        z = torch.empty(2, names=('N',))
        output = torch.align_tensors(x, y, z)
        expected_tensors = reference_fn(x, y, z)
        for tensor, expected in zip(output, expected_tensors):
            self.assertTensorDataAndNamesEqual(tensor, expected)

    def test_mm(self):
        for device in get_all_device_types():
            self._test_name_inference(
                torch.mm, device=device,
                args=(create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            # left arg is unnamed
            self._test_name_inference(
                torch.mm, device=device,
                args=(create('3,2'), create('W:2,H:5')),
                expected_names=(None, 'H'))

            # right arg is unnamed
            self._test_name_inference(
                torch.mm, device=device,
                args=(create('N:3,C:2'), create('2,5')),
                expected_names=('N', None))

            # out=
            self._test_name_inference(
                out_fn(torch.mm), device=device,
                args=(create('0'), create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            self._test_name_inference(
                torch.mm, device=device,
                args=(create('N:3,C:2'), create('W:2,N:5')),
                maybe_raises_regex='with duplicate names')

    def test_expand(self):
        for device in get_all_device_types():
            self._test_name_inference(
                Tensor.expand, device=device,
                args=(create('D:1'), [3]), expected_names=('D',))

            self._test_name_inference(
                Tensor.expand, device=device,
                args=(create('H:3,W:2'), [10, 3, 3, 2]),
                expected_names=(None, None, 'H', 'W'))

            self._test_name_inference(
                Tensor.expand, device=device,
                args=(create('3, 2'), [10, 3, 3, 2]),
                expected_names=(None, None, None, None))

    def test_addmm(self):
        for device in get_all_device_types():
            # full names
            self._test_name_inference(
                torch.addmm, device=device,
                args=(create('N:3,H:5'), create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            # no name on bias
            self._test_name_inference(
                torch.addmm, device=device,
                args=(create('3,5'), create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            # partially named bias
            self._test_name_inference(
                torch.addmm, device=device,
                args=(create('N:3,None:5'), create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            # out=
            self._test_name_inference(
                out_fn(torch.addmm), device=device,
                args=(create('0'), create('N:3,None:5'), create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            # inplace
            self._test_name_inference(
                torch.Tensor.addmm_, device=device,
                args=(create('N:3,H:5'), create('N:3,C:2'), create('W:2,H:5')),
                expected_names=('N', 'H'))

            self._test_name_inference(
                torch.addmm, device=device,
                args=(create('N:3,H:5'), create('N:3,C:2'), create('W:2,N:5')),
                maybe_raises_regex='with duplicate names')

    def test_bmm(self):
        for device in get_all_device_types():
            # full names
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:7,A:3,B:2'), create('N:7,A:2,B:5')),
                expected_names=('N', 'A', 'B'))

            # no name on left tensor
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('7,3,2'), create('N:7,A:2,B:5')),
                expected_names=('N', None, 'B'))

            # no name on right tensor
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:7,A:3,B:2'), create('7,2,5')),
                expected_names=('N', 'A', None))

            # out=
            self._test_name_inference(
                out_fn(torch.bmm), device=device,
                args=(create('0'), create('N:7,A:3,B:2'), create('N:7,A:2,B:5')),
                expected_names=('N', 'A', 'B'))

            # duplicate names after mm
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:7,A:3,B:2'), create('N:7,B:2,A:5')),
                maybe_raises_regex='with duplicate names')

            # matching error (batch dimensions must be alignable)
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:3,A:3,B:3'), create('M:3,A:3,B:3')),
                maybe_raises_regex='do not match')

            # misalignment (batch dimension is getting contracted)
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:3,A:3,B:3'), create('None:3,N:3,B:3')),
                maybe_raises_regex='misaligned')

    def test_matmul(self):
        for device in get_all_device_types():
            # input tensors are less than 1D
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create(''), create('A:2')),
                maybe_raises_regex='at least 1D')
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('A:2'), create('')),
                maybe_raises_regex='at least 1D')

            # 1D @ 1D
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('A:2'), create('B:2')),
                expected_names=[])

            # ND @ 1D
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('A:3,C:2'), create('B:2')),
                expected_names=['A'])
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('A:5,C:3,D:2'), create('B:2')),
                expected_names=['A', 'C'])

            # 1D @ ND
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('C:2'), create('A:2,B:3')),
                expected_names=['B'])
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('C:2'), create('A:3,B:2,D:5')),
                expected_names=['A', 'D'])

            # 2D @ 2D
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('A:3,B:2'), create('A:2,B:3')),
                expected_names=['A', 'B'])
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('A:3,B:2'), create('B:2,A:5')),
                maybe_raises_regex='with duplicate names')

            # ND @ ND where N >= 2
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('C:5,A:3,B:2'), create('A:2,B:3')),
                expected_names=['C', 'A', 'B'])
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('C:5,A:3,B:2'), create('None:1,A:2,B:3')),
                expected_names=['C', 'A', 'B'])
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('C:5,A:3,B:2'), create('None:2,None:1,A:2,B:3')),
                expected_names=[None, 'C', 'A', 'B'])

            # out=
            self._test_name_inference(
                out_fn(torch.matmul), device=device,
                args=(create('0'), create('N:7,A:3,B:2'), create('N:7,A:2,B:5')),
                expected_names=('N', 'A', 'B'))

            # duplicate names after mm
            self._test_name_inference(
                torch.bmm, device=device,
                args=(create('N:7,A:3,B:2'), create('N:7,B:2,A:5')),
                maybe_raises_regex='with duplicate names')

            # misalignment (batch dimension is getting contracted)
            self._test_name_inference(
                torch.matmul, device=device,
                args=(create('N:3,A:3,B:3'), create('A:3,N:3,B:3')),
                maybe_raises_regex='do not match')

    def test_mv(self):
        for device in get_all_device_types():
            self._test_name_inference(
                torch.mv, device=device,
                args=(create('N:3,C:2'), create('W:2')),
                expected_names=('N',))

            # left arg is unnamed
            self._test_name_inference(
                torch.mv, device=device,
                args=(create('3,2'), create('W:2')),
                expected_names=(None,))

            # right arg is unnamed
            self._test_name_inference(
                torch.mv, device=device,
                args=(create('N:3,C:2'), create('2')),
                expected_names=('N',))

            # out=
            self._test_name_inference(
                out_fn(torch.mv), device=device,
                args=(create('0'), create('N:3,C:2'), create('W:2')),
                expected_names=('N',))

    def test_addmv(self):
        for device in get_all_device_types():
            # full names
            self._test_name_inference(
                torch.addmv, device=device,
                args=(create('N:3'), create('N:3,C:2'), create('H:2')),
                expected_names=['N'])

            # no name on bias
            self._test_name_inference(
                torch.addmv, device=device,
                args=(create('3'), create('N:3,C:2'), create('H:2')),
                expected_names=('N',))

            # out=
            self._test_name_inference(
                out_fn(torch.addmv), device=device,
                args=(create('0'), create('N:3'), create('N:3,C:2'), create('H:2')),
                expected_names=('N',))

            # inplace
            self._test_name_inference(
                torch.Tensor.addmv_, device=device,
                args=(create('N:3'), create('N:3,C:2'), create('H:2')),
                expected_names=('N',))

    def test_autograd_ignores_names(self):
        # sigmoid forward is supported by named tensors, but sigmoid_backward
        # is not (see native_functions.yaml). Test that autograd ignores names
        # and that the sigmoid_backward succeeds.
        x = torch.randn(3, 3, names=('N', 'C'), requires_grad=True)
        x.sigmoid().sum().backward()

    def test_tensor_grad_is_unnamed(self):
        x = torch.randn(3, 3, names=(None, None), requires_grad=True)
        y = torch.randn(3, 3, names=('N', 'C'), requires_grad=True)
        (x * y).sum().backward()

        # Check that names weren't propagated
        self.assertEqual(y.grad.names, [None, None])
        self.assertEqual(x.grad.names, [None, None])

    def test_autograd_warns_named_grad(self):
        base = torch.randn(3, 3, names=('N', 'C'))
        named_grad = base.clone()
        base.requires_grad_()

        with warnings.catch_warnings(record=True) as warns:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            base.clone().backward(named_grad)
            self.assertEqual(len(warns), 1)
            self.assertTrue(
                str(warns[0].message).startswith('Autograd was passed a named grad tensor'))

    def test_nyi_dimname_overload_msg(self):
        x = torch.randn(3, 3)
        with self.assertRaisesRegex(RuntimeError, "squeeze: You passed a dimname"):
            x.squeeze_("N")

    def test_dot(self):
        for device in get_all_device_types():
            # torch.dot ignores the names of both tensors
            self._test_name_inference(
                torch.dot, device=device,
                args=(create('C:2'), create('W:2')),
                expected_names=[])

    def test_comparison_ops(self):
        for device in get_all_device_types():
            a = torch.randn(3, 3, names=('N', 'C'), device=device)
            b = torch.randn(3, 3, names=('N', 'C'), device=device)
            scalar = torch.randn([], device=device)

            self.assertEqual((a == b).names, ['N', 'C'])
            self.assertEqual((a != b).names, ['N', 'C'])
            self.assertEqual((a > b).names, ['N', 'C'])
            self.assertEqual((a < b).names, ['N', 'C'])
            self.assertEqual((a >= b).names, ['N', 'C'])
            self.assertEqual((a <= b).names, ['N', 'C'])

            self.assertEqual((a == 1).names, ['N', 'C'])
            self.assertEqual((a != 1).names, ['N', 'C'])
            self.assertEqual((a > 1).names, ['N', 'C'])
            self.assertEqual((a < 1).names, ['N', 'C'])
            self.assertEqual((a >= 1).names, ['N', 'C'])
            self.assertEqual((a <= 1).names, ['N', 'C'])

            self.assertEqual((a == scalar).names, ['N', 'C'])
            self.assertEqual((a != scalar).names, ['N', 'C'])
            self.assertEqual((a > scalar).names, ['N', 'C'])
            self.assertEqual((a < scalar).names, ['N', 'C'])
            self.assertEqual((a >= scalar).names, ['N', 'C'])
            self.assertEqual((a <= scalar).names, ['N', 'C'])

            res = torch.empty(3, 3, dtype=torch.bool, device=device)
            torch.eq(a, b, out=res)
            self.assertEqual(res.names, ['N', 'C'])
            torch.ne(a, b, out=res)
            self.assertEqual(res.names, ['N', 'C'])
            torch.lt(a, b, out=res)
            self.assertEqual(res.names, ['N', 'C'])
            torch.gt(a, b, out=res)
            self.assertEqual(res.names, ['N', 'C'])
            torch.le(a, b, out=res)
            self.assertEqual(res.names, ['N', 'C'])
            torch.ge(a, b, out=res)
            self.assertEqual(res.names, ['N', 'C'])

            res = torch.isnan(a)
            self.assertEqual(res.names, ['N', 'C'])

            res = torch.isinf(a)
            self.assertEqual(res.names, ['N', 'C'])

    def test_support_device_named_grad(self):
        named_tensor = torch.randn(3, 3, device='meta')
        with self.assertRaisesRegex(RuntimeError, 'NYI: named tensors only support CPU, CUDA'):
            named_tensor.rename_('N', 'C')
            named_tensor.names = ['N', 'C']
            named_tensor = torch.randn(3, 3, device='meta', names=['N', 'C'])


if __name__ == '__main__':
    run_tests()
