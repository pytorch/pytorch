r"""Importing this file must **not** initialize CUDA context. test_distributed
relies on this assumption to properly run. This means that when this is imported
no CUDA calls shall be made, including torch.cuda.device_count(), etc.

common_cuda.py can freely initialize CUDA context when imported.
"""

import sys
import os
import platform
import re
import gc
import types
import inspect
import argparse
import unittest
import warnings
import random
import contextlib
import socket
from collections import OrderedDict
from functools import wraps
from itertools import product
from copy import deepcopy
from numbers import Number

import __main__
import errno

import expecttest
import hashlib

import torch
import torch.cuda
from torch._utils_internal import get_writable_path
from torch._six import string_classes, inf
import torch.backends.cudnn
import torch.backends.mkl


torch.set_default_tensor_type('torch.DoubleTensor')
torch.backends.cudnn.disable_global_flags()


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--accept', action='store_true')
args, remaining = parser.parse_known_args()
SEED = args.seed
if not expecttest.ACCEPT:
    expecttest.ACCEPT = args.accept
UNITTEST_ARGS = [sys.argv[0]] + remaining
torch.manual_seed(SEED)


def run_tests(argv=UNITTEST_ARGS):
    unittest.main(argv=argv)

PY3 = sys.version_info > (3, 0)
PY34 = sys.version_info >= (3, 4)

IS_WINDOWS = sys.platform == "win32"
IS_PPC = platform.machine() == "ppc64le"


def _check_module_exists(name):
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    if not PY3:  # Python 2
        import imp
        try:
            imp.find_module(name)
            return True
        except ImportError:
            return False
    elif not PY34:  # Python [3, 3.4)
        import importlib
        loader = importlib.find_loader(name)
        return loader is not None
    else:  # Python >= 3.4
        import importlib
        import importlib.util
        spec = importlib.util.find_spec(name)
        return spec is not None

TEST_NUMPY = _check_module_exists('numpy')
TEST_SCIPY = _check_module_exists('scipy')
TEST_MKL = torch.backends.mkl.is_available()
TEST_NUMBA = _check_module_exists('numba')

# On Py2, importing librosa 0.6.1 triggers a TypeError (if using newest joblib)
# see librosa/librosa#729.
# TODO: allow Py2 when librosa 0.6.2 releases
TEST_LIBROSA = _check_module_exists('librosa') and PY3

# Python 2.7 doesn't have spawn
NO_MULTIPROCESSING_SPAWN = os.environ.get('NO_MULTIPROCESSING_SPAWN', '0') == '1' or sys.version_info[0] == 2
TEST_WITH_ASAN = os.getenv('PYTORCH_TEST_WITH_ASAN', '0') == '1'
TEST_WITH_UBSAN = os.getenv('PYTORCH_TEST_WITH_UBSAN', '0') == '1'
TEST_WITH_ROCM = os.getenv('PYTORCH_TEST_WITH_ROCM', '0') == '1'

if TEST_NUMPY:
    import numpy


def skipIfRocm(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if TEST_WITH_ROCM:
            raise unittest.SkipTest("test doesn't currently work on the ROCm stack")
        else:
            fn(*args, **kwargs)
    return wrapper


def skipIfNoLapack(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not torch._C.has_lapack:
            raise unittest.SkipTest('PyTorch compiled without Lapack')
        else:
            fn(*args, **kwargs)
    return wrapper


def skipCUDAMemoryLeakCheckIf(condition):
    def dec(fn):
        if getattr(fn, '_do_cuda_memory_leak_check', True):  # if current True
            fn._do_cuda_memory_leak_check = not condition
        return fn
    return dec


def suppress_warnings(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args, **kwargs)
    return wrapper


def get_cpu_type(type_name):
    module, name = type_name.rsplit('.', 1)
    assert module == 'torch.cuda'
    return getattr(torch, name)


def get_gpu_type(type_name):
    if isinstance(type_name, type):
        type_name = '{}.{}'.format(type_name.__module__, type_name.__name__)
    module, name = type_name.rsplit('.', 1)
    assert module == 'torch'
    return getattr(torch.cuda, name)


def to_gpu(obj, type_map={}):
    if isinstance(obj, torch.Tensor):
        assert obj.is_leaf
        t = type_map.get(obj.type(), get_gpu_type(obj.type()))
        with torch.no_grad():
            res = obj.clone().type(t)
            res.requires_grad = obj.requires_grad
        return res
    elif torch.is_storage(obj):
        return obj.new().resize_(obj.size()).copy_(obj)
    elif isinstance(obj, list):
        return [to_gpu(o, type_map) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(to_gpu(o, type_map) for o in obj)
    else:
        return deepcopy(obj)


def get_function_arglist(func):
    return inspect.getargspec(func).args


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if TEST_NUMPY:
        numpy.random.seed(seed)


@contextlib.contextmanager
def freeze_rng_state():
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    yield
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_rng_state)
    torch.set_rng_state(rng_state)


def iter_indices(tensor):
    if tensor.dim() == 0:
        return range(0)
    if tensor.dim() == 1:
        return range(tensor.size(0))
    return product(*(range(s) for s in tensor.size()))


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class CudaMemoryLeakCheck():
    def __init__(self, testcase, name=None):
        self.name = testcase.id() if name is None else name
        self.testcase = testcase

        # initialize context & RNG to prevent false positive detections
        # when the test is the first to initialize those
        from common_cuda import initialize_cuda_context_rng
        initialize_cuda_context_rng()

    @staticmethod
    def get_cuda_memory_usage():
        # we don't need CUDA synchronize because the statistics are not tracked at
        # actual freeing, but at when marking the block as free.
        num_devices = torch.cuda.device_count()
        gc.collect()
        return tuple(torch.cuda.memory_allocated(i) for i in range(num_devices))

    def __enter__(self):
        self.befores = self.get_cuda_memory_usage()

    def __exit__(self, exec_type, exec_value, traceback):
        # Don't check for leaks if an exception was thrown
        if exec_type is not None:
            return
        afters = self.get_cuda_memory_usage()
        for i, (before, after) in enumerate(zip(self.befores, afters)):
            self.testcase.assertEqual(
                before, after, '{} leaked {} bytes CUDA memory on device {}'.format(
                    self.name, after - before, i))


class TestCase(expecttest.TestCase):
    precision = 1e-5
    maxDiff = None
    _do_cuda_memory_leak_check = False

    def __init__(self, method_name='runTest'):
        super(TestCase, self).__init__(method_name)
        # Wraps the tested method if we should do CUDA memory check.
        test_method = getattr(self, method_name)
        self._do_cuda_memory_leak_check &= getattr(test_method, '_do_cuda_memory_leak_check', True)
        # FIXME: figure out the flaky -1024 anti-leaks on windows. See #8044
        if self._do_cuda_memory_leak_check and not IS_WINDOWS:
            # the import below may initialize CUDA context, so we do it only if
            # self._do_cuda_memory_leak_check is True.
            from common_cuda import TEST_CUDA
            fullname = self.id().lower()  # class_name.method_name
            if TEST_CUDA and ('gpu' in fullname or 'cuda' in fullname):
                setattr(self, method_name, self.wrap_with_cuda_memory_check(test_method))

    def assertLeaksNoCudaTensors(self, name=None):
        name = self.id() if name is None else name
        return CudaMemoryLeakCheck(self, name)

    def wrap_with_cuda_memory_check(self, method):
        # Assumes that `method` is the tested function in `self`.
        # NOTE: Python Exceptions (e.g., unittest.Skip) keeps objects in scope
        #       alive, so this cannot be done in setUp and tearDown because
        #       tearDown is run unconditionally no matter whether the test
        #       passes or not. For the same reason, we can't wrap the `method`
        #       call in try-finally and always do the check.
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            with self.assertLeaksNoCudaTensors():
                method(*args, **kwargs)
        return types.MethodType(wrapper, self)

    def setUp(self):
        set_rng_seed(SEED)

    def assertTensorsSlowEqual(self, x, y, prec=None, message=''):
        max_err = 0
        self.assertEqual(x.size(), y.size())
        for index in iter_indices(x):
            max_err = max(max_err, abs(x[index] - y[index]))
        self.assertLessEqual(max_err, prec, message)

    def genSparseTensor(self, size, sparse_dim, nnz, is_uncoalesced, device='cpu'):
        # Assert not given impossible combination, where the sparse dims have
        # empty numel, but nnz > 0 makes the indices containing values.
        assert all(size[d] > 0 for d in range(sparse_dim)) or nnz == 0, 'invalid arguments'

        v_size = [nnz] + list(size[sparse_dim:])
        v = torch.randn(*v_size, device=device)
        i = torch.rand(sparse_dim, nnz, device=device)
        i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
        i = i.to(torch.long)
        if is_uncoalesced:
            v = torch.cat([v, torch.randn_like(v)], 0)
            i = torch.cat([i, i], 1)

        x = torch.sparse_coo_tensor(i, v, torch.Size(size))

        if not is_uncoalesced:
            x = x.coalesce()
        else:
            # FIXME: `x` is a sparse view of `v`. Currently rebase_history for
            #        sparse views is not implemented, so this workaround is
            #        needed for inplace operations done on `x`, e.g., copy_().
            #        Remove after implementing something equivalent to CopySlice
            #        for sparse views.
            x = x.detach()
        return x, x._indices().clone(), x._values().clone()

    def safeToDense(self, t):
        r = self.safeCoalesce(t)
        return r.to_dense()

    def safeCoalesce(self, t):
        tc = t.coalesce()
        self.assertEqual(tc.to_dense(), t.to_dense())
        self.assertTrue(tc.is_coalesced())

        # Our code below doesn't work when nnz is 0, because
        # then it's a 0D tensor, not a 2D tensor.
        if t._nnz() == 0:
            self.assertEqual(t._indices(), tc._indices())
            self.assertEqual(t._values(), tc._values())
            return tc

        value_map = {}
        for idx, val in zip(t._indices().t(), t._values()):
            idx_tup = tuple(idx.tolist())
            if idx_tup in value_map:
                value_map[idx_tup] += val
            else:
                value_map[idx_tup] = val.clone() if isinstance(val, torch.Tensor) else val

        new_indices = sorted(list(value_map.keys()))
        new_values = [value_map[idx] for idx in new_indices]
        if t._values().ndimension() < 2:
            new_values = t._values().new(new_values)
        else:
            new_values = torch.stack(new_values)

        new_indices = t._indices().new(new_indices).t()
        tg = t.new(new_indices, new_values, t.size())

        self.assertEqual(tc._indices(), tg._indices())
        self.assertEqual(tc._values(), tg._values())

        if t.is_coalesced():
            self.assertEqual(tc._indices(), t._indices())
            self.assertEqual(tc._values(), t._values())

        return tg

    def assertEqual(self, x, y, prec=None, message='', allow_inf=False):
        if isinstance(prec, str) and message == '':
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

        if isinstance(x, torch.Tensor) and isinstance(y, Number):
            self.assertEqual(x.item(), y, prec, message, allow_inf)
        elif isinstance(y, torch.Tensor) and isinstance(x, Number):
            self.assertEqual(x, y.item(), prec, message, allow_inf)
        elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            def assertTensorsEqual(a, b):
                super(TestCase, self).assertEqual(a.size(), b.size(), message)
                if a.numel() > 0:
                    b = b.type_as(a)
                    b = b.cuda(device=a.get_device()) if a.is_cuda else b.cpu()
                    # check that NaNs are in the same locations
                    nan_mask = a != a
                    self.assertTrue(torch.equal(nan_mask, b != b), message)
                    diff = a - b
                    diff[nan_mask] = 0
                    # inf check if allow_inf=True
                    if allow_inf:
                        inf_mask = (a == float("inf")) | (a == float("-inf"))
                        self.assertTrue(torch.equal(inf_mask,
                                                    (b == float("inf")) | (b == float("-inf"))),
                                        message)
                        diff[inf_mask] = 0
                    # TODO: implement abs on CharTensor
                    if diff.is_signed() and 'CharTensor' not in diff.type():
                        diff = diff.abs()
                    max_err = diff.max()
                    self.assertLessEqual(max_err, prec, message)
            super(TestCase, self).assertEqual(x.is_sparse, y.is_sparse, message)
            if x.is_sparse:
                x = self.safeCoalesce(x)
                y = self.safeCoalesce(y)
                assertTensorsEqual(x._indices(), y._indices())
                assertTensorsEqual(x._values(), y._values())
            else:
                assertTensorsEqual(x, y)
        elif isinstance(x, string_classes) and isinstance(y, string_classes):
            super(TestCase, self).assertEqual(x, y, message)
        elif type(x) == set and type(y) == set:
            super(TestCase, self).assertEqual(x, y, message)
        elif isinstance(x, dict) and isinstance(y, dict):
            if isinstance(x, OrderedDict) and isinstance(y, OrderedDict):
                self.assertEqual(x.items(), y.items())
            else:
                self.assertEqual(set(x.keys()), set(y.keys()))
                key_list = list(x.keys())
                self.assertEqual([x[k] for k in key_list], [y[k] for k in key_list])
        elif is_iterable(x) and is_iterable(y):
            super(TestCase, self).assertEqual(len(x), len(y), message)
            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, prec, message)
        elif isinstance(x, bool) and isinstance(y, bool):
            super(TestCase, self).assertEqual(x, y, message)
        elif isinstance(x, Number) and isinstance(y, Number):
            if abs(x) == inf or abs(y) == inf:
                if allow_inf:
                    super(TestCase, self).assertEqual(x, y, message)
                else:
                    self.fail("Expected finite numeric values - x={}, y={}".format(x, y))
                return
            super(TestCase, self).assertLessEqual(abs(x - y), prec, message)
        else:
            super(TestCase, self).assertEqual(x, y, message)

    def assertAlmostEqual(self, x, y, places=None, msg=None, delta=None, allow_inf=None):
        prec = delta
        if places:
            prec = 10**(-places)
        self.assertEqual(x, y, prec, msg, allow_inf)

    def assertNotEqual(self, x, y, prec=None, message=''):
        if isinstance(prec, str) and message == '':
            message = prec
            prec = None
        if prec is None:
            prec = self.precision

        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.size() != y.size():
                super(TestCase, self).assertNotEqual(x.size(), y.size())
            self.assertGreater(x.numel(), 0)
            y = y.type_as(x)
            y = y.cuda(device=x.get_device()) if x.is_cuda else y.cpu()
            nan_mask = x != x
            if torch.equal(nan_mask, y != y):
                diff = x - y
                if diff.is_signed():
                    diff = diff.abs()
                diff[nan_mask] = 0
                max_err = diff.max()
                self.assertGreaterEqual(max_err, prec, message)
        elif type(x) == str and type(y) == str:
            super(TestCase, self).assertNotEqual(x, y)
        elif is_iterable(x) and is_iterable(y):
            super(TestCase, self).assertNotEqual(x, y)
        else:
            try:
                self.assertGreaterEqual(abs(x - y), prec, message)
                return
            except (TypeError, AssertionError):
                pass
            super(TestCase, self).assertNotEqual(x, y, message)

    def assertObjectIn(self, obj, iterable):
        for elem in iterable:
            if id(obj) == id(elem):
                return
        raise AssertionError("object not found in iterable")

    # TODO: Support context manager interface
    # NB: The kwargs forwarding to callable robs the 'subname' parameter.
    # If you need it, manually apply your callable in a lambda instead.
    def assertExpectedRaises(self, exc_type, callable, *args, **kwargs):
        subname = None
        if 'subname' in kwargs:
            subname = kwargs['subname']
            del kwargs['subname']
        try:
            callable(*args, **kwargs)
        except exc_type as e:
            self.assertExpected(str(e), subname)
            return
        # Don't put this in the try block; the AssertionError will catch it
        self.fail(msg="Did not raise when expected to")

    def assertWarns(self, callable, msg=''):
        r"""
        Test if :attr:`callable` raises a warning.
        """
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            callable()
            self.assertTrue(len(ws) > 0, msg)

    def assertWarnsRegex(self, callable, regex, msg=''):
        r"""
        Test if :attr:`callable` raises any warning with message that contains
        the regex pattern :attr:`regex`.
        """
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            callable()
            self.assertTrue(len(ws) > 0, msg)
            found = any(re.search(regex, str(w.message)) is not None for w in ws)
            self.assertTrue(found, msg)

    def assertExpected(self, s, subname=None):
        r"""
        Test that a string matches the recorded contents of a file
        derived from the name of this test and subname.  This file
        is placed in the 'expect' directory in the same directory
        as the test script. You can automatically update the recorded test
        output using --accept.

        If you call this multiple times in a single function, you must
        give a unique subname each time.
        """
        if not (isinstance(s, str) or (sys.version_info[0] == 2 and isinstance(s, unicode))):
            raise TypeError("assertExpected is strings only")

        def remove_prefix(text, prefix):
            if text.startswith(prefix):
                return text[len(prefix):]
            return text
        # NB: we take __file__ from the module that defined the test
        # class, so we place the expect directory where the test script
        # lives, NOT where test/common_utils.py lives.  This doesn't matter in
        # PyTorch where all test scripts are in the same directory as
        # test/common_utils.py, but it matters in onnx-pytorch
        module_id = self.__class__.__module__
        munged_id = remove_prefix(self.id(), module_id + ".")
        test_file = os.path.realpath(sys.modules[module_id].__file__)
        expected_file = os.path.join(os.path.dirname(test_file),
                                     "expect",
                                     munged_id)

        subname_output = ""
        if subname:
            expected_file += "-" + subname
            subname_output = " ({})".format(subname)
        expected_file += ".expect"
        expected = None

        def accept_output(update_type):
            print("Accepting {} for {}{}:\n\n{}".format(update_type, munged_id, subname_output, s))
            with open(expected_file, 'w') as f:
                f.write(s)

        try:
            with open(expected_file) as f:
                expected = f.read()
        except IOError as e:
            if e.errno != errno.ENOENT:
                raise
            elif expecttest.ACCEPT:
                return accept_output("output")
            else:
                raise RuntimeError(
                    ("I got this output for {}{}:\n\n{}\n\n"
                     "No expect file exists; to accept the current output, run:\n"
                     "python {} {} --accept").format(munged_id, subname_output, s, __main__.__file__, munged_id))

        # a hack for JIT tests
        if IS_WINDOWS:
            expected = re.sub(r'CppOp\[(.+?)\]', 'CppOp[]', expected)
            s = re.sub(r'CppOp\[(.+?)\]', 'CppOp[]', s)

        if expecttest.ACCEPT:
            if expected != s:
                return accept_output("updated output")
        else:
            if hasattr(self, "assertMultiLineEqual"):
                # Python 2.7 only
                # NB: Python considers lhs "old" and rhs "new".
                self.assertMultiLineEqual(expected, s)
            else:
                self.assertEqual(s, expected)

    if sys.version_info < (3, 2):
        # assertRegexpMatches renamed to assertRegex in 3.2
        assertRegex = unittest.TestCase.assertRegexpMatches
        # assertRaisesRegexp renamed to assertRaisesRegex in 3.2
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp


def download_file(url, binary=True):
    if sys.version_info < (3,):
        from urlparse import urlsplit
        import urllib2
        request = urllib2
        error = urllib2
    else:
        from urllib.parse import urlsplit
        from urllib import request, error

    filename = os.path.basename(urlsplit(url)[2])
    data_dir = get_writable_path(os.path.join(os.path.dirname(__file__), 'data'))
    path = os.path.join(data_dir, filename)

    if os.path.exists(path):
        return path
    try:
        data = request.urlopen(url, timeout=15).read()
        with open(path, 'wb' if binary else 'w') as f:
            f.write(data)
        return path
    except error.URLError:
        msg = "could not download test file '{}'".format(url)
        warnings.warn(msg, RuntimeWarning)
        raise unittest.SkipTest(msg)


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('localhost', 0))
    sockname = sock.getsockname()
    sock.close()
    return sockname[1]


# Methods for matrix generation
# Used in test_autograd.py and test_torch.py
def prod_single_zero(dim_size):
    result = torch.randn(dim_size, dim_size)
    result[0, 1] = 0
    return result


def random_square_matrix_of_rank(l, rank):
    assert rank <= l
    A = torch.randn(l, l)
    u, s, v = A.svd()
    for i in range(l):
        if i >= rank:
            s[i] = 0
        elif s[i] == 0:
            s[i] = 1
    return u.mm(torch.diag(s)).mm(v.transpose(0, 1))


def random_symmetric_matrix(l):
    A = torch.randn(l, l)
    for i in range(l):
        for j in range(i):
            A[i, j] = A[j, i]
    return A


def random_symmetric_psd_matrix(l):
    A = torch.randn(l, l)
    return A.mm(A.transpose(0, 1))


def random_symmetric_pd_matrix(l, *batches):
    A = torch.randn(*(batches + (l, l)))
    return A.matmul(A.transpose(-2, -1)) + torch.eye(l) * 1e-5


def make_nonzero_det(A, sign=None, min_singular_value=0.1):
    u, s, v = A.svd()
    s[s < min_singular_value] = min_singular_value
    A = u.mm(torch.diag(s)).mm(v.t())
    det = A.det().item()
    if sign is not None:
        if (det < 0) ^ (sign < 0):
            A[0, :].neg_()
    return A


def random_fullrank_matrix_distinct_singular_value(l, *batches, **kwargs):
    silent = kwargs.get("silent", False)
    if silent and not torch._C.has_lapack:
        return torch.ones(l, l)

    if len(batches) == 0:
        A = torch.randn(l, l)
        u, _, v = A.svd()
        s = torch.arange(1., l + 1).mul_(1.0 / (l + 1))
        return u.mm(torch.diag(s)).mm(v.t())
    else:
        all_matrices = []
        for _ in range(0, torch.prod(torch.as_tensor(batches)).item()):
            A = torch.randn(l, l)
            u, _, v = A.svd()
            s = torch.arange(1., l + 1).mul_(1.0 / (l + 1))
            all_matrices.append(u.mm(torch.diag(s)).mm(v.t()))
        return torch.stack(all_matrices).reshape(*(batches + (l, l)))


def do_test_dtypes(self, dtypes, layout, device):
    for dtype in dtypes:
        if dtype != torch.float16:
            out = torch.zeros((2, 3), dtype=dtype, layout=layout, device=device)
            self.assertIs(dtype, out.dtype)
            self.assertIs(layout, out.layout)
            self.assertEqual(device, out.device)


def do_test_empty_full(self, dtypes, layout, device):
    shape = torch.Size([2, 3])

    def check_value(tensor, dtype, layout, device, value, requires_grad):
        self.assertEqual(shape, tensor.shape)
        self.assertIs(dtype, tensor.dtype)
        self.assertIs(layout, tensor.layout)
        self.assertEqual(tensor.requires_grad, requires_grad)
        if tensor.is_cuda and device is not None:
            self.assertEqual(device, tensor.device)
        if value is not None:
            fill = tensor.new(shape).fill_(value)
            self.assertEqual(tensor, fill)

    def get_int64_dtype(dtype):
        module = '.'.join(str(dtype).split('.')[1:-1])
        if not module:
            return torch.int64
        return operator.attrgetter(module)(torch).int64

    default_dtype = torch.get_default_dtype()
    check_value(torch.empty(shape), default_dtype, torch.strided, -1, None, False)
    check_value(torch.full(shape, -5), default_dtype, torch.strided, -1, None, False)
    for dtype in dtypes:
        for rg in {dtype.is_floating_point, False}:
            int64_dtype = get_int64_dtype(dtype)
            v = torch.empty(shape, dtype=dtype, device=device, layout=layout, requires_grad=rg)
            check_value(v, dtype, layout, device, None, rg)
            out = v.new()
            check_value(torch.empty(shape, out=out, device=device, layout=layout, requires_grad=rg),
                        dtype, layout, device, None, rg)
            check_value(v.new_empty(shape), dtype, layout, device, None, False)
            check_value(v.new_empty(shape, dtype=int64_dtype, device=device, requires_grad=False),
                        int64_dtype, layout, device, None, False)
            check_value(torch.empty_like(v), dtype, layout, device, None, False)
            check_value(torch.empty_like(v, dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                        int64_dtype, layout, device, None, False)

            if dtype is not torch.float16 and layout != torch.sparse_coo:
                fv = 3
                v = torch.full(shape, fv, dtype=dtype, layout=layout, device=device, requires_grad=rg)
                check_value(v, dtype, layout, device, fv, rg)
                check_value(v.new_full(shape, fv + 1), dtype, layout, device, fv + 1, False)
                out = v.new()
                check_value(torch.full(shape, fv + 2, out=out, device=device, layout=layout, requires_grad=rg),
                            dtype, layout, device, fv + 2, rg)
                check_value(v.new_full(shape, fv + 3, dtype=int64_dtype, device=device, requires_grad=False),
                            int64_dtype, layout, device, fv + 3, False)
                check_value(torch.full_like(v, fv + 4), dtype, layout, device, fv + 4, False)
                check_value(torch.full_like(v, fv + 5,
                                            dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                            int64_dtype, layout, device, fv + 5, False)


IS_SANDCASTLE = os.getenv('SANDCASTLE') == '1' or os.getenv('TW_JOB_USER') == 'sandcastle'

THESE_TAKE_WAY_TOO_LONG = {
    'test_Conv3d_groups',
    'test_conv_double_backward_groups',
    'test_Conv3d_dilated',
    'test_Conv3d_stride_padding',
    'test_Conv3d_dilated_strided',
    'test_Conv3d',
    'test_Conv2d_dilated',
    'test_ConvTranspose3d_dilated',
    'test_ConvTranspose2d_dilated',
    'test_snli',
    'test_Conv2d',
    'test_Conv2d_padding',
    'test_ConvTranspose2d_no_bias',
    'test_ConvTranspose2d',
    'test_ConvTranspose3d',
    'test_Conv2d_no_bias',
    'test_matmul_4d_4d',
    'test_multinomial_invalid_probs',
}


running_script_path = None


def set_running_script_path():
    global running_script_path
    try:
        running_file = os.path.abspath(os.path.realpath(sys.argv[0]))
        if running_file.endswith('.py'):  # skip if the running file is not a script
            running_script_path = running_file
    except Exception:
        pass


def check_test_defined_in_running_script(test_case):
    if running_script_path is None:
        return
    test_case_class_file = os.path.abspath(os.path.realpath(inspect.getfile(test_case.__class__)))
    assert test_case_class_file == running_script_path, "Class of loaded TestCase \"{}\" " \
        "is not defined in the running script \"{}\", but in \"{}\". Did you " \
        "accidentally import a unittest.TestCase from another file?".format(
            test_case.id(), running_script_path, test_case_class_file)


num_shards = os.environ.get('TEST_NUM_SHARDS', None)
shard = os.environ.get('TEST_SHARD', None)
if num_shards is not None and shard is not None:
    num_shards = int(num_shards)
    shard = int(shard)

    def load_tests(loader, tests, pattern):
        set_running_script_path()
        test_suite = unittest.TestSuite()
        for test_group in tests:
            for test in test_group:
                check_test_defined_in_running_script(test)
                name = test.id().split('.')[-1]
                if name in THESE_TAKE_WAY_TOO_LONG:
                    continue
                hash_id = int(hashlib.sha256(str(test).encode('utf-8')).hexdigest(), 16)
                if hash_id % num_shards == shard:
                    test_suite.addTest(test)
        return test_suite
else:

    def load_tests(loader, tests, pattern):
        set_running_script_path()
        test_suite = unittest.TestSuite()
        for test_group in tests:
            for test in test_group:
                check_test_defined_in_running_script(test)
                test_suite.addTest(test)
        return test_suite
