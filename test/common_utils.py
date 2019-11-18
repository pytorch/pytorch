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
import subprocess
import time
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from itertools import product
from copy import deepcopy
from numbers import Number
import tempfile

import __main__
import errno

import expecttest

import torch
import torch.cuda
from torch._utils_internal import get_writable_path
from torch._six import string_classes, inf
import torch.backends.cudnn
import torch.backends.mkl
from enum import Enum

torch.backends.disable_global_flags()

IS_SANDCASTLE = os.getenv('SANDCASTLE') == '1' or os.getenv('TW_JOB_USER') == 'sandcastle'

class ProfilingMode(Enum):
    LEGACY = 1
    SIMPLE = 2
    PROFILING = 3

@contextmanager
def enable_profiling_mode():
    if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
        old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
        old_prof_mode_state = torch._C._jit_set_profiling_mode(True)
    try:
        yield
    finally:
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            torch._C._jit_set_profiling_executor(old_prof_exec_state)
            torch._C._jit_set_profiling_mode(old_prof_mode_state)

func_call = torch._C.ScriptFunction.__call__
meth_call = torch._C.ScriptMethod.__call__

def prof_callable(callable, *args, **kwargs):
    if 'profile_and_replay' in kwargs:
        del kwargs['profile_and_replay']
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            with enable_profiling_mode():
                callable(*args, **kwargs)
                return callable(*args, **kwargs)

    return callable(*args, **kwargs)

def prof_func_call(*args, **kwargs):
    return prof_callable(func_call, *args, **kwargs)

def prof_meth_call(*args, **kwargs):
    return prof_callable(meth_call, *args, **kwargs)

torch._C.ScriptFunction.__call__ = prof_func_call
torch._C.ScriptMethod.__call__ = prof_meth_call

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--subprocess', action='store_true',
                    help='whether to run each test in a subprocess')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--accept', action='store_true')
parser.add_argument('--ge_config', type=str)

GRAPH_EXECUTOR = ProfilingMode.SIMPLE if IS_SANDCASTLE else ProfilingMode.PROFILING
args, remaining = parser.parse_known_args()
if args.ge_config == 'legacy':
    GRAPH_EXECUTOR = ProfilingMode.LEGACY
elif args.ge_config == 'simple':
    GRAPH_EXECUTOR = ProfilingMode.SIMPLE

TEST_IN_SUBPROCESS = args.subprocess
SEED = args.seed
if not expecttest.ACCEPT:
    expecttest.ACCEPT = args.accept
UNITTEST_ARGS = [sys.argv[0]] + remaining
torch.manual_seed(SEED)


def shell(command, cwd=None):
    sys.stdout.flush()
    sys.stderr.flush()
    # The following cool snippet is copied from Py3 core library subprocess.call
    # only the with
    #   1. `except KeyboardInterrupt` block added for SIGINT handling.
    #   2. In Py2, subprocess.Popen doesn't return a context manager, so we do
    #      `p.wait()` in a `final` block for the code to be portable.
    #
    # https://github.com/python/cpython/blob/71b6c1af727fbe13525fb734568057d78cea33f3/Lib/subprocess.py#L309-L323
    assert not isinstance(command, torch._six.string_classes), "Command to shell should be a list or tuple of tokens"
    p = subprocess.Popen(command, universal_newlines=True, cwd=cwd)
    try:
        return p.wait()
    except KeyboardInterrupt:
        # Give `p` a chance to handle KeyboardInterrupt. Without this,
        # `pytest` can't print errors it collected so far upon KeyboardInterrupt.
        exit_status = p.wait(timeout=5)
        if exit_status is not None:
            return exit_status
        else:
            p.kill()
            raise
    except:  # noqa E722, copied from python core library
        p.kill()
        raise
    finally:
        # Always call p.wait() to ensure exit
        p.wait()

ALL_TENSORTYPES = [torch.float,
                   torch.double,
                   torch.half]

def fake_empty_like(*args, **kwargs):
    if 'memory_format' not in kwargs:
        kwargs['memory_format'] = torch.contiguous_format
    return torch.empty_like(*args, **kwargs)

def fake_rand_like(*args, **kwargs):
    if 'memory_format' not in kwargs:
        kwargs['memory_format'] = torch.contiguous_format
    return torch.rand_like(*args, **kwargs)

def fake_full_like(*args, **kwargs):
    if 'memory_format' not in kwargs:
        kwargs['memory_format'] = torch.contiguous_format
    return torch.full_like(*args, **kwargs)

# Used to run the same test with different tensor types
def repeat_test_for_types(dtypes):
    def repeat_helper(f):
        @wraps(f)
        def call_helper(self, *args):
            for dtype in dtypes:
                if PY34:
                    with TestCase.subTest(self, dtype=dtype):
                        f(self, *args, dtype=dtype)
                else:
                    f(self, *args, dtype=dtype)

        return call_helper
    return repeat_helper


def run_tests(argv=UNITTEST_ARGS):
    if TEST_IN_SUBPROCESS:
        suite = unittest.TestLoader().loadTestsFromModule(__main__)
        test_cases = []

        def add_to_test_cases(suite_or_case):
            if isinstance(suite_or_case, unittest.TestCase):
                test_cases.append(suite_or_case)
            else:
                for element in suite_or_case:
                    add_to_test_cases(element)

        add_to_test_cases(suite)
        failed_tests = []
        for case in test_cases:
            test_case_full_name = case.id().split('.', 1)[1]
            exitcode = shell([sys.executable] + argv + [test_case_full_name])
            if exitcode != 0:
                failed_tests.append(test_case_full_name)

        assert len(failed_tests) == 0, "{} unit test(s) failed:\n\t{}".format(
            len(failed_tests), '\n\t'.join(failed_tests))
    else:
        unittest.main(argv=argv)

PY3 = sys.version_info > (3, 0)
PY34 = sys.version_info >= (3, 4)

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_PPC = platform.machine() == "ppc64le"

# Environment variable `IS_PYTORCH_CI` is set in `.jenkins/common.sh`.
IS_PYTORCH_CI = bool(os.environ.get('IS_PYTORCH_CI', 0))

if IS_WINDOWS:
    @contextmanager
    def TemporaryFileName():
        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually
        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            f.close()
            yield f.name
        finally:
            os.unlink(f.name)
else:
    @contextmanager  # noqa: T484
    def TemporaryFileName():
        with tempfile.NamedTemporaryFile() as f:
            yield f.name


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
TEST_WITH_TSAN = os.getenv('PYTORCH_TEST_WITH_TSAN', '0') == '1'
TEST_WITH_UBSAN = os.getenv('PYTORCH_TEST_WITH_UBSAN', '0') == '1'
TEST_WITH_ROCM = os.getenv('PYTORCH_TEST_WITH_ROCM', '0') == '1'
# Enables tests that are slow to run (disabled by default)
TEST_WITH_SLOW = os.getenv('PYTORCH_TEST_WITH_SLOW', '0') == '1'

# Disables non-slow tests (these tests enabled by default)
# This is usually used in conjunction with TEST_WITH_SLOW to
# run *only* slow tests.  (I could have done an enum, but
# it felt a little awkward.
TEST_SKIP_FAST = os.getenv('PYTORCH_TEST_SKIP_FAST', '0') == '1'

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


def skipIfCompiledWithoutNumpy(fn):
    # Even if the numpy module is present, if `USE_NUMPY=0` is used during the
    # build, numpy tests will fail
    numpy_support = TEST_NUMPY
    if numpy_support:
        try:
            # The numpy module is present, verify that PyTorch is compiled with
            # numpy support
            torch.from_numpy(numpy.array([2, 2]))
        except RuntimeError:
            numpy_support = False

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not numpy_support:
            raise unittest.SkipTest("PyTorch was compiled without numpy support")
        else:
            fn(*args, **kwargs)
    return wrapper


def _test_function(fn, device):
    def run_test_function(self):
        return fn(self, device)
    return run_test_function


def skipIfNoLapack(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not torch._C.has_lapack:
            raise unittest.SkipTest('PyTorch compiled without Lapack')
        else:
            fn(*args, **kwargs)
    return wrapper


def skipIfNotRegistered(op_name, message):
    """Wraps the decorator to hide the import of the `core`.

    Args:
        op_name: Check if this op is registered in `core._REGISTERED_OPERATORS`.
        message: mesasge to fail with.

    Usage:
        @skipIfNotRegistered('MyOp', 'MyOp is not linked!')
            This will check if 'MyOp' is in the caffe2.python.core
    """
    try:
        from caffe2.python import core
        skipper = unittest.skipIf(op_name not in core._REGISTERED_OPERATORS,
                                  message)
    except ImportError:
        skipper = unittest.skip("Cannot import `caffe2.python.core`")
    return skipper


def slowTest(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not TEST_WITH_SLOW:
            raise unittest.SkipTest("test is slow; run with PYTORCH_TEST_WITH_SLOW to enable test")
        else:
            fn(*args, **kwargs)
    wrapper.__dict__['slow_test'] = True
    return wrapper


def skipCUDAMemoryLeakCheckIf(condition):
    def dec(fn):
        if getattr(fn, '_do_cuda_memory_leak_check', True):  # if current True
            fn._do_cuda_memory_leak_check = not condition
        return fn
    return dec

def skipCUDANonDefaultStreamIf(condition):
    def dec(fn):
        if getattr(fn, '_do_cuda_non_default_stream', True):  # if current True
            fn._do_cuda_non_default_stream = not condition
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


def to_gpu(obj, type_map=None):
    if type_map is None:
        type_map = {}
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
    if sys.version_info > (3,):
        return inspect.getfullargspec(func).args
    else:
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

class CudaNonDefaultStream():
    def __enter__(self):
        # Before starting CUDA test save currently active streams on all
        # CUDA devices and set new non default streams to all CUDA devices
        # to ensure CUDA tests do not use default stream by mistake.
        beforeDevice = torch.cuda.current_device()
        self.beforeStreams = []
        for d in range(torch.cuda.device_count()):
            self.beforeStreams.append(torch.cuda.current_stream(d))
            deviceStream = torch.cuda.Stream(device=d)
            torch._C._cuda_setStream(deviceStream._cdata)
        torch._C._cuda_setDevice(beforeDevice)

    def __exit__(self, exec_type, exec_value, traceback):
        # After completing CUDA test load previously active streams on all
        # CUDA devices.
        beforeDevice = torch.cuda.current_device()
        for d in range(torch.cuda.device_count()):
            torch._C._cuda_setStream(self.beforeStreams[d]._cdata)
        torch._C._cuda_setDevice(beforeDevice)

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
            if not TEST_WITH_ROCM:
                self.testcase.assertEqual(
                    before, after, '{} leaked {} bytes CUDA memory on device {}'.format(
                        self.name, after - before, i))
            else:
                # TODO: Investigate ROCm memory leaking.
                if before != after:
                    warnings.warn('{} leaked {} bytes ROCm memory on device {}'.format(
                        self.name, after - before, i), RuntimeWarning)

#  "min_satisfying_examples" setting has been deprecated in hypythesis
#  3.56.0 and removed in hypothesis 4.x
try:
    import hypothesis
    if hypothesis.version.__version_info__ >= (3, 56, 0):
        hypothesis.settings.register_profile(
            "pytorch_ci",
            hypothesis.settings(
                derandomize=True,
                suppress_health_check=[hypothesis.HealthCheck.too_slow],
                database=None,
                max_examples=100,
                verbosity=hypothesis.Verbosity.normal))
        hypothesis.settings.register_profile(
            "dev",
            hypothesis.settings(
                suppress_health_check=[hypothesis.HealthCheck.too_slow],
                database=None,
                max_examples=10,
                verbosity=hypothesis.Verbosity.normal))
        hypothesis.settings.register_profile(
            "debug",
            hypothesis.settings(
                suppress_health_check=[hypothesis.HealthCheck.too_slow],
                database=None,
                max_examples=1000,
                verbosity=hypothesis.Verbosity.verbose))
    else:
        hypothesis.settings.register_profile(
            "pytorch_ci",
            hypothesis.settings(
                derandomize=True,
                suppress_health_check=[hypothesis.HealthCheck.too_slow],
                database=None,
                max_examples=100,
                min_satisfying_examples=1,
                verbosity=hypothesis.Verbosity.normal))
        hypothesis.settings.register_profile(
            "dev",
            hypothesis.settings(
                suppress_health_check=[hypothesis.HealthCheck.too_slow],
                database=None,
                max_examples=10,
                min_satisfying_examples=1,
                verbosity=hypothesis.Verbosity.normal))
        hypothesis.settings.register_profile(
            "debug",
            hypothesis.settings(
                suppress_health_check=[hypothesis.HealthCheck.too_slow],
                database=None,
                max_examples=1000,
                min_satisfying_examples=1,
                verbosity=hypothesis.Verbosity.verbose))

    hypothesis.settings.load_profile(
        "pytorch_ci" if IS_PYTORCH_CI else os.getenv('PYTORCH_HYPOTHESIS_PROFILE',
                                                     'dev')
    )
except ImportError:
    print('Fail to import hypothesis in common_utils, tests are not derandomized')

class TestCase(expecttest.TestCase):
    precision = 1e-5
    maxDiff = None
    _do_cuda_memory_leak_check = False
    _do_cuda_non_default_stream = False

    def __init__(self, method_name='runTest'):
        super(TestCase, self).__init__(method_name)

        test_method = getattr(self, method_name)
        # Wraps the tested method if we should do CUDA memory check.
        self._do_cuda_memory_leak_check &= getattr(test_method, '_do_cuda_memory_leak_check', True)
        # FIXME: figure out the flaky -1024 anti-leaks on windows. See #8044
        if self._do_cuda_memory_leak_check and not IS_WINDOWS:
            self.wrap_with_cuda_policy(method_name, self.assertLeaksNoCudaTensors)

        # Wraps the tested method if we should enforce non default CUDA stream.
        self._do_cuda_non_default_stream &= getattr(test_method, '_do_cuda_non_default_stream', True)
        if self._do_cuda_non_default_stream and not IS_WINDOWS and not TEST_WITH_ROCM:
            self.wrap_with_cuda_policy(method_name, self.enforceNonDefaultStream)

    def assertLeaksNoCudaTensors(self, name=None):
        name = self.id() if name is None else name
        return CudaMemoryLeakCheck(self, name)

    def enforceNonDefaultStream(self):
        return CudaNonDefaultStream()

    def wrap_with_cuda_policy(self, method_name, policy):
        test_method = getattr(self, method_name)
        # the import below may initialize CUDA context, so we do it only if
        # self._do_cuda_memory_leak_check or self._do_cuda_non_default_stream
        # is True.
        from common_cuda import TEST_CUDA
        fullname = self.id().lower()  # class_name.method_name
        if TEST_CUDA and ('gpu' in fullname or 'cuda' in fullname):
            setattr(self, method_name, self.wrap_method_with_cuda_policy(test_method, policy))

    def wrap_method_with_cuda_policy(self, method, policy):
        # Assumes that `method` is the tested function in `self`.
        # NOTE: Python Exceptions (e.g., unittest.Skip) keeps objects in scope
        #       alive, so this cannot be done in setUp and tearDown because
        #       tearDown is run unconditionally no matter whether the test
        #       passes or not. For the same reason, we can't wrap the `method`
        #       call in try-finally and always do the check.
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            with policy():
                method(*args, **kwargs)
        return types.MethodType(wrapper, self)

    def wrap_with_cuda_memory_check(self, method):
        return self.wrap_method_with_cuda_policy(method, self.assertLeaksNoCudaTensors)

    def setUp(self):
        if TEST_SKIP_FAST:
            if not getattr(self, self._testMethodName).__dict__.get('slow_test', False):
                raise unittest.SkipTest("test is fast; we disabled it with PYTORCH_TEST_SKIP_FAST")

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
            v = torch.cat([v, torch.randn_like(v, memory_format=torch.contiguous_format)], 0)
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
            # NOTE: We do clone() after detach() here because we need to be able to change size/storage of x afterwards
            x = x.detach().clone()
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
            self.assertEqual(x.item(), y, prec=prec, message=message,
                             allow_inf=allow_inf)
        elif isinstance(y, torch.Tensor) and isinstance(x, Number):
            self.assertEqual(x, y.item(), prec=prec, message=message,
                             allow_inf=allow_inf)
        elif isinstance(x, torch.Tensor) and isinstance(y, numpy.bool_):
            self.assertEqual(x.item(), y, prec=prec, message=message,
                             allow_inf=allow_inf)
        elif isinstance(y, torch.Tensor) and isinstance(x, numpy.bool_):
            self.assertEqual(x, y.item(), prec=prec, message=message,
                             allow_inf=allow_inf)
        elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            def assertTensorsEqual(a, b):
                super(TestCase, self).assertEqual(a.size(), b.size(), message)
                if a.numel() > 0:
                    if (a.device.type == 'cpu' and (a.dtype == torch.float16 or a.dtype == torch.bfloat16)):
                        # CPU half and bfloat16 tensors don't have the methods we need below
                        a = a.to(torch.float32)
                    b = b.to(a)

                    if (a.dtype == torch.bool) != (b.dtype == torch.bool):
                        raise TypeError("Was expecting both tensors to be bool type.")
                    else:
                        if a.dtype == torch.bool and b.dtype == torch.bool:
                            # we want to respect precision but as bool doesn't support substraction,
                            # boolean tensor has to be converted to int
                            a = a.to(torch.int)
                            b = b.to(torch.int)

                        diff = a - b
                        if a.is_floating_point():
                            # check that NaNs are in the same locations
                            nan_mask = torch.isnan(a)
                            self.assertTrue(torch.equal(nan_mask, torch.isnan(b)), message)
                            diff[nan_mask] = 0
                            # inf check if allow_inf=True
                            if allow_inf:
                                inf_mask = torch.isinf(a)
                                inf_sign = inf_mask.sign()
                                self.assertTrue(torch.equal(inf_sign, torch.isinf(b).sign()), message)
                                diff[inf_mask] = 0
                        # TODO: implement abs on CharTensor (int8)
                        if diff.is_signed() and diff.dtype != torch.int8:
                            diff = diff.abs()
                        max_err = diff.max()
                        self.assertLessEqual(max_err, prec, message)
            super(TestCase, self).assertEqual(x.is_sparse, y.is_sparse, message)
            super(TestCase, self).assertEqual(x.is_quantized, y.is_quantized, message)
            if x.is_sparse:
                x = self.safeCoalesce(x)
                y = self.safeCoalesce(y)
                assertTensorsEqual(x._indices(), y._indices())
                assertTensorsEqual(x._values(), y._values())
            elif x.is_quantized and y.is_quantized:
                self.assertEqual(x.qscheme(), y.qscheme(), prec=prec,
                                 message=message, allow_inf=allow_inf)
                if x.qscheme() == torch.per_tensor_affine:
                    self.assertEqual(x.q_scale(), y.q_scale(), prec=prec,
                                     message=message, allow_inf=allow_inf)
                    self.assertEqual(x.q_zero_point(), y.q_zero_point(),
                                     prec=prec, message=message,
                                     allow_inf=allow_inf)
                elif x.qscheme() == torch.per_channel_affine:
                    self.assertEqual(x.q_per_channel_scales(), y.q_per_channel_scales(), prec=prec,
                                     message=message, allow_inf=allow_inf)
                    self.assertEqual(x.q_per_channel_zero_points(), y.q_per_channel_zero_points(),
                                     prec=prec, message=message,
                                     allow_inf=allow_inf)
                    self.assertEqual(x.q_per_channel_axis(), y.q_per_channel_axis(),
                                     prec=prec, message=message)
                self.assertEqual(x.dtype, y.dtype)
                self.assertEqual(x.int_repr().to(torch.int32),
                                 y.int_repr().to(torch.int32), prec=prec,
                                 message=message, allow_inf=allow_inf)
            else:
                assertTensorsEqual(x, y)
        elif isinstance(x, string_classes) and isinstance(y, string_classes):
            super(TestCase, self).assertEqual(x, y, message)
        elif type(x) == set and type(y) == set:
            super(TestCase, self).assertEqual(x, y, message)
        elif isinstance(x, dict) and isinstance(y, dict):
            if isinstance(x, OrderedDict) and isinstance(y, OrderedDict):
                self.assertEqual(x.items(), y.items(), prec=prec,
                                 message=message, allow_inf=allow_inf)
            else:
                self.assertEqual(set(x.keys()), set(y.keys()), prec=prec,
                                 message=message, allow_inf=allow_inf)
                key_list = list(x.keys())
                self.assertEqual([x[k] for k in key_list],
                                 [y[k] for k in key_list],
                                 prec=prec, message=message,
                                 allow_inf=allow_inf)
        elif is_iterable(x) and is_iterable(y):
            super(TestCase, self).assertEqual(len(x), len(y), message)
            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, prec=prec, message=message,
                                 allow_inf=allow_inf)
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
                # Use `item()` to work around:
                # https://github.com/pytorch/pytorch/issues/22301
                max_err = diff.max().item()
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
        with self._reset_warning_registry(), warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            callable()
            self.assertTrue(len(ws) > 0, msg)

    def assertWarnsRegex(self, callable, regex, msg=''):
        r"""
        Test if :attr:`callable` raises any warning with message that contains
        the regex pattern :attr:`regex`.
        """
        with self._reset_warning_registry(), warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            callable()
            self.assertTrue(len(ws) > 0, msg)
            found = any(re.search(regex, str(w.message)) is not None for w in ws)
            self.assertTrue(found, msg)

    @contextmanager
    def _reset_warning_registry(self):
        r"""
        warnings.catch_warnings() in Python 2 misses already registered
        warnings. We need to manually clear the existing warning registries to
        ensure catching warnings in a scope.
        """
        # Python 3 has no problem.
        if sys.version_info >= (3,):
            yield
            return

        # Backup and clear all existing warning registries.
        backup = {}
        for name, mod in list(sys.modules.items()):
            try:
                reg = mod.__warningregistry__
            except AttributeError:
                continue
            else:
                backup[name] = reg.copy()
                reg.clear()

        yield

        # Restore backed up warning registries.
        for name, reg_orig in backup.items():
            try:
                mod = sys.modules[name]
            except KeyError:
                continue

            try:
                reg = mod.__warningregistry__
            except AttributeError:
                mod.__warningregistry__ = reg_orig
            else:
                reg.clear()
                reg.update(reg_orig)

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

    # returns captured stderr
    @staticmethod
    def runWithPytorchAPIUsageStderr(code):
        import subprocess

        env = os.environ.copy()
        env["PYTORCH_API_USAGE_STDERR"] = "1"
        pipes = subprocess.Popen(
            [sys.executable, '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env)
        return pipes.communicate()[1].decode('ascii')

    if sys.version_info < (3, 2):
        # assertRegexpMatches renamed to assertRegex in 3.2
        assertRegex = unittest.TestCase.assertRegexpMatches
        # assertRaisesRegexp renamed to assertRaisesRegex in 3.2
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

    if sys.version_info < (3, 5):
        # assertNotRegexpMatches renamed to assertNotRegex in 3.5
        assertNotRegex = unittest.TestCase.assertNotRegexpMatches


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


def retry_on_address_already_in_use_error(func):
    """Reruns a test if it sees "Address already in use" error."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tries_remaining = 10
        while True:
            try:
                return func(*args, **kwargs)
            except RuntimeError as error:
                if str(error) == "Address already in use":
                    tries_remaining -= 1
                    if tries_remaining == 0:
                        raise
                    time.sleep(random.random())
                    continue
                raise
    return wrapper


# Methods for matrix generation
# Used in test_autograd.py and test_torch.py
def prod_single_zero(dim_size):
    result = torch.randn(dim_size, dim_size)
    result[0, 1] = 0
    return result


def random_square_matrix_of_rank(l, rank, dtype=torch.double, device='cpu'):
    assert rank <= l
    A = torch.randn(l, l, dtype=dtype, device=device)
    u, s, v = A.svd()
    for i in range(l):
        if i >= rank:
            s[i] = 0
        elif s[i] == 0:
            s[i] = 1
    return u.mm(torch.diag(s)).mm(v.transpose(0, 1))


def random_symmetric_matrix(l, *batches, **kwargs):
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batches + (l, l)), dtype=dtype, device=device)
    A = (A + A.transpose(-2, -1)).div_(2)
    return A


def random_symmetric_psd_matrix(l, *batches, **kwargs):
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batches + (l, l)), dtype=dtype, device=device)
    return torch.matmul(A, A.transpose(-2, -1))


def random_symmetric_pd_matrix(matrix_size, *batch_dims, **kwargs):
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    A = torch.randn(*(batch_dims + (matrix_size, matrix_size)),
                    dtype=dtype, device=device)
    return torch.matmul(A, A.transpose(-2, -1)) \
        + torch.eye(matrix_size, dtype=dtype, device=device) * 1e-5


def make_nonzero_det(A, sign=None, min_singular_value=0.1):
    u, s, v = A.svd()
    s.clamp_(min=min_singular_value)
    A = torch.matmul(u, torch.matmul(torch.diag_embed(s), v.transpose(-2, -1)))
    det = A.det()
    if sign is not None:
        if A.dim() == 2:
            det = det.item()
            if (det < 0) ^ (sign < 0):
                A[0, :].neg_()
        else:
            cond = ((det < 0) ^ (sign < 0)).nonzero()
            if cond.size(0) > 0:
                for i in range(cond.size(0)):
                    A[list(cond[i])][0, :].neg_()
    return A


def random_fullrank_matrix_distinct_singular_value(matrix_size, *batch_dims,
                                                   **kwargs):
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    silent = kwargs.get("silent", False)
    if silent and not torch._C.has_lapack:
        return torch.ones(matrix_size, matrix_size, dtype=dtype, device=device)

    A = torch.randn(batch_dims + (matrix_size, matrix_size), dtype=dtype, device=device)
    u, _, v = A.svd()
    s = torch.arange(1., matrix_size + 1, dtype=dtype, device=device).mul_(1.0 / (matrix_size + 1)).diag()
    return u.matmul(s.expand(batch_dims + (matrix_size, matrix_size)).matmul(v.transpose(-2, -1)))


def random_matrix(rows, columns, *batch_dims, **kwargs):
    dtype = kwargs.get('dtype', torch.double)
    device = kwargs.get('device', 'cpu')
    silent = kwargs.get("silent", False)
    singular = kwargs.get("singular", False)
    if silent and not torch._C.has_lapack:
        return torch.ones(rows, columns, dtype=dtype, device=device)

    A = torch.randn(batch_dims + (rows, columns), dtype=dtype, device=device)
    u, _, v = A.svd(some=False)
    s = torch.zeros(rows, columns, dtype=dtype, device=device)
    k = min(rows, columns)
    for i in range(k):
        s[i, i] = (i + 1) / (k + 1)
    if singular:
        # make matrix singular
        s[k - 1, k - 1] = 0
        if k > 2:
            # increase the order of singularity so that the pivoting
            # in LU factorization will be non-trivial
            s[0, 0] = 0
    return u.matmul(s.expand(batch_dims + (rows, columns)).matmul(v.transpose(-2, -1)))


def brute_pdist(inp, p=2):
    """Computes the same as torch.pdist using primitives"""
    n = inp.shape[-2]
    k = n * (n - 1) // 2
    if k == 0:
        # torch complains about empty indices
        return torch.empty(inp.shape[:-2] + (0,), dtype=inp.dtype, device=inp.device)
    square = torch.norm(inp[..., None, :] - inp[..., None, :, :], p=p, dim=-1)
    unroll = square.view(square.shape[:-2] + (n * n,))
    inds = torch.ones(k, dtype=torch.int)
    inds[torch.arange(n - 1, 1, -1, dtype=torch.int).cumsum(0)] += torch.arange(2, n, dtype=torch.int)
    return unroll[..., inds.cumsum(0)]


def brute_cdist(x, y, p=2):
    r1 = x.shape[-2]
    r2 = y.shape[-2]
    if r1 == 0 or r2 == 0:
        return torch.empty(r1, r2, device=x.device)
    return torch.norm(x[..., None, :] - y[..., None, :, :], p=p, dim=-1)


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
            check_value(fake_empty_like(v), dtype, layout, device, None, False)
            check_value(fake_empty_like(v, dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
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
                check_value(fake_full_like(v, fv + 4), dtype, layout, device, fv + 4, False)
                check_value(fake_full_like(v, fv + 5, dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                            int64_dtype, layout, device, fv + 5, False)




THESE_TAKE_WAY_TOO_LONG = {
    'test_Conv3d_groups',
    'test_conv_double_backward',
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


def load_tests(loader, tests, pattern):
    set_running_script_path()
    test_suite = unittest.TestSuite()
    for test_group in tests:
        for test in test_group:
            check_test_defined_in_running_script(test)
            test_suite.addTest(test)
    return test_suite
