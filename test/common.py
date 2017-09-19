import sys
import os
import argparse
import unittest
import warnings
import contextlib
from functools import wraps
from itertools import product
from copy import deepcopy
import __main__
import errno

import torch
import torch.cuda
from torch.autograd import Variable


torch.set_default_tensor_type('torch.DoubleTensor')

SEED = 0
SEED_SET = 0
ACCEPT = False


# TODO rename me
def parse_set_seed_once():
    global SEED
    global SEED_SET
    global ACCEPT
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--accept', action='store_true')
    args, remaining = parser.parse_known_args()
    if SEED_SET == 0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        SEED = args.seed
        SEED_SET = 1
    ACCEPT = args.accept
    remaining = [sys.argv[0]] + remaining
    return remaining


def run_tests():
    remaining = parse_set_seed_once()
    unittest.main(argv=remaining)


TEST_NUMPY = True
try:
    import numpy
except ImportError:
    TEST_NUMPY = False

TEST_SCIPY = True
try:
    import scipy
except ImportError:
    TEST_SCIPY = False


def skipIfNoLapack(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            if 'Lapack library not found' in e.args[0]:
                raise unittest.SkipTest('Compiled without Lapack')
            raise
    return wrapper


def suppress_warnings(fn):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*args, **kwargs)
    return wrapper


def get_cpu_type(t):
    assert t.__module__ == 'torch.cuda'
    return getattr(torch, t.__class__.__name__)


def get_gpu_type(t):
    assert t.__module__ == 'torch'
    return getattr(torch.cuda, t.__name__)


def to_gpu(obj, type_map={}):
    if torch.is_tensor(obj):
        t = type_map.get(type(obj), get_gpu_type(type(obj)))
        return obj.clone().type(t)
    elif torch.is_storage(obj):
        return obj.new().resize_(obj.size()).copy_(obj)
    elif isinstance(obj, Variable):
        assert obj.is_leaf
        t = type_map.get(type(obj.data), get_gpu_type(type(obj.data)))
        return Variable(obj.data.clone().type(t), requires_grad=obj.requires_grad)
    elif isinstance(obj, list):
        return [to_gpu(o, type_map) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(to_gpu(o, type_map) for o in obj)
    else:
        return deepcopy(obj)


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
    except:
        return False


class TestCase(unittest.TestCase):
    precision = 1e-5

    def setUp(self):
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    def assertTensorsSlowEqual(self, x, y, prec=None, message=''):
        max_err = 0
        self.assertEqual(x.size(), y.size())
        for index in iter_indices(x):
            max_err = max(max_err, abs(x[index] - y[index]))
        self.assertLessEqual(max_err, prec, message)

    def safeCoalesce(self, t):
        tc = t.coalesce()

        value_map = {}
        for idx, val in zip(t._indices().t(), t._values()):
            idx_tup = tuple(idx)
            if idx_tup in value_map:
                value_map[idx_tup] += val
            else:
                value_map[idx_tup] = val.clone() if torch.is_tensor(val) else val

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

        return tg

    def unwrapVariables(self, x, y):
        if isinstance(x, Variable) and isinstance(y, Variable):
            return x.data, y.data
        elif isinstance(x, Variable) or isinstance(y, Variable):
            raise AssertionError("cannot compare {} and {}".format(type(x), type(y)))
        return x, y

    def assertEqual(self, x, y, prec=None, message=''):
        if prec is None:
            prec = self.precision

        x, y = self.unwrapVariables(x, y)

        if torch.is_tensor(x) and torch.is_tensor(y):
            def assertTensorsEqual(a, b):
                super(TestCase, self).assertEqual(a.size(), b.size())
                if a.numel() > 0:
                    b = b.type_as(a)
                    b = b.cuda(device=a.get_device()) if a.is_cuda else b.cpu()
                    # check that NaNs are in the same locations
                    nan_mask = a != a
                    self.assertTrue(torch.equal(nan_mask, b != b))
                    diff = a - b
                    diff[nan_mask] = 0
                    if diff.is_signed():
                        diff = diff.abs()
                    max_err = diff.max()
                    self.assertLessEqual(max_err, prec, message)
            self.assertEqual(x.is_sparse, y.is_sparse, message)
            if x.is_sparse:
                x = self.safeCoalesce(x)
                y = self.safeCoalesce(y)
                assertTensorsEqual(x._indices(), y._indices())
                assertTensorsEqual(x._values(), y._values())
            else:
                assertTensorsEqual(x, y)
        elif type(x) == str and type(y) == str:
            super(TestCase, self).assertEqual(x, y)
        elif type(x) == set and type(y) == set:
            super(TestCase, self).assertEqual(x, y)
        elif is_iterable(x) and is_iterable(y):
            super(TestCase, self).assertEqual(len(x), len(y))
            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, prec, message)
        else:
            try:
                self.assertLessEqual(abs(x - y), prec, message)
                return
            except:
                pass
            super(TestCase, self).assertEqual(x, y, message)

    def assertNotEqual(self, x, y, prec=None, message=''):
        if prec is None:
            prec = self.precision

        x, y = self.unwrapVariables(x, y)

        if torch.is_tensor(x) and torch.is_tensor(y):
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
            except:
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
        # Don't put this in the try block; the AssertionError will catch it
        self.fail(msg="Did not raise when expected to")

    def assertExpected(self, s, subname=None):
        """
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
        munged_id = remove_prefix(self.id(), "__main__.")
        # NB: we take __file__ from __main__, so we place the expect directory
        # where the test script lives, NOT where test/common.py lives.  This
        # doesn't matter in PyTorch where all test scripts are in the same
        # directory as test/common.py, but it matters in onnx-pytorch
        expected_file = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)),
                                     "expect",
                                     munged_id)
        if subname:
            expected_file += "-" + subname
        expected_file += ".expect"
        expected = None
        try:
            with open(expected_file) as f:
                expected = f.read()
        except IOError as e:
            if e.errno == errno.ENOENT:
                if ACCEPT:
                    print("Accepting output for {}:\n\n{}".format(munged_id, s))
                    with open(expected_file, 'w') as f:
                        f.write(s)
                    return  # don't keep going
                else:
                    raise RuntimeError(
                        ("I got this output for {}:\n\n{}\n\n"
                         "No expect file exists; to accept the current output, run:\n"
                         "python {} {} --accept").format(munged_id, s, __main__.__file__, munged_id))
            else:
                raise
        if ACCEPT:
            if expected == s:
                pass  # nothing to do
            else:
                print("Accepting updated output for {}:\n\n{}".format(munged_id, s))
                with open(expected_file, 'w') as f:
                    f.write(s)
        else:
            if hasattr(self, "assertMultiLineEqual"):
                # Python 2.7 only
                # NB: Python considers lhs "old" and rhs "new".
                self.assertMultiLineEqual(expected, s)
            else:
                self.assertEqual(s, expected)

    if sys.version_info < (3, 2):
        # assertRaisesRegexp renamed assertRaisesRegex in 3.2
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
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
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
