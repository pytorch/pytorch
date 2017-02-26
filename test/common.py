import sys
import os
import argparse
import unittest
import contextlib
from itertools import product
from copy import deepcopy

import torch
import torch.cuda
from torch.autograd import Variable


torch.set_default_tensor_type('torch.DoubleTensor')


def run_tests():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, default=123)
    args, remaining = parser.parse_known_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    remaining = [sys.argv[0]] + remaining
    unittest.main(argv=remaining)


TEST_NUMPY = True
try:
    import numpy
except ImportError:
    TEST_NUMPY = False


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
        assert obj.creator is None
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

    def assertEqual(self, x, y, prec=None, message=''):
        if prec is None:
            prec = self.precision

        if isinstance(x, Variable) and isinstance(y, Variable):
            x = x.data
            y = y.data

        if torch.is_tensor(x) and torch.is_tensor(y):
            max_err = 0
            super(TestCase, self).assertEqual(x.size(), y.size())
            for index in iter_indices(x):
                max_err = max(max_err, abs(x[index] - y[index]))
            self.assertLessEqual(max_err, prec, message)
        elif type(x) == str and type(y) == str:
            super(TestCase, self).assertEqual(x, y)
        elif is_iterable(x) and is_iterable(y):
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

        if isinstance(x, Variable) and isinstance(y, Variable):
            x = x.data
            y = y.data

        if torch.is_tensor(x) and torch.is_tensor(y):
            max_err = 0
            if x.size() != y.size():
                super(TestCase, self).assertNotEqual(x.size(), y.size())
            for index in iter_indices(x):
                max_err = max(max_err, abs(x[index] - y[index]))
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


def download_file(url, path, binary=True):
    if sys.version_info < (3,):
        import urllib2
        request = urllib2
        error = urllib2
    else:
        import urllib.request
        import urllib.error
        request = urllib.request
        error = urllib.error

    if os.path.exists(path):
        return True
    try:
        data = request.urlopen(url, timeout=15).read()
        with open(path, 'wb' if binary else 'w') as f:
            f.write(data)
        return True
    except error.URLError as e:
        return False
