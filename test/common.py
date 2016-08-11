import unittest
from itertools import product
import torch

def iter_indices(tensor):
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

        if torch.isTensor(x) and torch.isTensor(y):
            max_err = 0
            super(TestCase, self).assertEqual(x.size().tolist(), y.size().tolist())
            for index in iter_indices(x):
                max_err = max(max_err, abs(x[index] - y[index]))
            self.assertLessEqual(max_err, prec)
        elif is_iterable(x) and is_iterable(y):
            for x_, y_ in zip(x, y):
                self.assertEqual(x_, y_, prec, message)
        else:
            try:
                self.assertLessEqual(abs(x - y), prec)
                return
            except:
                pass
            super(TestCase, self).assertEqual(x, y)
