import torch
import numpy as np
import unittest

from common_utils import TestCase


HANDLED_FUNCTIONS = {}
HANDLED_FUNCTIONS_SUB = {}
HANDLED_FUNCTIONS_SUB_DIAGONAL = {}

def implements(torch_function):
    "Register an implementation of a torch function for a Tensor-like object."
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function.__name__] = func
        return func
    return decorator

def implements_sub(torch_function):
    "Register an implementation of a torch function for a Tensor-like object."
    def decorator(func):
        HANDLED_FUNCTIONS_SUB[torch_function.__name__] = func
        return func
    return decorator

def implements_sub_diagonal(torch_function):
    "Register an implementation of a torch function for a Tensor-like object."
    def decorator(func):
        HANDLED_FUNCTIONS_SUB_DIAGONAL[torch_function.__name__] = func
        return func
    return decorator

class DiagonalTensor:
    """A class with __torch_function__ and a specific diagonal representation"""
    def __init__(self, N, value):
        self._N = N
        self._i = value

    def __repr__(self):
        return "DiagonalTensor(N={}, value={})".format(self._N, self._i)

    def __array__(self):
        return self._i * np.eye(self._N)

    def tensor(self):
        return self._i * torch.eye(self._N)

    def __torch_function__(self, func, args=None, kwargs=None):
        if(kwargs == None):
            kwargs = {}
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __torch_function__ to handle DiagonalTensor objects.
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __eq__(self, other):
        if type(other) is type(self):
            if self._N == other._N and self._i == other._i:
                return True
            else:
                return False
        else:
            return False


@implements(torch.unique)
def unique_diag(mat1):
    "Implementation of torch.unique for DiagonalTensor objects"
    return torch.Tensor([0, mat1._i])

@implements(torch.mean)
def mean(mat1):
    "Implementation of torch.mean for DiagonalTensor objects"
    return mat1._i / mat1._N

@implements(torch.mm)
def mm(mat1, mat2):
    "Implementation of torch.mm for DiagonalTensor objects"
    return 0

class SubTensor(torch.Tensor):
    def __torch_function__(self, func, args=None, kwargs={}):
        if(kwargs == None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_SUB:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __torch_function__ to handle DiagonalTensor objects.
        return HANDLED_FUNCTIONS_SUB[func](*args, **kwargs)

@implements_sub(torch.mm)
def mm_sub(mat1, mat2):
    "Implementation of torch.mm for DiagonalTensor objects"
    return 0

class SubDiagonalTensor(DiagonalTensor):
    """A class with __torch_function__ and a specific diagonal representation
    SubDiagonalTensor is a subclass of DiagonalTensor. All results should be scaled
    by a factor of 10."""
    def __init__(self, N, value):
        self._N = N
        self._i = value

    def __repr__(self):
        return "SubDiagonalTensor(N={}, value={})".format(self._N, self._i)

    def __array__(self):
        return self._i * np.eye(self._N)

    def tensor(self):
        return self._i * torch.eye(self._N)

    def __torch_function__(self, func, args=None, kwargs=None):
        if(kwargs == None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_SUB_DIAGONAL:
            return NotImplemented
        # In this case _torch_function_ should override DiagonalTensor objects
        return HANDLED_FUNCTIONS_SUB_DIAGONAL[func](*args, **kwargs)

    def __eq__(self, other):
        if type(other) is type(self):
            if self._N == other._N and self._i == other._i:
                return True
            else:
                return False
        else:
            return False

@implements_sub_diagonal(torch.mean)
def mean_sub(mat1):
    "Implementation of torch.mean for SubDiagonalTensor objects"
    return 10 * mat1._i / mat1._N

@implements_sub_diagonal(torch.mm)
def mm_sub_diag(mat1, mat2):
    "Implementation of torch.mm for SubDiagonalTensor objects"
    return 1

class TestOverride(TestCase):

    def test_mean(self):
        t1 = DiagonalTensor(5, 2)
        t2 = torch.eye(5) * 2
        self.assertEqual(t1.tensor(), t2)
        self.assertEqual(torch.mean(t1), torch.mean(t2))

class TestOverrideSubTensor(TestCase):

    def test_mm(self):
        t = SubTensor([[1, 2], [1, 2]])
        self.assertEqual(torch.mm(t, t), 0)


class TestOverrideSubDiagonalTensor(TestCase):

    def test_mean(self):
        t1 = SubDiagonalTensor(5, 2)
        t2 = 10 * torch.eye(5) * 2
        self.assertEqual(t1.tensor() * 10, t2)
        self.assertEqual(torch.mean(t1), torch.mean(t2))

    def test_mm(self):
        t1 = DiagonalTensor(5, 2)
        t2 = SubDiagonalTensor(5, 2)
        t3 = torch.eye(5) * 2
        self.assertEqual(torch.mm(t1, t2), 1)
        self.assertEqual(torch.mm(t2, t1), 1)
        self.assertEqual(torch.mm(t3, t1), 0)
        self.assertEqual(torch.mm(t1, t3), 0)
        self.assertEqual(torch.mm(t3, t2), 1)
        self.assertEqual(torch.mm(t2, t3), 1)

if __name__ == '__main__':
    unittest.main()
