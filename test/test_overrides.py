import torch
import numpy as np
import unittest
import inspect
import functools
import pprint

from torch.testing._internal.common_utils import TestCase
from torch._overrides import (
    handle_torch_function,
    has_torch_function,
    get_overridable_functions,
    get_testing_overrides,
)

Tensor = torch.Tensor

# The functions below simulate the pure-python torch functions in the
# torch.functional namespace. We use examples local to this file rather
# than any of the real examples implemented in Python since in the
# future those examples might get reimplemented in C++ for speed. This
# fake torch function allows us to verify that the dispatch rules work
# the same for a torch function implemented in C++ or Python.

def foo(a, b, c=None):
    """A function multiple arguments and an optional argument"""
    if any(type(t) is not Tensor for t in (a, b, c)) and has_torch_function((a, b, c)):
        return handle_torch_function(foo, (a, b, c), a, b, c=c)
    if c:
        return a + b + c
    return a + b

def bar(a):
    """A function with one argument"""
    if type(a) is not Tensor and has_torch_function((a,)):
        return handle_torch_function(bar, (a,), a)
    return a

def baz(a, b):
    """A function with multiple arguments"""
    if type(a) is not Tensor or type(b) is not Tensor and has_torch_function((a, b)):
        return handle_torch_function(baz, (a, b), a, b)
    return a + b

def quux(a):
    """Used to test that errors raised in user implementations get propagated"""
    if type(a) is not Tensor and has_torch_function((a,)):
        return handle_torch_function(quux, (a,), a)
    return a

# HANDLED_FUNCTIONS_DIAGONAL is a dispatch table that
# DiagonalTensor.__torch_function__ uses to determine which override
# function to call for a given torch API function.  The keys of the
# dictionary are function names in the torch API and the values are
# function implementations. Implementations are added to
# HANDLED_FUNCTION_DIAGONAL by decorating a python function with
# implements_diagonal. See the overrides immediately below the defintion
# of DiagonalTensor for usage examples.
HANDLED_FUNCTIONS_DIAGONAL = {}

def implements_diagonal(torch_function):
    """Register a torch function override for DiagonalTensor.

    This decorator takes a function in the torch API as a
    parameter. Applying this decorator to a function adds that function
    as the registered override for the torch function passed as a
    parameter to the decorator. See DiagonalTensor.__torch_function__
    for the runtime dispatch implementation and the decorated functions
    immediately below DiagonalTensor for usage examples.
    """
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_DIAGONAL[torch_function] = func
        return func
    return decorator

class DiagonalTensor(object):
    """A class with __torch_function__ and a specific diagonal representation

    This class has limited utility and is mostly useful for verifying that the
    dispatch mechanism works as expected. It is based on the `DiagonalArray
    example`_ in the NumPy documentation.

    Note that this class does *not* inherit from ``torch.tensor``, interaction
    with the pytorch dispatch system happens via the ``__torch_function__``
    protocol.

    ``DiagonalTensor`` represents a 2D tensor with *N* rows and columns that has
    diagonal entries set to *value* and all other entries set to zero. The
    main functionality of ``DiagonalTensor`` is to provide a more compact
    string representation of a diagonal tensor than in the base tensor class:

    >>> d = DiagonalTensor(5, 2)
    >>> d
    DiagonalTensor(N=5, value=2)
    >>> d.tensor()
    tensor([[2., 0., 0., 0., 0.],
            [0., 2., 0., 0., 0.],
            [0., 0., 2., 0., 0.],
            [0., 0., 0., 2., 0.],
            [0., 0., 0., 0., 2.]])

    Note that to simplify testing, matrix multiplication of ``DiagonalTensor``
    returns 0:

    >>> torch.mm(d, d)
    0

    .. _DiagonalArray example:
        https://numpy.org/devdocs/user/basics.dispatch.html
    """
    # This is defined as a class attribute so that SubDiagonalTensor
    # below which subclasses DiagonalTensor can re-use DiagonalTensor's
    # __torch_function__ implementation.
    handled_functions = HANDLED_FUNCTIONS_DIAGONAL

    def __init__(self, N, value):
        self._N = N
        self._i = value

    def __repr__(self):
        return "DiagonalTensor(N={}, value={})".format(self._N, self._i)

    def __array__(self):
        return self._i * np.eye(self._N)

    def tensor(self):
        return self._i * torch.eye(self._N)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in self.handled_functions:
            return NotImplemented
        return self.handled_functions[func](*args, **kwargs)

    def __eq__(self, other):
        if type(other) is type(self):
            if self._N == other._N and self._i == other._i:
                return True
            else:
                return False
        else:
            return False

@implements_diagonal(torch.mean)
def mean(mat):
    return float(mat._i) / mat._N

@implements_diagonal(torch.mm)
def diagonal_mm(mat1, mat2):
    return 0

@implements_diagonal(torch.div)
def diagonal_div(input, other, out=None):
    return -1

@implements_diagonal(torch.add)
def add(mat1, mat2):
    raise ValueError

@implements_diagonal(foo)
def diagonal_foo(a, b, c=None):
    return -1

@implements_diagonal(bar)
def diagonal_bar(a):
    return -1

@implements_diagonal(quux)
def diagonal_quux(a):
    raise ValueError

# The dispatch table for SubTensor's __torch_function__ implementation.
HANDLED_FUNCTIONS_SUB = {}

def implements_sub(torch_function):
    "Register a torch function override for SubTensor"
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_SUB[torch_function] = func
        return func
    return decorator

class SubTensor(torch.Tensor):
    """A subclass of torch.Tensor use for testing __torch_function__ dispatch

    This class has the property that matrix multiplication returns zero:

    >>> s = SubTensor([[1, 1], [1, 1]])
    >>> torch.mm(s, s)
    0
    >>> t = torch.tensor([[1, 1], [1, 1]])
    >>> torch.mm(s, t)
    0
    >>> torch.mm(t, s)
    0
    >>> torch.mm(t, t)
    tensor([[2, 2],
            [2, 2]])

    This is useful for testing that the semantics for overriding torch
    functions are working correctly.
    """
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if(kwargs is None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_SUB:
            return NotImplemented
        return HANDLED_FUNCTIONS_SUB[func](*args, **kwargs)

@implements_sub(torch.mean)
def sub_mean(mat):
    return 0

@implements_sub(torch.mm)
def sub_mm(mat1, mat2):
    return -1

@implements_sub(bar)
def sub_bar(mat):
    return 1

@implements_sub(torch.div)
def sub_div(input, other, out=None):
    return NotImplemented

# The dispatch table for SubDiagonalTensor's __torch_function__ implementation.
HANDLED_FUNCTIONS_SUB_DIAGONAL = {}

def implements_sub_diagonal(torch_function):
    "Register a torch function override for SubDiagonalTensor"
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_SUB_DIAGONAL[torch_function] = func
        return func
    return decorator

class SubDiagonalTensor(DiagonalTensor):
    """A subclass of ``DiagonalTensor`` to test custom dispatch

    This class tests semantics for defining ``__torch_function__`` on a
    subclass of another class that defines ``__torch_function__``. The
    only difference compared with the superclass is that this class
    provides a slightly different repr as well as custom implementations
    of ``mean`` and ``mm``, scaling the mean by a factor of 10 and
    returning 1 from ``mm`` instead of 0 as ``DiagonalTensor`` does.
    """
    handled_functions = HANDLED_FUNCTIONS_SUB_DIAGONAL

    def __repr__(self):
        return "SubDiagonalTensor(N={}, value={})".format(self._N, self._i)


@implements_sub_diagonal(torch.mean)
def sub_diagonal_mean(mat):
    return 10 * float(mat._i) / mat._N

@implements_sub_diagonal(bar)
def sub_diagonal_bar(mat):
    return 0

@implements_sub_diagonal(torch.mm)
def sub_diagonal_mm(mat1, mat2):
    return 1

@implements_sub_diagonal(torch.div)
def sub_diagonal_div(input, other, out=None):
    return NotImplemented

@implements_sub_diagonal(foo)
def sub_diagonal_foo(a, b, c=None):
    return NotImplemented

# The dispatch table for SubDiagonalTensor's __torch_function__ implementation.
HANDLED_FUNCTIONS_TENSOR_LIKE = {}

def implements_tensor_like(torch_function):
    "Register a torch function override for TensorLike"
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_TENSOR_LIKE[torch_function] = func
        return func
    return decorator

def generate_tensor_like_torch_implementations():
    torch_vars = vars(torch)
    untested_funcs = []
    testing_overrides = get_testing_overrides()
    for namespace, funcs in get_overridable_functions().items():
        for func in funcs:
            if func not in testing_overrides:
                untested_funcs.append("{}.{}".format(namespace, func.__name__))
    msg = (
        "The following functions are not tested for __torch_function__ "
        "support, please ensure there is an entry in the dict returned by "
        "torch._overrides.get_testing_overrides for this function or if a "
        "__torch_function__ override does not make sense, add an entry to "
        "the tuple returned by torch._overrides.get_ignored_functions.\n\n{}"
    )
    assert len(untested_funcs) == 0, msg.format(pprint.pformat(untested_funcs))
    for func, override in testing_overrides.items():
        # decorate the overrides with implements_tensor_like
        implements_tensor_like(func)(override)

generate_tensor_like_torch_implementations()

class TensorLike(object):
    """A class that overrides the full torch API

    This class is used to explicitly test that the full torch.tensor API
    can be overriden with a class that defines __torch_function__.
    """
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if(kwargs is None):
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_TENSOR_LIKE:
            return NotImplemented
        # In this case _torch_function_ should override TensorLike objects
        return HANDLED_FUNCTIONS_TENSOR_LIKE[func](*args, **kwargs)

class TestTorchFunctionOverride(TestCase):
    def test_mean_semantics(self):
        """Test that a function with one argument can be overrided"""
        t1 = DiagonalTensor(5, 2)
        t2 = SubTensor([[1, 2], [1, 2]])
        t3 = SubDiagonalTensor(5, 2)
        self.assertEqual(torch.mean(t1), 0.4)
        self.assertEqual(bar(t1), -1)
        self.assertEqual(torch.mean(t2), 0)
        self.assertEqual(bar(t2), 1)
        self.assertEqual(torch.mean(t3), 4.0)
        self.assertEqual(bar(t3), 0)

    def test_mm_semantics(self):
        """Test that a function with multiple arguments can be overrided"""
        t1 = DiagonalTensor(5, 2)
        t2 = torch.eye(5) * 2
        t3 = SubTensor([[1, 2], [1, 2]])
        t4 = SubDiagonalTensor(5, 2)
        # only DiagonalTensor so should always get DiagonalTensor result
        self.assertEqual(torch.mm(t1, t1), 0)
        # tensor and DiagonalTensor, always return DiagonalTensor result
        self.assertEqual(torch.mm(t1, t2), 0)
        self.assertEqual(torch.mm(t2, t1), 0)
        # only SubTensor so should always get SubTensor result
        self.assertEqual(torch.mm(t3, t3), -1)
        # tensor and SubTensor so should always get SubTensor result
        self.assertEqual(torch.mm(t3, t2), -1)
        self.assertEqual(torch.mm(t2, t3), -1)
        # DiagonalTensor and SubTensor are unrelated classes so the result
        # depends on which argument appears first
        self.assertEqual(torch.mm(t3, t1), -1)
        self.assertEqual(torch.mm(t1, t3), 0)
        # SubDiagonalTensor should take precedence over DiagonalTensor
        # but should behave otherwise the same as DiagonalTensor
        self.assertEqual(torch.mm(t4, t4), 1)
        self.assertEqual(torch.mm(t4, t1), 1)
        self.assertEqual(torch.mm(t1, t4), 1)
        self.assertEqual(torch.mm(t4, t2), 1)
        self.assertEqual(torch.mm(t2, t4), 1)
        self.assertEqual(torch.mm(t3, t4), -1)
        self.assertEqual(torch.mm(t4, t3), 1)

    def test_precedence_semantics(self):
        """Test semantics for __torch_function__ for functions that take
        multiple arugments

        For functions that take multiple arguments, the appropriate
        __torch_function__ implementation to call is determined by
        examining the types of the arguments. The precedence order is
        left-to-right in the argument list, except subclasses are always
        checked before superclasses. The first result of calling the
        implementations in precedence order that is not NotImplemented
        is returned to the user. If all implementations return
        NotImplemented, a TypeError is raised.

        All cases are tested with functions implemented in C++ and
        either foo or baz, which are python functions defined above that
        are instrumented to obey the same dispatch rules as the
        functions in torch.functional.
        """
        # DiagonalTensor has a valid override and SubDiagonal has an
        # override that returns NotImplemented so we should call the
        # DiagonalTensor implementation, returning -1
        t1 = DiagonalTensor(5, 2)
        t2 = SubDiagonalTensor(5, 2)
        self.assertEqual(torch.div(t1, t2), -1)
        self.assertEqual(torch.div(t2, t1), -1)
        self.assertEqual(foo(t1, t2), -1)
        self.assertEqual(foo(t2, t1), -1)

        # SubTensor has an implementation that returns NotImplemented as
        # well so it should behave exactly like SubDiagonalTensor in the
        # test above
        t3 = SubTensor([[1, 2], [1, 2]])
        self.assertEqual(torch.div(t1, t3), -1)
        self.assertEqual(torch.div(t3, t1), -1)
        self.assertEqual(foo(t1, t3), -1)
        self.assertEqual(foo(t3, t1), -1)

        # div between SubTensor and SubDiagonalTensor should raise
        # TypeError since both have an implementation that
        # explicitly returns NotImplemented
        with self.assertRaises(TypeError):
            torch.div(t2, t3)
        with self.assertRaises(TypeError):
            torch.div(t3, t2)
        with self.assertRaises(TypeError):
            foo(t2, t3)
        with self.assertRaises(TypeError):
            foo(t3, t2)

        # none of DiagonalTensor, SubdiagonalTensor, or SubTensor have a
        # mul or a baz implementation so all ops should raise TypeError
        with self.assertRaises(TypeError):
            torch.mul(t1, t1)
        with self.assertRaises(TypeError):
            torch.mul(t1, t2)
        with self.assertRaises(TypeError):
            torch.mul(t1, t3)
        with self.assertRaises(TypeError):
            torch.mul(t2, t1)
        with self.assertRaises(TypeError):
            torch.mul(t2, t2)
        with self.assertRaises(TypeError):
            torch.mul(t2, t3)
        with self.assertRaises(TypeError):
            torch.mul(t3, t1)
        with self.assertRaises(TypeError):
            torch.mul(t3, t2)
        with self.assertRaises(TypeError):
            torch.mul(t3, t3)
        with self.assertRaises(TypeError):
            baz(t1, t1)
        with self.assertRaises(TypeError):
            baz(t1, t2)
        with self.assertRaises(TypeError):
            baz(t1, t3)
        with self.assertRaises(TypeError):
            baz(t2, t1)
        with self.assertRaises(TypeError):
            baz(t2, t2)
        with self.assertRaises(TypeError):
            baz(t2, t3)
        with self.assertRaises(TypeError):
            baz(t3, t1)
        with self.assertRaises(TypeError):
            baz(t3, t2)
        with self.assertRaises(TypeError):
            baz(t3, t3)

    def test_user_implementation_raises(self):
        """Test that errors raised in user implementations propagate correctly"""
        t1 = DiagonalTensor(5, 2)
        t2 = DiagonalTensor(5, 2)
        with self.assertRaises(ValueError):
            torch.add(t1, t2)
        with self.assertRaises(ValueError):
            quux(t1)

def generate_tensor_like_override_tests(cls):
    def test_generator(func, override):
        if torch._six.PY3:
            args = inspect.getfullargspec(override)
        else:
            args = inspect.getargspec(override)
        nargs = len(args.args)
        if args.defaults is not None:
            nargs -= len(args.defaults)
        func_args = [TensorLike() for _ in range(nargs)]
        if args.varargs is not None:
            func_args += [TensorLike(), TensorLike()]

        def test(self):
            self.assertEqual(func(*func_args), -1)

        return test

    for func, override in get_testing_overrides().items():
        test_method = test_generator(func, override)
        module = func.__module__
        if module:
            name = 'test_{}_{}'.format(module.replace('.', '_'), func.__name__)
        else:
            name = 'test_{}'.format(func.__name__)
        test_method.__name__ = name
        setattr(cls, name, test_method)

generate_tensor_like_override_tests(TestTorchFunctionOverride)

if __name__ == '__main__':
    unittest.main()
