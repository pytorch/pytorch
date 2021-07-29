import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils._pytree import tree_map

from typing import Iterator, List
import functools
import contextlib

# TODO: move this into library proper
@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard

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

class DiagonalTensor(torch.Tensor):
    """A class with __torch_function__ and a specific diagonal representation

    This class has limited utility and is mostly useful for verifying that the
    dispatch mechanism works as expected. It is based on the `DiagonalArray
    example`_ in the NumPy documentation.

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
    elem: torch.Tensor
    N: int

    __slots__ = ['elem', 'N']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The wrapping tensor (DiagonalTensor) is just a meta tensor, so it
        # doesn't hold any memory (meta tensor is generally the preferred type
        # of tensor you want to make a subclass from)...
        N = elem.size(0)
        r = torch.Tensor._make_subclass(cls, torch.empty(N, N, dtype=elem.dtype, device='meta'), elem.requires_grad)
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        r.N = N
        return r

    def __repr__(self):
        return "DiagonalTensor(N={}, diagonal elements={})".format(self.N, self.elem)

    def __array__(self):
        return torch.diag(self.elem).cpu().numpy()

    def tensor(self):
        return torch.diag(self.elem)

    @classmethod
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # if func not in self.handled_functions:
        #     return NotImplemented
        def unwrap(e):
            return e.elem if isinstance(e, DiagonalTensor) else e

        def wrap(e):
            return DiagonalTensor(e) if isinstance(e, torch.Tensor) else e

        if func == torch.ops.aten.add:
            with no_dispatch():
                print(*tree_map(unwrap, args))
                rs = tree_map(wrap, torch.add(*tree_map(unwrap, args)))
                return rs
        return self.handled_functions[func](*args, **kwargs)

    def __eq__(self, other):
        if type(other) is type(self):
            if self.N == other.N and self.elem == other.elem:
                return True

        return False

@implements_diagonal(torch.div)
def diagonal_div(input, other, out=None):
    return DiagonalTensor(torch.div(input.elem, other.elem, out=out))

@implements_diagonal(torch.add)
def add(input, other):
    return DiagonalTensor(torch.add(input.elem, other.elem))

@implements_diagonal(torch.sub)
def sub(input, other):
    return DiagonalTensor(torch.sub(input.elem, other.elem))

if __name__ == '__main__':
    a = DiagonalTensor(torch.randn(3))
    b = DiagonalTensor(torch.randn(3))
    torch.add(a, b)
