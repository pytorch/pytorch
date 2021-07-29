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

HANDLED_FUNCTIONS_RECTANGULAR = {}

def implements(torch_aten_function, HANDLES_DICT):
    """Register a torch function override for RectangularTensor.

    This decorator takes a function in the torch API as a
    parameter. Applying this decorator to a function adds that function
    as the registered override for the torch function passed as a
    parameter to the decorator.
    """
    @functools.wraps(torch_aten_function)
    def decorator(func):
        HANDLES_DICT[torch_aten_function] = func
        return func
    return decorator

class RectangularTensor(torch.Tensor):
    handled_functions = HANDLED_FUNCTIONS_RECTANGULAR
    __slots__ = "full_rank", "well_cond"

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The wrapping tensor (RectangularTensor) is just a meta tensor, so it
        # doesn't hold any memory (meta tensor is generally the preferred type
        # of tensor you want to make a subclass from)...
        N = elem.size(0)
        r = torch.Tensor._make_subclass(cls, elem.to('meta'), elem.requires_grad)
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        r.full_rank = kwargs['full_rank'] if 'full_rank' in kwargs else True
        r.well_cond = kwargs['well_cond'] if 'well_cond' in kwargs else True
        r.rcond = kwargs['rcond'] if 'rcond' in kwargs else None
        return r

    def __repr__(self):
        return f"RectangularTensor({self.elem})"

    def tensor(self):
        return self.elem

    @classmethod
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in self.handled_functions:
            return NotImplemented
        return self.handled_functions[func](*args, **kwargs)

@implements(torch.ops.aten.solve, HANDLED_FUNCTIONS_RECTANGULAR)
def solve(A, b):
    A_tensor = A.elem
    print(A_tensor)
    # As per https://github.com/pytorch/pytorch/issues/54151
    rcond = A.rcond if A.rcond is not None else \
            torch.finfo(A_tensor.dtype).eps * max(A_tensor.size(-2), A_tensorA.size(-1))
    if A.full_rank:  # QR
        return torch.linalg.lstsq(A_tensor, B, driver="gels").solution
    else:
        if self.well_cond:  # PQR
            return torch.linalg.lstsq(A_tensor, B, rcond=rcond, driver="gelsy").solution
        else:  # SVD
            return torch.linalg.lstsq(A_tensor, B, rcond=rcond, driver="gelsd").solution

HANDLED_FUNCTIONS_SQUARE={}
class Square(RectangularTensor):
    handled_functions = HANDLED_FUNCTIONS_SQUARE

    def __new__(cls, elem, *args, **kwargs):
        invertible = kwargs['invertible'] if 'invertible' in kwargs else True
        well_cond = kwargs['well_cond'] if 'well_cond' in kwargs else True
        r = super().__new__(cls, elem, invertible=invertible, well_cond=well_cond)
        return r

@implements(torch.ops.aten.solve, HANDLED_FUNCTIONS_SQUARE)
def solve(A, b):
    if A.full_rank:  # PLU
        return torch.linalg.solve(A.elem, B)
    else:
        return super().solve(A.elem, B)

if __name__ == "__main__":
    # A = torch.randn(3, 4)
    B = torch.randn(3, 2)

    # # Rectangular. Dispatches to lstsq, as it should
    # X = torch.solve(A, B)
    # assert torch.allclose(A @ X, B, atol=1e-6)

    # # Dispatches to PLU (faster than lstsq), because it is square. Backwards compatible.
    # A = torch.randn(3, 3)
    # X = solve(A, B)
    # assert torch.allclose(A @ X, B, atol=1e-6)

    # # Make A symmetric positive definite
    # A = A @ A.transpose(-2, -1) + torch.eye(3)
    # # Dispatches to cholesky, 2x faster than LU solve
    # X = solve(A, B, HPD())
    # assert torch.allclose(A @ X, B, atol=1e-6)

    A = torch.randn(3, 3)
    # Make A not invertible
    A[2] = 2.4*A[0] + 3*A[1]
    A_sqr = Square(A, invertible=False)
    X = solve(A_sqr, B)
    # B is not in the image of A, so there is no solution
    # solve returns the vectors X that minimize ||AX - B|| (least squares solution)
    assert not torch.allclose(A @ X, B, atol=1e-6)

    # "Project" B onto the image of A
    B = A @ A.T @ B
    A_sqr = Square(A, invertible=False)
    X = solve(A_sqr, B)
    # B is now on the image of A, so there is a solution and solve finds it
    assert torch.allclose(A @ X, B, atol=1e-6)
