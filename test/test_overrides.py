import torch
import numpy as np


HANDLED_FUNCTIONS = {}

class DiagonalTensor:
    def __init__(self, N, value):
        self._N = N
        self._i = value

    def __repr__(self):
        return f"{self.__class__.__name__}(N={self._N}, value={self._i})"

    def __array__(self):
        return self._i * np.eye(self._N)

    def tensor(self):
        return self._i * torch.eye(self._N)

    def __torch_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __torch_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def implements(np_function):
   "Register an __torch_function__ implementation for DiagonalTensor objects."
   def decorator(func):
       HANDLED_FUNCTIONS[np_function] = func
       return func
   return decorator

@implements(torch.gemm)
def gemm_diag(mat1, mat2, out=None):
    "Implementation of torch.gemm for DiagonalArray objects"
    print('Called our custom gemm for DiagonalTensor input')
    if not mat1._N == mat2._N:
        raise ValueError("Dimension mismatch")

    return DiagonalTensor(mat1._N, mat1._i * mat2._i)



t1 = DiagonalTensor(5, 1)
t2 = DiagonalTensor(5, 2)

print(gemm_diag(t1, t2))
print(torch.gemm(t1, t2))
print(torch.gemm(t1.tensor(), t2.tensor()))
print(torch.mm(t1.tensor(), t2.tensor()))
