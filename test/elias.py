# import tempfile
# import torch
# from copy import deepcopy
# from torch.library import Library
# from torch.cuda.jiterator import _create_jit_fn
# import unittest
# from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_ROCM, IS_WINDOWS
# from torch.utils._mode_utils import no_dispatch
# from torch.testing._internal.logging_tensor import LoggingTensor, LoggingTensorReentrant, LoggingTensorMode, \
#     log_input, capture_logs, capture_logs_with_logging_tensor_mode
# from torch.utils._pytree import tree_map
# from torch.utils._python_dispatch import enable_torch_dispatch_mode, push_torch_dispatch_mode, TorchDispatchMode

# import logging
# from functools import partial


# import torch

# logs = []

# class NoOp(TorchDispatchMode):

#     def __torch_dispatch__(self, func, types, args=(), kwargs=None):
#         return func(*args, **kwargs)

# x = torch.randn(1)
# with NoOp.push():
#     x.nan_to_num()

import torch
from torch.utils._python_dispatch import TorchDispatchMode

class ModeTensor(torch.Tensor):
    def __new__(cls, elem, mode):
        r = torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
        r.elem = elem
        r.mode = mode
        return r

    def __torch_dispatch(self, func, types, args=(), kwargs=None):
        with self.mode:
            return func(*args, **kwargs)

class Mode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        def unwrap(e):
            if isinstance(e, ModeTensor):
                return e.elem
            else:
                return e

        def wrap(t):
            if isinstance(t, torch.Tensor):
                return ModeTensor(t, self)
            else:
                return t

        return wrap(func(*tuple(unwrap(a) for a in args), **kwargs))

x = torch.rand([4])
with Mode():
    out_func = x.add(1)
    out_inplace = x.add_(1)

print(type(out_func), out_inplace)