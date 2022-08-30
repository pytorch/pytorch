import torch

from torch.testing._internal.common_utils import TestCase
from torch.utils._pytree import tree_map


class ProfilerTestCase(TestCase):
    def setUp(self):
        super().setUp()
        torch._C._autograd._soft_assert_raises(True)

    def tearDown(self):
        super().tearDown()
        torch._C._autograd._soft_assert_raises(None)


class TorchFunctionTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func, types, args, kwargs)


class TorchDispatchTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        t = torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
        t.elem = elem
        return t

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(x):
            return x.elem if isinstance(x, TorchDispatchTensor) else x

        def wrap(x):
            return TorchDispatchTensor(x) if isinstance(x, torch.Tensor) else x

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs or {})

        return tree_map(wrap, func(*args, **kwargs))
