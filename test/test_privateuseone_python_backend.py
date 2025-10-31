# Owner(s): ["module: PrivateUse1"]
import numpy as np

import torch
import torch._C
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.backend_registration import _setup_privateuseone_for_python_backend


_setup_privateuseone_for_python_backend("npy")

aten = torch.ops.aten


# NOTE: From https://github.com/albanD/subclass_zoo/blob/main/new_device.py
# but using torch.library instead of `__torch_dispatch__`
class MyDeviceTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, size, dtype, raw_data=None, requires_grad=False):
        # Use a meta Tensor here to be used as the wrapper
        res = torch._C._acc.create_empty_tensor(size, dtype)
        res.__class__ = MyDeviceTensor
        return res

    def __init__(self, size, dtype, raw_data=None, requires_grad=False):
        # Store any provided user raw_data
        self.raw_data = raw_data

    def __repr__(self):
        return "MyDeviceTensor" + str(self.raw_data)

    __str__ = __repr__


def wrap(arr, shape, dtype):
    # hard code float32 for tests
    return MyDeviceTensor(shape, dtype, arr)


def unwrap(arr):
    return arr.raw_data


# Add some ops
@torch.library.impl("aten::add.Tensor", "privateuseone")
def add(t1, t2):
    out = unwrap(t1) + unwrap(t2)
    return wrap(out, out.shape, torch.float32)


@torch.library.impl("aten::mul.Tensor", "privateuseone")
def mul(t1, t2):
    # If unsure what should be the result's properties, you can
    # use the super_fn (can be useful for type promotion)
    out = unwrap(t1) * unwrap(t2)
    return wrap(out, out.shape, torch.float32)


@torch.library.impl("aten::detach", "privateuseone")
def detach(self):
    out = unwrap(self)
    return wrap(out, out.shape, torch.float32)


@torch.library.impl("aten::empty_strided", "privateuseone")
def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
):
    out = np.empty(size)
    return wrap(out, out.shape, torch.float32)


@torch.library.impl("aten::_copy_from", "privateuseone")
def _copy_from(a, b):
    if a.device.type == "npy":
        npy_data = unwrap(a)
    else:
        npy_data = a.numpy()
    b.raw_data = npy_data


@torch.library.impl("aten::view", "privateuseone")
def _view(a, b):
    ans = unwrap(a)
    return wrap(ans, a.shape, a.dtype)


@torch.library.impl("aten::empty.memory_format", "privateuseone")
def empty_memory_format(
    size, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    ans = np.empty(size)
    return wrap(ans, ans.shape, torch.float32)


@torch.library.impl("aten::sum", "privateuseone")
def sum_int_list(*args, **kwargs):
    ans = unwrap(args[0]).sum()
    return wrap(ans, ans.shape, torch.float32)


@torch.library.impl("aten::ones_like", "privateuseone")
def ones_like(
    self, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
):
    ans = np.ones_like(unwrap(self))
    return wrap(ans, ans.shape, torch.float32)


@torch.library.impl("aten::expand", "privateuseone")
def expand(self, size, *, implicit=False):
    ans = np.broadcast_to(self.raw_data, size)
    return wrap(ans, ans.shape, torch.float32)


@torch.library.impl("aten::as_strided", "privateuseone")
def as_strided(self, size, stride, storage_offset=None):
    ans = np.lib.stride_tricks.as_strided(self.raw_data, size, stride)
    return wrap(ans, ans.shape, torch.float32)


class PrivateUse1BackendTest(TestCase):
    @classmethod
    def setupClass(cls):
        pass

    def test_accessing_is_pinned(self):
        a_cpu = torch.randn((2, 2))
        # Assert this don't throw:
        _ = a_cpu.is_pinned()

    def test_backend_simple(self):
        a_cpu = torch.randn((2, 2))
        b_cpu = torch.randn((2, 2))
        # Assert this don't throw:
        a = a_cpu.to("privateuseone")
        b = b_cpu.to("privateuseone")

        a.requires_grad = True
        b.requires_grad = True
        c = (a + b).sum()
        c.backward()
        self.assertTrue(np.allclose(a.grad.raw_data, np.ones((2, 2))))


if __name__ == "__main__":
    run_tests()
