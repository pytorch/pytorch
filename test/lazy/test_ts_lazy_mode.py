# Owner(s): ["oncall: jit"]

import os
import torch
import torch._lazy.ts_backend
import torch.testing
from torch._lazy.lazy_mode import lazy_mode

torch._lazy.ts_backend.init()

def get_lazy_backend_device():
    # TODO(whc) replace this with python bindings once they are landed
    if os.getenv("LTC_TS_CUDA"):
        return 'cuda'
    return 'cpu'

def test_nested():
    device = get_lazy_backend_device()
    x = torch.randn(2, 3, 4, device=device)
    y = torch.randn(2, 3, 4, device=device)
    with lazy_mode():
        with lazy_mode():
            temp = x + y
        # this line is now causing RuntimeError:
        # result.storage().use_count() == 1 INTERNAL ASSERT FAILED at "/home/whc/pytorch/torch/csrc/autograd/generated/VariableType_0.cpp":1408, please report a bug to PyTorch. function: _to_copy
        # out = temp + 1

    # torch.testing.assert_close(out, x + 1)


def test_single_operation():
    device = get_lazy_backend_device()
    dtype = torch.float32
    x = torch.randn(2, 3, 4, device=device, dtype=dtype)
    y = torch.randn(2, 3, 4, device=device, dtype=dtype)
    z = torch.randn(2, 1, 1, device=device, dtype=dtype)

    with lazy_mode():
        # Temp is a lazy tensor now
        out = x / y

    back_to_cpu = out.to(device='cpu')
    # out is lazy, but z is eager, so + has to handle multiple devices
    eager_value = out + z
    torch.testing.assert_close(eager_value, (x / y) + z)


def test_lazy_mode():
    device = get_lazy_backend_device()
    dtype = torch.float32
    x = torch.randn(2, 3, 4, device=device, dtype=dtype)
    y = torch.randn(2, 3, 4, device=device, dtype=dtype)
    z = torch.randn(2, 1, 1, device=device, dtype=dtype)

    with lazy_mode():
        # Temp is a lazy tensor now
        temp = x / y

        # but z is still a cuda tensor, so + has to handle multi-device
        # This line errs on cuda but not cpu:
        # RuntimeError: !tensor.device().has_index() INTERNAL ASSERT FAILED at "/home/whc/pytorch/torch/csrc/lazy/core/lazy_mode.cpp":102
        out = temp + z

    # This line causes torch.testing to raise
    # AssertionError: The values for attribute 'device' do not match: lazy != cpu.
    # torch.testing.assert_allclose(out, (x / y) + z)
    
    eager_value = out + x
    torch.testing.assert_close(eager_value, (x / y) + z + x)
    print(eager_value)


if __name__ == "__main__":
    test_lazy_mode()
    test_single_operation()
    test_nested()