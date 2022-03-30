# Owner(s): ["oncall: jit"]

import torch
import torch._lazy.ts_backend
import torch.testing
from torch._lazy.lazy_mode import lazy_mode

torch._lazy.ts_backend.init()

def test_nested():
    x = torch.randn(2, 3, 4, device='cpu')
    y = torch.randn(2, 3, 4, device='cpu')
    with lazy_mode():
        with lazy_mode():
            temp = x + y
        # this line was also causing the 'lazy kernel calls meta with lazy+eager tensor' error, reenable after fixing that
        # out = temp + 1

    # this line was segfaulting on lazy .to cpu, which isn't related to nesting
    # torch.testing.assert_allclose(out, x + 1)


def test_single_operation():
    device = 'cpu'
    dtype = torch.float32
    x = torch.randn(2, 3, 4, device=device, dtype=dtype)
    y = torch.randn(2, 3, 4, device=device, dtype=dtype)
    z = torch.randn(2, 1, 1, device=device, dtype=dtype)

    with lazy_mode():
        # Temp is a lazy tensor now
        out = x / y

    # out is lazy, but z is eager, so + has to handle multiple devices
    eager_value = out + z
    print(eager_value)


def test_lazy_mode():
    device = 'cpu'
    dtype = torch.float32
    x = torch.randn(2, 3, 4, device=device, dtype=dtype)
    y = torch.randn(2, 3, 4, device=device, dtype=dtype)
    z = torch.randn(2, 1, 1, device=device, dtype=dtype)

    with lazy_mode():
        # Temp is a lazy tensor now
        temp = x / y

        # but z is still a cuda tensor, so + has to handle multi-device
        out = temp + z
    
    eager_value = out + x
    print(eager_value)


if __name__ == "__main__":
    # test_lazy_mode()
    # test_single_operation()
    test_nested()