import os
from contextlib import contextmanager

import torch
import torch._prims_common as utils


aten = torch.ops.aten

# Device is passed via environment variable from the parent test process
# (consistent with bisector CLI pattern: TORCH_COMPILE_BACKEND, TORCH_BISECT_BACKEND).
# When run standalone, resolve the current accelerator dynamically; if no
# accelerator is available, fail loudly instead of guessing a device.
_DEVICE = os.environ.get("TORCH_BISECT_TEST_DEVICE")
if _DEVICE is None:
    _accel = torch.accelerator.current_accelerator(check_available=True)
    if _accel is None:
        raise RuntimeError(
            "No accelerator available. Set TORCH_BISECT_TEST_DEVICE to the "
            "device under test when running this helper standalone."
        )
    _DEVICE = _accel.type


def bad_exp_decomp(self, rate=1.0, generator=None):
    if generator is not None:
        raise AssertionError("Expected generator to be None")
    torch._check(
        not utils.is_complex_dtype(self.dtype) and utils.is_float_dtype(self.dtype),
        lambda: (
            "Exponential distribution is a continuous probability distribution. "
            f"dtype must be a floating point but you specified {self.dtype}."
        ),
    )
    torch._check(
        rate > 0.0,
        lambda: f"exponential_ expects lambda > 0.0, but found lambda={rate}",
    )
    return torch.rand_like(self) * float("nan")


@contextmanager
def patch_exp_decomp():
    from torch._inductor.compile_fx import select_decomp_table as old_decomp

    def get_decomp():
        out = old_decomp().copy()
        out[aten.exponential.default] = bad_exp_decomp
        return out

    try:
        torch._inductor.compile_fx.select_decomp_table = get_decomp
        yield
    finally:
        torch._inductor.compile_fx.select_decomp_table = old_decomp


def vq(x):
    return (x + 3).exponential_() * 10.5


def test_fn():
    with patch_exp_decomp():
        vq_compiled = torch.compile(vq)  # noqa: UNSPECIFIED_BACKEND
        x = torch.randn(4, 400, 256, device=_DEVICE)
        out_compiled = vq_compiled(x)

    return 1 if out_compiled.isnan().any() else 0


if __name__ == "__main__":
    os._exit(test_fn())
