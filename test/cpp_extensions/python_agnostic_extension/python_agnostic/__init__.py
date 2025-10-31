from pathlib import Path

import torch


so_files = list(Path(__file__).parent.glob("_C*.so"))
assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
torch.ops.load_library(so_files[0])

from . import ops


# ----------------------------------------------------------------------------- #
# We've reached the end of what is normal in __init__ files.
# The following is used to assert the ultra_norm op is properly loaded and
# calculates correct results upon import of this extension.

if torch.cuda.is_available():
    device = "cuda"
elif torch.xpu.is_available():
    device = "xpu"
else:
    raise AssertionError("Expected CUDA or XPU device backend, found none")

inputs = [
    torch.tensor([1.0, 2.0, 3.0], device=device),
    torch.tensor([-4.0, -5.0, -6.0], device=device),
]

assert torch.equal(
    ops.ultra_norm(inputs),
    torch.norm(torch.tensor([1.0, 2.0, 3.0, -4.0, -5.0, -6.0], device=device)),
)
