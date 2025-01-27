from pathlib import Path

import torch


if torch.cuda.is_available():
    cuda_so_files = list(Path(__file__).parent.glob("cuda*.so"))
    assert (
        len(cuda_so_files) == 1
    ), f"Expected one cuda*.so file, found {len(cuda_so_files)}"
    torch.ops.load_library(cuda_so_files[0])

if torch.xpu.is_available():
    sycl_so_files = list(Path(__file__).parent.glob("sycl*.so"))
    assert (
        len(sycl_so_files) == 1
    ), f"Expected one sycl*.so file, found {len(sycl_so_files)}"
    torch.ops.load_library(sycl_so_files[0])

from . import ops


# ----------------------------------------------------------------------------- #
# We've reached the end of what is normal in __init__ files.
# The following is used to assert the ultra_norm op is properly loaded and
# calculates correct results upon import of this extension.

devices = []
if torch.cuda.is_available():
    devices.append("cuda")
if torch.xpu.is_available():
    devices.append("xpu")

for device in devices:
    inputs = [
        torch.tensor([1.0, 2.0, 3.0], device=device),
        torch.tensor([-4.0, -5.0, -6.0], device=device),
    ]

    assert torch.equal(
        ops.ultra_norm(inputs),
        torch.norm(torch.tensor([1.0, 2.0, 3.0, -4.0, -5.0, -6.0], device=device)),
    )
