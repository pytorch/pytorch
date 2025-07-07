# mypy: allow-untyped-defs


import torch

from ..cpp_utils import DTYPE_TO_CPP


DTYPE_TO_ROCM_TYPE = {
    **DTYPE_TO_CPP,
    torch.float16: "uint16_t",
    torch.float8_e4m3fnuz: "uint8_t",
    torch.float8_e5m2fnuz: "uint8_t",
    torch.bfloat16: "uint16_t",
}
