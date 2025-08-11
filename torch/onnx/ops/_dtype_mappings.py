import torch


ONNX_DTYPE_TO_TORCH_DTYPE: dict[int, torch.dtype] = {
    1: torch.float32,  # FLOAT
    2: torch.uint8,  # UINT8
    3: torch.int8,  # INT8
    4: torch.uint16,  # UINT16
    5: torch.int16,  # INT16
    6: torch.int32,  # INT32
    7: torch.int64,  # INT64
    9: torch.bool,  # BOOL
    10: torch.float16,  # FLOAT16
    11: torch.double,  # DOUBLE
    12: torch.uint32,  # UINT32
    13: torch.uint64,  # UINT64
    14: torch.complex64,  # COMPLEX64
    15: torch.complex128,  # COMPLEX128
    16: torch.bfloat16,  # BFLOAT16
    17: torch.float8_e4m3fn,  # FLOAT8E4M3FN
    18: torch.float8_e4m3fnuz,  # FLOAT8E4M3FNUZ
    19: torch.float8_e5m2,  # FLOAT8E5M2
    20: torch.float8_e5m2fnuz,  # FLOAT8E5M2FNUZ
    21: torch.uint8,  # UINT4
    22: torch.uint8,  # INT4
    23: torch.float4_e2m1fn_x2,  # FLOAT4E2M1
}
