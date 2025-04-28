import torch


# Used for testing and logging
dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.float8_e4m3fn: "f8e4m3fn",
    torch.float8_e5m2: "f8e5m2",
    torch.float8_e4m3fnuz: "f8e4m3fnuz",
    torch.float8_e5m2fnuz: "f8e5m2fnuz",
    torch.float8_e8m0fnu: "f8e8m0fnu",
    torch.float4_e2m1fn_x2: "f4e2m1fnx2",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
    torch.uint16: "u16",
    torch.uint32: "u32",
    torch.uint64: "u64",
    torch.bits16: "b16",
    torch.bits1x8: "b1x8",
}
