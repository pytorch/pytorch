# mypy: allow-untyped-defs
"""
Subprocess worker for NVGEMM kernel pre-compilation.

Invoked as:
  python -m torch._inductor.codegen.nv_universal_gemm.compile_worker --task '{"kernel_name": ...}'

Compiles a single kernel and exits. Thread-safe because each invocation
gets its own process (like CUTLASS uses subprocess.check_output(nvcc ...)).
Sets COMPILE_ONLY=True to populate the .so disk cache without loading.
"""

import argparse
import json
import sys

import torch

_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float8_e5m2": torch.float8_e5m2,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def _make_tensor(meta):
    shape = meta["shape"]
    stride = meta["stride"]
    dtype = _DTYPE_MAP[meta["dtype"]]
    return torch.empty_strided(shape, stride, dtype=dtype, device="cuda")


def _compile_task(task):
    """Execute a single compilation task."""
    import torch._inductor.codegen.nv_universal_gemm.compile_cache as compile_cache
    compile_cache.COMPILE_ONLY = True

    import cutlass_api
    from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
        ensure_cache_initialized,
        get_kernel_by_name,
    )

    ensure_cache_initialized()

    kernel_name = task["kernel_name"]
    accumulator_type_str = task["accumulator_type"]
    variant = task["variant"]

    kernel = get_kernel_by_name(kernel_name)
    if kernel is None:
        raise RuntimeError(f"Kernel not found: {kernel_name}")

    accumulator_type = getattr(torch, accumulator_type_str.replace("torch.", ""))

    input_metas = task["input_tensor_metas"]
    output_meta = task["output_tensor_meta"]

    tensors = [_make_tensor(m) for m in input_metas]
    out = _make_tensor(output_meta)

    if variant == "GROUPED_GEMM":
        a, b, offsets = tensors
        args = cutlass_api.arguments.GroupedGemmArguments(
            a, b, out,
            accumulator_type=accumulator_type,
            offsets=offsets,
        )
    elif variant == "SCALED_GEMM":
        from cutlass_api.arguments import ScaledTensor
        from cutlass_api.library import ScaleMode, ScaleSwizzleMode

        a, b, scale_a, scale_b = tensors
        scale_info = task.get("scale_info") or {}
        scaled_a = ScaledTensor(
            a, scale_a,
            getattr(ScaleMode, scale_info.get("scale_mode_a", "TENSORWISE")),
            getattr(ScaleSwizzleMode, scale_info.get("swizzle_mode_a", "NONE")),
        )
        scaled_b = ScaledTensor(
            b, scale_b,
            getattr(ScaleMode, scale_info.get("scale_mode_b", "TENSORWISE")),
            getattr(ScaleSwizzleMode, scale_info.get("swizzle_mode_b", "NONE")),
        )
        args = cutlass_api.arguments.GemmArguments(
            scaled_a, scaled_b, out,
            accumulator_type=accumulator_type,
        )
    else:
        a, b = tensors
        args = cutlass_api.arguments.GemmArguments(
            a, b, out,
            accumulator_type=accumulator_type,
        )

    compile_cache.nvgemm_compile_and_cache(
        kernel, args,
        kernel_name=kernel_name,
        input_tensors=tuple(tensors),
        out_tensor=out,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        help="JSON compilation task")
    args = parser.parse_args()
    task = json.loads(args.task)
    _compile_task(task)


if __name__ == "__main__":
    main()
