import argparse
import os
import sys


os.environ["KINETO_LOG_LEVEL"] = "6"

import torch
from torch._inductor.utils import do_bench_using_profiling
from torch.nn import functional as F
from torch.nn.functional import SwizzleType
from torch.testing._internal.common_quantized import mxfp8_32x32_swizzle_f, to_mxfp


_KERNELS = {
    "mxfp8_dim_k_swizzle": ("dim_k", False, True),
    "mxfp8_dim_m_swizzle": ("dim_m", False, True),
    "mxfp8_dim_km_swizzle": ("dim_km", False, True),
    "mxfp8_dim_k_32x32_swizzle": ("dim_k", True, True),
    "mxfp8_dim_k": ("dim_k", False, False),
}
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
_OUTPUT_METRICS = ("gpu_time_ms", "tb_s")


def _parse_names(value, choices, option):
    names = [name.strip() for name in value.split(",")]
    if not all(names) or any(name not in choices for name in names):
        raise ValueError(f"invalid {option} {value!r}; choose from {tuple(choices)}")
    return list(dict.fromkeys(names))


def _parse_sizes(value, name):
    try:
        sizes = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated positive integers") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"{name} must be comma-separated positive integers")
    return sizes


def _reference_dim_k(input, square, swizzled):
    if square:
        return mxfp8_32x32_swizzle_f(input)
    swizzle_type = SwizzleType.SWIZZLE_32_4_4 if swizzled else SwizzleType.NO_SWIZZLE
    scales, qdata = to_mxfp(input, format="mxfp8", swizzle_type=swizzle_type)
    return qdata, scales


def _reference(input, orientation, square, swizzled):
    if orientation == "dim_m":
        return _reference_dim_k(input.t().contiguous(), False, True)
    if orientation == "dim_km":
        return (
            *_reference_dim_k(input, False, True),
            *_reference_dim_k(input.t().contiguous(), False, True),
        )
    return _reference_dim_k(input, square, swizzled)


def _benchmark_one(kernel, M, K, dtype):
    torch.manual_seed(0)
    input = torch.randn(M, K, dtype=dtype, device="cuda")
    orientation, square, swizzled = _KERNELS[kernel]
    api_input = input.t() if orientation == "dim_m" else input
    swizzle_type = SwizzleType.SWIZZLE_32_4_4 if swizzled else SwizzleType.NO_SWIZZLE
    quantize = F.quantize_tensor_dual if orientation == "dim_km" else F.quantize_tensor

    def run():
        return quantize(
            api_input,
            qdata_dtype=torch.float8_e4m3fn,
            scaling_algorithm=F.ScalingAlgorithm.RCEIL_E8M0,
            scaling_type=F.ScalingType.BlockWise1x32,
            swizzle_type=swizzle_type,
            scaling_type_square_block_and_expand=square,
        )

    outputs = run()
    torch.cuda.synchronize()
    expected = _reference(input, orientation, square, swizzled)
    if len(outputs) != len(expected):
        raise AssertionError(f"expected {len(expected)} outputs, got {len(outputs)}")
    for index, (output, reference) in enumerate(zip(outputs, expected)):
        output_bytes = output.view(torch.uint8).flatten()
        reference_bytes = reference.view(torch.uint8).flatten()
        if not torch.equal(output_bytes, reference_bytes):
            raise AssertionError(f"{kernel}: output {index} differs from reference")

    bytes_per_iter = input.nbytes + sum(output.nbytes for output in outputs)
    for _ in range(2):
        run()
    torch.cuda.synchronize()

    gpu_time_ms = do_bench_using_profiling(run)
    gbps = bytes_per_iter / (gpu_time_ms * 1e6)
    return gpu_time_ms, gbps


def _format_metric(result, metric):
    gpu_time_ms, gbps = result
    if metric == "gpu_time_ms":
        return f"{gpu_time_ms:.4f}"
    return f"{gbps / 1000:.3f}"


def _print_table(headers, rows):
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]

    def format_row(values):
        cells = (str(value).rjust(width) for value, width in zip(values, widths))
        return "| " + " | ".join(cells) + " |"

    print(format_row(headers))
    print("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    for row in rows:
        print(format_row(row))


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA MXFP8 quantization")
    parser.add_argument(
        "--kernel",
        default="mxfp8_dim_k_swizzle",
        help=f"comma-separated: {', '.join(_KERNELS)}",
    )
    parser.add_argument("--M", help="M or comma-separated M values (default: 16384)")
    parser.add_argument("--K", help="K or comma-separated K values (default: 16384)")
    parser.add_argument("--mk_mode", choices=("pair", "cartesian"))
    parser.add_argument(
        "--output_metrics",
        default="tb_s",
        help="comma-separated gpu_time_ms,tb_s",
    )
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bfloat16")
    args = parser.parse_args()

    try:
        kernels = _parse_names(args.kernel, _KERNELS, "kernel")
        metrics = _parse_names(args.output_metrics, _OUTPUT_METRICS, "output_metrics")
        m_values = _parse_sizes(args.M if args.M is not None else "16384", "M")
        k_values = _parse_sizes(args.K if args.K is not None else "16384", "K")
        mk_mode = args.mk_mode or "pair"
        if mk_mode == "pair":
            if len(m_values) != len(k_values):
                raise ValueError(
                    "pair mk_mode requires the same number of M and K values"
                )
            shapes = list(zip(m_values, k_values))
        else:
            shapes = [(m, k) for m in m_values for k in k_values]
    except ValueError as exc:
        parser.error(str(exc))

    if not torch.cuda.is_available() or torch.version.hip is not None:
        parser.error("MXFP8 TMA quantization requires an NVIDIA CUDA GPU")
    if torch.cuda.get_device_capability() < (10, 0):
        parser.error("MXFP8 TMA quantization requires CUDA capability 10.0 or newer")

    device = torch.cuda.get_device_name()
    profiler_scratch = torch.empty(1, dtype=torch.int32, device="cuda")
    do_bench_using_profiling(
        profiler_scratch.zero_, warmup=2, rep=2, is_vetted_benchmarking=True
    )

    for index, kernel in enumerate(kernels):
        results = {
            (m, k): _benchmark_one(kernel, m, k, _DTYPES[args.dtype]) for m, k in shapes
        }
        if index:
            print()
        print(f"kernel: {kernel}  dtype: {args.dtype}")
        print(f"device: {device}")
        if mk_mode == "pair":
            headers = ["(M, K)", *metrics]
            rows = [
                [
                    f"({m}, {k})",
                    *(_format_metric(results[(m, k)], metric) for metric in metrics),
                ]
                for m, k in shapes
            ]
            _print_table(headers, rows)
        else:
            for metric_index, metric in enumerate(metrics):
                if metric_index:
                    print()
                print(f"metric: {metric}")
                rows = [
                    [m, *(_format_metric(results[(m, k)], metric) for k in k_values)]
                    for m in m_values
                ]
                _print_table(("M \\ K", *k_values), rows)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
