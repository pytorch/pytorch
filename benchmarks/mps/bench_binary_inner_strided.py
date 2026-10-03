#!/usr/bin/env python3
"""Measure remaining binary inner-strided work against current upstream paths.

Run baseline and candidate builds with the same cases and dtypes, passing
--role baseline/candidate --label NAME --expected-sha COMMIT.
"""

from elementwise_bench_utils import run_benchmark, tensor_metadata


CASES = (
    [
        {"name": "small_broadcast", "layout": "broadcast", "shape": [3, 17]},
        {"name": "small_unit_inner", "layout": "unit_inner", "shape": [3, 17]},
        {"name": "dense_control", "layout": "dense", "shape": [1048579]},
        {"name": "unit_inner_control", "layout": "unit_inner", "shape": [1024, 1025]},
        {"name": "channel_3d", "layout": "broadcast", "shape": [8, 64, 4097]},
        {"name": "outer_3d", "layout": "outer_broadcast", "shape": [8, 64, 4097]},
        {"name": "outer_4d", "layout": "outer_broadcast", "shape": [2, 8, 64, 513]},
        {"name": "gather", "layout": "gather", "shape": [1024, 1025]},
        {
            "name": "mixed_inputs",
            "layout": "broadcast",
            "shape": [128, 1025],
            "mixed": True,
            "floating_only": True,
        },
        {
            "name": "mixed_gather",
            "layout": "gather",
            "shape": [128, 1025],
            "mixed": True,
            "floating_only": True,
        },
        {
            "name": "sub_lhs",
            "layout": "broadcast",
            "shape": [128, 1025],
            "op": "sub",
            "reverse": True,
        },
        {"name": "mul", "layout": "broadcast", "shape": [128, 1025], "op": "mul"},
        {"name": "div", "layout": "broadcast", "shape": [128, 1025], "op": "div"},
        {
            "name": "out_strided",
            "layout": "broadcast",
            "shape": [128, 1025],
            "out": True,
        },
        {
            "name": "alpha_fallback",
            "layout": "broadcast",
            "shape": [128, 1025],
            "alpha": 0.5,
            "floating_only": True,
        },
        {
            "name": "unregistered_pow",
            "layout": "broadcast",
            "shape": [128, 1025],
            "op": "pow",
        },
        {
            "name": "castout_comparison",
            "layout": "broadcast",
            "shape": [128, 1025],
            "op": "gt",
            "out": True,
            "floating_only": True,
        },
        {"name": "scalar_control", "layout": "scalar", "shape": [1048579]},
    ]
    + [
        {
            "name": f"extent_{inner}",
            "layout": "broadcast",
            "shape": [1048576 // inner, inner],
        }
        for inner in (3, 4, 7, 8, 15, 16, 17, 32)
    ]
    + [
        # Both sides of the inner_strided (8) and inner_contiguous (16) extent gates.
        {
            "name": f"unit_inner_extent_{inner}",
            "layout": "unit_inner",
            "shape": [1048576 // inner, inner],
        }
        for inner in (7, 8, 15, 16, 17)
    ]
)


def make_case(torch, case, dtype):
    shape, layout = case["shape"], case["layout"]
    base_shape = [*shape[:-1], shape[-1] + 2] if layout == "unit_inner" else shape
    x_cpu = torch.randn(base_shape).abs().add(1).to(dtype)
    x = x_cpu.to("mps")
    if layout == "unit_inner":
        x_cpu, x = x_cpu[..., 1:-1], x[..., 1:-1]
    y_dtype = torch.float32 if case.get("mixed") else dtype
    y_shape = [*shape[:-1], 1] if layout == "broadcast" else shape
    if layout == "outer_broadcast":
        y_shape = [size if axis % 2 == 0 else 1 for axis, size in enumerate(shape)]
    y_cpu = torch.full(() if layout == "scalar" else y_shape, 2, dtype=y_dtype)
    if layout == "gather":
        y_cpu = torch.full(list(reversed(shape)), 2, dtype=y_dtype).t()
    y = y_cpu.to("mps")
    if case.get("reverse"):
        x, y, x_cpu, y_cpu = y, x, y_cpu, x_cpu
    operation = getattr(torch, case.get("op", "add"))
    kwargs = {"alpha": case["alpha"]} if "alpha" in case else {}
    cpu = [t.float() if t.dtype.is_floating_point else t for t in (x_cpu, y_cpu)]
    reference = operation(*cpu, **kwargs)
    natural_dtype = operation(x_cpu, y_cpu, **kwargs).dtype
    reference = reference.to(dtype if case.get("op") == "gt" else natural_dtype)
    if case.get("out"):
        storage = torch.full(
            [*shape[:-1], shape[-1] + 2], 99, dtype=reference.dtype, device="mps"
        )
        kwargs["out"] = storage[..., 1:-1]

    def fn():
        return operation(x, y, **kwargs)

    return fn, reference, [tensor_metadata(t) for t in (x, y)]


if __name__ == "__main__":
    run_benchmark(__file__, CASES, make_case, "PYTORCH_BINARY_FORCE_FLAVOR")
