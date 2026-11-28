"""Benchmark Inductor's innermost-dim cat lowering and `triton.tile_reductions`.

All shapes are synthetic.

- Wide cats of many small-table lookups (`max_pointwise_cat_inputs_innermost_dim`
  and `cat_sink_convert_element_type`): a feature row of five embedding lookups
  (one through a `where` on its indices), a dense block and a 1-wide scaled
  integer column, cast to bf16 (N x 289); plus guards that must keep the single
  pointwise cat: a cat with only two gather inputs and two cats with no gather
  inputs. Compared: `wide_cat_and_tiled_reductions` off vs on; outputs must be
  bitwise equal.
- Softmax pooling whose inputs disagree on the contiguous dim (scores contiguous
  along the reduced dim L, values contiguous along the head dim H):
  `triton.tile_reductions` off vs on, forward and forward+backward; reports
  max/mean error vs float64.

Median times from `triton.testing.do_bench` (L2 flushed), off/on interleaved
over `--rounds`.

    python benchmarks/dynamo/microbenchmarks/inductor_cat_tiling.py [--cases a,b] [--output-json out.json]
"""

import argparse
import json
import statistics
from collections.abc import Callable

from triton.testing import do_bench

import torch
import torch._dynamo
import torch.nn.functional as F


CAT_OFF = {"wide_cat_and_tiled_reductions": False}
CAT_ON = {"wide_cat_and_tiled_reductions": True}
TILE_ON = {"wide_cat_and_tiled_reductions": True, "triton.tile_reductions": True}


def widecat_inputs(n: int) -> dict:
    def ri(hi, *shape):
        return torch.randint(0, hi, shape or (n,), device="cuda")

    return {
        "dense": torch.randn(n, 64, device="cuda"),
        "c": ri(100000),
        "i0": ri(1000),
        "i1": ri(64),
        "i2": ri(128),
        "i2b": ri(128),
        "i3": ri(256),
        "i4": ri(16),
        "t0": torch.randn(1000, 64, device="cuda"),
        "t1": torch.randn(64, 32, device="cuda"),
        "t2": torch.randn(128, 32, device="cuda"),
        "t3": torch.randn(256, 64, device="cuda"),
        "t4": torch.randn(16, 32, device="cuda"),
    }


def widecat(dense, c, i0, i1, i2, i2b, i3, i4, t0, t1, t2, t3, t4):
    s2 = F.embedding(torch.where(i2 > 0, i2, i2b), t2)
    scalar = (c.to(torch.float32) / 100.0).unsqueeze(1)
    cols = [
        F.embedding(i0, t0),
        F.embedding(i1, t1),
        dense,
        s2,
        F.embedding(i3, t3),
        F.embedding(i4, t4),
        scalar,
    ]
    return torch.cat(cols, dim=1).to(torch.bfloat16)


def two_gathers_inputs(n: int) -> dict:
    return {
        "i0": torch.randint(0, 512, (n,), device="cuda"),
        "i1": torch.randint(0, 512, (n,), device="cuda"),
        "t0": torch.randn(512, 64, device="cuda"),
        "t1": torch.randn(512, 64, device="cuda"),
        **dense_inputs(n, [64] * 4),
    }


def two_gathers(i0, i1, t0, t1, x0, x1, x2, x3):
    return torch.cat(
        [F.embedding(i0, t0), F.embedding(i1, t1), x0, x1, x2, x3], dim=1
    ).to(torch.bfloat16)


def dense6(x0, x1, x2, x3, x4, x5):
    return torch.cat([x0 * 1.5, x1, x2, x3, x4, x5], dim=1).to(torch.bfloat16)


def narrowmix(x0, x1, x2, x3, x4, x5, x6, x7):
    return torch.cat([x.sigmoid() for x in (x0, x1, x2, x3, x4, x5, x6, x7)], dim=1).to(
        torch.bfloat16
    )


def dense_inputs(n: int, widths: list[int]) -> dict:
    return {f"x{i}": torch.randn(n, w, device="cuda") for i, w in enumerate(widths)}


def softmax_inputs(h: int, b: int, l: int) -> dict:
    return {
        "q": torch.randn(h, b, l, device="cuda", dtype=torch.bfloat16),
        "kt": torch.randn(b, l, h, device="cuda", dtype=torch.bfloat16),
        "bias": 0.1 * torch.randn(h, 1, l, device="cuda"),
        "temp": 1.0 + 0.1 * torch.rand(h, 1, 1, device="cuda"),
    }


def softmax_pool(q, kt, bias, temp):
    s = q.to(temp.dtype) * temp + bias
    out = (torch.softmax(s, dim=-1) * kt.permute(2, 0, 1).to(temp.dtype)).sum(-1)
    return out.permute(1, 0)


# name, fn, make_inputs, off options, on options, backward
CASES = [
    (
        "widecat_gather_1M",
        widecat,
        lambda: widecat_inputs(1 << 20),
        CAT_OFF,
        CAT_ON,
        False,
    ),
    (
        "widecat_gather_4M",
        widecat,
        lambda: widecat_inputs(4 << 20),
        CAT_OFF,
        CAT_ON,
        False,
    ),
    (
        "guard_two_gathers_4M",
        two_gathers,
        lambda: two_gathers_inputs(4 << 20),
        CAT_OFF,
        CAT_ON,
        False,
    ),
    (
        "guard_dense6_4M",
        dense6,
        lambda: dense_inputs(4 << 20, [64] * 6),
        CAT_OFF,
        CAT_ON,
        False,
    ),
    (
        "guard_narrowmix_4M",
        narrowmix,
        lambda: dense_inputs(4 << 20, [64, 1, 2, 64, 1, 64, 4, 1]),
        CAT_OFF,
        CAT_ON,
        False,
    ),
    (
        "softmax_pool_48x32768x384",
        softmax_pool,
        lambda: softmax_inputs(48, 32768, 384),
        {},
        TILE_ON,
        False,
    ),
    (
        "softmax_pool_48x32768x384_bwd",
        softmax_pool,
        lambda: softmax_inputs(48, 32768, 384),
        {},
        TILE_ON,
        True,
    ),
    (
        "softmax_pool_32x65536x256",
        softmax_pool,
        lambda: softmax_inputs(32, 65536, 256),
        {},
        TILE_ON,
        False,
    ),
    (
        "softmax_pool_32x65536x256_bwd",
        softmax_pool,
        lambda: softmax_inputs(32, 65536, 256),
        {},
        TILE_ON,
        True,
    ),
]


def fresh(fn: Callable, tag: str) -> Callable:
    code = fn.__code__.replace(co_name=f"{fn.__name__}_{tag}")
    return type(fn)(code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__)


def compile_with(fn: Callable, options: dict, tag: str) -> Callable:
    return torch.compile(
        fresh(fn, tag), dynamic=False, options={"force_disable_caches": True, **options}
    )


def fwd_bwd(f: Callable, inputs: dict) -> Callable:
    leaves = {
        k: v.detach().requires_grad_(v.dtype == torch.bfloat16)
        for k, v in inputs.items()
    }
    grad = torch.randn_like(f(**leaves))

    def run():
        out = f(**leaves)
        out.backward(grad)
        return out

    return run


def run_case(case: tuple, rounds: int) -> dict:
    name, fn, make, off_opts, on_opts, backward = case
    torch.manual_seed(0)
    inputs = make()
    torch._dynamo.reset()
    off, on = compile_with(fn, off_opts, "off"), compile_with(fn, on_opts, "on")
    y_off, y_on = off(**inputs), on(**inputs)
    if y_off.dtype == torch.bfloat16 and not backward and "softmax" not in name:
        if torch.equal(y_off, y_on):
            check = "bitwise"
        else:
            diff = y_off.ne(y_on)
            col = diff.any(0).nonzero().flatten().tolist()
            d = (y_off.float() - y_on.float()).abs().max().item()
            check = f"NOT bitwise: {diff.sum().item()} of {diff.numel()} elements differ (max abs diff {d:.3g}; columns {col[:8]})"
    else:
        ref = fn(
            **{k: v.double() if v.is_floating_point() else v for k, v in inputs.items()}
        )
        e_off, e_on = (y_off.double() - ref).abs(), (y_on.double() - ref).abs()
        check = f"off {e_off.max().item():.2e}/{e_off.mean().item():.2e} on {e_on.max().item():.2e}/{e_on.mean().item():.2e}"
    run_off = fwd_bwd(off, inputs) if backward else (lambda: off(**inputs))
    run_on = fwd_bwd(on, inputs) if backward else (lambda: on(**inputs))
    run_off()
    run_on()
    t_off, t_on = [], []
    for _ in range(rounds):
        t_off.append(do_bench(run_off, return_mode="median"))
        t_on.append(do_bench(run_on, return_mode="median"))
    row = {
        "case": name,
        "off_ms": statistics.median(t_off),
        "on_ms": statistics.median(t_on),
        "check": check,
    }
    row["speedup"] = row["off_ms"] / row["on_ms"]
    print(
        f"| {name} | {row['off_ms']:.3f} | {row['on_ms']:.3f} | {row['speedup']:.2f}x | {check} |",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    # optional leading action, for benchmark harnesses that pass one
    parser.add_argument("action", nargs="?", default="perf", choices=["perf"])
    parser.add_argument(
        "--cases", default="", help="comma-separated case names (default: all)"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    names = set(args.cases.split(",")) if args.cases else None
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}")
    print("| case | off ms | on ms | off/on | numerics (max/mean err vs fp64) |")
    print("|---|---|---|---|---|")
    rows = [run_case(c, args.rounds) for c in CASES if names is None or c[0] in names]
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "args": vars(args),
                    "device": torch.cuda.get_device_name(),
                    "rows": rows,
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
