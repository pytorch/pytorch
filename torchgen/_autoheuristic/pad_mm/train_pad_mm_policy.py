#!/usr/bin/env python3
"""Train a pad_mm policy by benchmarking each shape on the local device.

Given a CSV of GEMM shapes (``M,K,N,dtype`` -- the same format the other pad_mm
scripts use), this benchmarks the padded vs. unpadded matmul on the current GPU
and emits the source of an ``AdsPadPolicy`` (a ``PadMMPolicy``) that force-pads
the shapes where padding was faster and force-skips the rest. Shapes not in the
table return ``None`` (defer to Inductor's built-in decision).

This is a simpler, benchmark-only alternative to the learned AutoHeuristic: the
AutoHeuristic only runs for ``aten.mm``, only above a size precondition, and is
skipped entirely in deterministic mode. The generated policy is honored for
mm/addmm/bmm and in deterministic mode, so it recovers padding wins the
AutoHeuristic leaves on the table.

Usage:

    python train_pad_mm_policy.py shapes.csv -o ads_pad_policy.py

Then, in your training job (e.g. in the trainer's inductor config setup):

    from ads_pad_policy import AdsPadPolicy
    torch._inductor.config.pad_mm_policy = AdsPadPolicy()

Re-run whenever the hardware or the shape set changes; the generated ``uuid()``
folds the table contents into the FX graph cache key, so a re-tuned policy
correctly invalidates cached artifacts.
"""

import argparse
import hashlib

# evaluate_pad_mm_heuristics is a sibling script; reuse its benchmarking helpers.
from evaluate_pad_mm_heuristics import (  # type: ignore[import-not-found]
    benchmark_both_choices,
    fits_in_memory,
    load_shapes_from_csv,
)

import torch
from torch._inductor.fx_passes.pad_mm import get_alignment_size_dtype


# Pad only when clearly faster, matching is_padded_faster() in pad_mm.py.
PAD_MULTIPLIER = 1.1

_POLICY_TEMPLATE = """\
# {generated} by torchgen/_autoheuristic/pad_mm/train_pad_mm_policy.py. Do not edit.
#
# Offline-tuned pad_mm policy. Install it in your training job with:
#
#     from ads_pad_policy import AdsPadPolicy
#     torch._inductor.config.pad_mm_policy = AdsPadPolicy()
from torch._inductor.custom_graph_pass import PadMMPolicy


class AdsPadPolicy(PadMMPolicy):
    # (m, k, n, str(dtype)) -> should_pad. Shapes absent from the table return
    # None, i.e. defer to Inductor's built-in decision.
    _TABLE = {{
{rows}
    }}

    def __call__(self, ctx):
        return self._TABLE.get((ctx.m, ctx.k, ctx.n, str(ctx.mat1_dtype)))

    def uuid(self):
        return "{uuid}"
"""


def render_policy(decisions: dict) -> str:
    rows = "\n".join(
        f"        ({m}, {k}, {n}, {str(dt)!r}): {should},"
        for (m, k, n, dt), should in decisions.items()
    )
    digest = hashlib.sha256(rows.encode()).hexdigest()[:16]
    return _POLICY_TEMPLATE.format(
        generated="Generated", rows=rows, uuid=f"ads-pad-policy-{digest}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv_file", help="CSV with M,K,N,dtype columns")
    parser.add_argument("-o", "--output", default="ads_pad_policy.py")
    parser.add_argument("--num-reps", type=int, default=10)
    args = parser.parse_args()

    torch.set_default_device("cuda")

    decisions: dict = {}
    for m, k, n, dtype in load_shapes_from_csv(args.csv_file):
        align = get_alignment_size_dtype(dtype)
        if align == 0 or all(dim % align == 0 for dim in (m, k, n)):
            continue  # already aligned: nothing to pad
        if not fits_in_memory(dtype, m, k, n):
            continue
        orig_time, pad_time = benchmark_both_choices(m, k, n, dtype, args.num_reps)
        should_pad = orig_time > pad_time * PAD_MULTIPLIER
        decisions[(m, k, n, dtype)] = should_pad
        print(
            f"M={m} K={k} N={n} {dtype}: orig={orig_time * 1e3:.3f}ms "
            f"pad={pad_time * 1e3:.3f}ms -> pad={should_pad}"
        )

    with open(args.output, "w") as f:
        f.write(render_policy(decisions))
    n_pad = sum(decisions.values())
    print(f"\nWrote {args.output}: {n_pad}/{len(decisions)} shapes set to pad=True")


if __name__ == "__main__":
    main()
