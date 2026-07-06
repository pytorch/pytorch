"""cross_entropy: fused Metal kernel vs the reference decomposed path.

Same-build A/B. Mode A = reference log_softmax + nll_loss (forced via
PYTORCH_MPS_FORCE_DECOMPOSED_CROSS_ENTROPY=1); mode B = the fused Metal CE kernel
(env unset). The toggle is read once per process (static in
can_use_fused_mps_cross_entropy), so each MODE runs in its own process:
  PYTORCH_MPS_FORCE_DECOMPOSED_CROSS_ENTROPY=1 BENCH_MODE=A python bench_crossentropy.py
  BENCH_MODE=B python bench_crossentropy.py
Methodology: amortized many-iter / one-sync (dodges the MPS sync floor).
Defaults to MPS (the A/B toggle only affects MPS) but --device works for a
portable timing run on CUDA/CPU. Covers contiguous and non-contiguous logits.
Requires this checkout importable as torch (pip install -e . or PYTHONPATH).
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F


MODE = os.environ.get("BENCH_MODE", "B")
DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
# (B=rows, V=num_classes) tied to real LM/classifier head sizes.
CONFIGS = [
    ("8192x4096", 8192, 4096),  # hidden-size head
    ("4096x32000", 4096, 32000),  # Llama-2 vocab
    ("16384x50304", 16384, 50304),  # GPT-NeoX vocab
    ("2048x128256", 2048, 128256),  # Llama-3 vocab
    ("4096x1024", 4096, 1024),  # small head
]
ITERS = int(os.environ.get("ITERS", "100"))


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(device, B, V, dt, layout, backward):
    if layout == "noncontig":
        # Non-contiguous (B, V) logits via transpose of a (V, B) tensor; the
        # dispatch then exercises its .contiguous() fallback.
        x = torch.randn(V, B, device=device, dtype=dt).t()
    else:
        x = torch.randn(B, V, device=device, dtype=dt)
    x.requires_grad_(backward)
    t = torch.randint(0, V, (B,), device=device)
    for _ in range(20):
        loss = F.cross_entropy(x, t)
        if backward:
            x.grad = None
            loss.backward()
    synchronize(device)
    s = time.perf_counter()
    for _ in range(ITERS):
        loss = F.cross_entropy(x, t)
        if backward:
            x.grad = None
            loss.backward()
    synchronize(device)
    return (time.perf_counter() - s) / ITERS * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)

    forced = os.environ.get("PYTORCH_MPS_FORCE_DECOMPOSED_CROSS_ENTROPY")
    sys.stderr.write(f"MODE={MODE} FORCE_DECOMP={forced!r} torch={torch.__file__}\n")
    out = {}
    for nm, B, V in CONFIGS:
        for dn, dt in DTYPES.items():
            for layout in ("contig", "noncontig"):
                for kind in ("fwd", "fwdbwd"):
                    key = f"{nm}|{dn}|{layout}|{kind}"
                    try:
                        out[key] = round(
                            bench(device, B, V, dt, layout, kind == "fwdbwd"), 1
                        )
                    except Exception as e:
                        out[key] = f"ERR:{str(e)[:50]}"
    print(json.dumps(out))


if __name__ == "__main__":
    main()
