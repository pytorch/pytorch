#!/usr/bin/env python3
"""Triton Kernel Trace real-device demo / verification (Task 9).

Runs a profiled torch.compile under INDUCTOR_PROVENANCE=1 on whatever accelerator
is present (xpu / cuda / cpu), confirms the inductor_triton_kernel_trace artifact is
emitted, post-processes the profiler Chrome trace, and prints the kernel events that
got enriched with inductor_kernel_ops (ordered inner ops + rematerialization).

Reused verbatim on CUDA (ba02dcaiubt022 / scaia253) and XPU (B580 DUT4064).
"""
import os

os.environ["INDUCTOR_PROVENANCE"] = "1"
import json

import torch
import torch._inductor.config as ic

ic.force_disable_caches = True

XPU = hasattr(torch, "xpu") and torch.xpu.is_available()
CUDA = torch.cuda.is_available()
DEV = "xpu" if XPU else "cuda" if CUDA else "cpu"


class RoPEBlock(torch.nn.Module):
    """RMSNorm + rotary; cos/sin recomputed for q and k -> rematerialization."""

    def __init__(self, d=256):
        super().__init__()
        self.wq = torch.nn.Linear(d, d, bias=False)
        self.wk = torch.nn.Linear(d, d, bias=False)
        self.g = torch.nn.Parameter(torch.randn(d))

    def forward(self, x, pos):
        v = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g
        q, k = self.wq(v), self.wk(v)
        cos, sin = torch.cos(pos), torch.sin(pos)
        q = q * cos + torch.roll(q, 1, -1) * sin
        k = k * cos + torch.roll(k, 1, -1) * sin
        return q + k


def main():
    dev_name = (
        torch.xpu.get_device_name(0)
        if XPU
        else torch.cuda.get_device_name(0)
        if CUDA
        else "cpu"
    )
    print(f"DEV={DEV} device={dev_name} torch={torch.__version__}")

    m = RoPEBlock().to(DEV).eval()
    x = torch.randn(4, 128, 256, device=DEV)
    pos = torch.randn(4, 128, 256, device=DEV)
    cm = torch.compile(m, backend="inductor")
    with torch.no_grad():
        cm(x, pos)  # warm compile -> emits the trace_structured artifact

    acts = [torch.profiler.ProfilerActivity.CPU]
    if DEV == "xpu":
        acts.append(torch.profiler.ProfilerActivity.XPU)
    elif DEV == "cuda":
        acts.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=acts, with_stack=True) as prof:
        with torch.no_grad():
            for _ in range(3):
                cm(x, pos)

    out = f"/tmp/kernel_trace_demo_{DEV}.json"
    prof.export_chrome_trace(out)

    from torch.profiler._utils import map_recorded_events_to_aten_ops_with_stack_trace

    data = json.load(open(out))
    map_recorded_events_to_aten_ops_with_stack_trace(data)

    enriched = [
        e
        for e in data.get("traceEvents", [])
        if isinstance(e.get("args"), dict) and e["args"].get("inductor_kernel_ops")
    ]
    print(f"enriched_kernel_events={len(enriched)}")
    for e in enriched[:8]:
        print("  ", e.get("name"), "->", e["args"]["inductor_kernel_ops"])

    enriched_path = out.replace(".json", "_enriched.json")
    json.dump(data, open(enriched_path, "w"))
    print(f"wrote {enriched_path}")


if __name__ == "__main__":
    main()
