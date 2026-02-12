#!/usr/bin/env python3
"""
Benchmark: ExternKernelOut vs FallbackKernel for symm_mem output buffer reuse.

Run with:
    torchrun --nproc_per_node=2 benchmark.py [label]

This measures peak GPU memory and execution time for a multi-layer
matmul → one_shot_all_reduce model, and saves the generated Inductor code
for inspection.

Pass an optional label (e.g. "baseline" or "modified") as argv[1] to tag
the generated-code snapshot that is written to the current directory.
"""

import gc
import os
import sys

import torch
import torch.distributed as dist
import torch._inductor.config as inductor_config


def fmt_bytes(b: int) -> str:
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.2f} MB"
    return f"{b / 1024:.2f} KB"


class MultiLayerAllReduce(torch.nn.Module):
    """matmul → one_shot_all_reduce per layer (tensor-parallel transformer)."""

    def __init__(self, hidden_size: int, num_layers: int, group_name: str):
        super().__init__()
        self.num_layers = num_layers
        self.group_name = group_name
        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16)
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = x @ self.weights[i]
            x = torch.ops.symm_mem.one_shot_all_reduce(x, "sum", self.group_name)
        return x


def save_generated_code(label: str):
    """Find and save the most recent Inductor-generated code containing one_shot_all_reduce."""
    import tempfile

    debug_dir = os.path.join(
        tempfile.gettempdir(), f"torchinductor_{os.environ.get('USER', 'user')}"
    )
    best_path = None
    best_mtime = 0.0
    for root, _dirs, files in os.walk(debug_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                mtime = os.path.getmtime(path)
                if mtime > best_mtime:
                    with open(path) as fh:
                        content = fh.read()
                    if "one_shot_all_reduce" in content:
                        best_path = path
                        best_mtime = mtime

    if best_path:
        save_path = os.path.join(os.path.dirname(__file__), f"generated_code_{label}.py")
        with open(best_path) as fh:
            code = fh.read()
        with open(save_path, "w") as fh:
            fh.write(code)
        return save_path, code
    return None, ""


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    group_name = dist.group.WORLD.group_name

    hidden_size = 4096
    seq_len = 128
    num_layers = 8
    warmup = 5
    repeats = 20

    if rank == 0:
        print("=" * 70)
        print("  symm_mem ExternKernelOut vs FallbackKernel benchmark")
        print("=" * 70)
        print(f"  World size: {world_size}")
        print(f"  Config: hidden={hidden_size}, seq={seq_len}, layers={num_layers}")
        tensor_size = seq_len * hidden_size * 2  # bfloat16
        print(f"  Per-tensor size: {fmt_bytes(tensor_size)}")
        print()

    model = MultiLayerAllReduce(hidden_size, num_layers, group_name).to(device)
    x = torch.randn(seq_len, hidden_size, device=device, dtype=torch.bfloat16)

    inductor_config.debug = True
    compiled = torch.compile(model, fullgraph=True)

    with torch.no_grad():
        for _ in range(warmup):
            _ = compiled(x)
    torch.cuda.synchronize()
    inductor_config.debug = False

    if rank == 0:
        label = sys.argv[1] if len(sys.argv) > 1 else "current"
        save_path, code = save_generated_code(label)
        if save_path:
            print(f"  Generated code saved to: {save_path}")
            print(f"  Allocations (empty_strided_cuda): {code.count('empty_strided_cuda(')}")
            print(f"  Reuses (# reuse):                 {code.count('# reuse')}")
            print(f"  P2P allocs (empty_strided_p2p):   {code.count('empty_strided_p2p(')}")
            print(f"  Out-variant calls (', out='):      {code.count(', out=')}")
        print()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(repeats):
            _ = compiled(x)
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        start_event.record()
        for _ in range(repeats):
            _ = compiled(x)
        end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    per_iter_us = (elapsed_ms * 1000) / repeats

    if rank == 0:
        print(f"  Peak memory: {fmt_bytes(peak_mem)}")
        print(f"  Time per iter: {per_iter_us:.1f} μs")
        print("=" * 70)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
