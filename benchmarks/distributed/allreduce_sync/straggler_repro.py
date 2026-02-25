"""
Minimal reproduction of the allreduce straggler effect.

All ranks do identical compute followed by a barrier allreduce, repeated
N times per step. One rank consistently becomes the straggler.

The effect scales with compute complexity between allreduces: a single
matmul shows ~7µs of skew, but a chain of small ops (mimicking a real
transformer layer) shows hundreds of µs.

Usage:
  torchrun --nproc_per_node=2 straggler_repro.py
  torchrun --nproc_per_node=8 straggler_repro.py
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    numel = 4096
    pool = symm_mem.get_mem_pool(device)
    with torch.cuda.use_mem_pool(pool):
        buf = torch.empty(numel, dtype=torch.bfloat16, device=device)
    group_name = dist.group.WORLD.group_name

    # Simulate a chain of small kernels (like QKV → norm → RoPE → SDPA → o_proj)
    W1 = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    W2 = torch.randn(4096, 512, dtype=torch.bfloat16, device=device)
    W3 = torch.randn(512, 4096, dtype=torch.bfloat16, device=device)
    x = torch.randn(1, 4096, dtype=torch.bfloat16, device=device)
    norm_w = torch.ones(4096, dtype=torch.bfloat16, device=device)

    def compute_chain():
        """Chain of small ops mimicking attention block compute."""
        h = F.linear(x, W1)            # QKV-like
        h = F.rms_norm(h, (4096,), norm_w)  # norm
        h = h * torch.rsqrt(torch.tensor(128.0, device=device))  # scale
        h = F.linear(h, W2.T)          # down-project
        h = F.silu(h)                   # activation
        h = F.linear(h, W3.T)          # up-project
        buf.copy_(h.view(-1))
        return buf

    n_allreduces = 4
    iters = 100
    warmup = 20

    def step(starts, ends, offset):
        for j in range(n_allreduces):
            compute_chain()
            starts[offset + j].record()
            torch.ops.symm_mem.one_shot_all_reduce(buf, "sum", group_name)
            ends[offset + j].record()

    # Warmup
    dummy_s = [torch.cuda.Event(enable_timing=True) for _ in range(n_allreduces)]
    dummy_e = [torch.cuda.Event(enable_timing=True) for _ in range(n_allreduces)]
    for _ in range(warmup):
        step(dummy_s, dummy_e, 0)
    torch.cuda.synchronize()

    # Timed
    total_events = iters * n_allreduces
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(total_events)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(total_events)]

    torch.cuda.synchronize()
    for i in range(iters):
        step(starts, ends, i * n_allreduces)
    torch.cuda.synchronize()

    # Per-slot median
    slot_times = [[] for _ in range(n_allreduces)]
    for i in range(iters):
        for j in range(n_allreduces):
            idx = i * n_allreduces + j
            slot_times[j].append(starts[idx].elapsed_time(ends[idx]) * 1000)

    medians = [sorted(t)[len(t) // 2] for t in slot_times]
    medians_tensor = torch.tensor(medians, dtype=torch.float64, device=device)
    all_medians = [torch.empty_like(medians_tensor) for _ in range(world_size)]
    dist.all_gather(all_medians, medians_tensor)

    if rank == 0:
        print(f"compute chain (6 small kernels) → allreduce [{numel} bf16]")
        print(f"world_size={world_size}, n_allreduces_per_step={n_allreduces}, iters={iters}")
        print()
        print("allreduce p50 per rank per slot (µs):")
        header = "         " + "  ".join(f"{'AR'+str(j):>8s}" for j in range(n_allreduces))
        print(header)
        for r in range(world_size):
            vals = all_medians[r].tolist()
            line = f"  rank {r}: " + "  ".join(f"{v:8.1f}" for v in vals)
            if sum(vals) == min(sum(all_medians[rr].tolist()) for rr in range(world_size)):
                line += "  <-- straggler"
            print(line)
        print()
        all_totals = [sum(all_medians[r].tolist()) for r in range(world_size)]
        straggler_r = all_totals.index(min(all_totals))
        non_straggler = [t for i, t in enumerate(all_totals) if i != straggler_r]
        print(f"  straggler (rank {straggler_r}) total: {all_totals[straggler_r]:8.1f} µs")
        print(f"  non-straggler mean total:    {sum(non_straggler)/len(non_straggler):8.1f} µs")
        print(f"  wait overhead per step:      {sum(non_straggler)/len(non_straggler) - all_totals[straggler_r]:8.1f} µs")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
