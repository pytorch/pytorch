#!/usr/bin/env python3
"""
Compare ``torch.distributed._symmetric_memory.all_to_all_permute`` against
``permute`` + ``torch.distributed.all_to_all_single`` (equivalent data movement).

The (scatter_dim=1, gather_dim=0) case maps per-rank input
``[seq_len / G, G * local_cols]`` to ``[G, seq_len / G, local_cols]``, where
``seq_len`` is the global sequence length (``rows = seq_len // G``). A standard
implementation is::

    x = input.view(rows, G, local_cols).permute(1, 0, 2).contiguous()
    dist.all_to_all_single(flat_out, x.view(-1))

Run (multi-process; the test harness spawns one process per GPU)::

    python benchmarks/distributed/bench_all_to_all_permute.py -v

``test_profile_col_scatter_implementations`` runs both implementations back to
back inside each profiler step (symm then ``permute`` + ``all_to_all_single``),
using ``schedule`` + ``p.step()``, ``with_stack=True``, prints one summary on
rank 0, and writes a single Chrome trace plus ``export_stacks`` under
``<cwd>/bench_a2a_permute_traces/run_*/`` on rank 0 (not ``/tmp``).

Requires: NCCL symmetric memory (2.29.7+), CUDA P2P, at least 2 GPUs.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable  # noqa: TC003

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl_version,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    requires_cuda_p2p_access,
    run_tests,
    skip_but_pass_in_sandcastle_if,
)


if not dist.is_available() or not dist.is_nccl_available():
    print("c10d NCCL not available, skipping benchmark module")
    sys.exit(0)

# Global sequence length (dim 0 after gather). Per-rank input rows = seq_len // world_size.
_COL_SCATTER_BENCH_SEQ_LEN = 4096


def _mean_cuda_ms_per_iter(
    device: torch.device, n_iters: int, run_iter: Callable[[], None]
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.set_device(device)
    start.record()
    for _ in range(n_iters):
        run_iter()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / n_iters


def _permute_all_to_all_single_col_scatter(
    inp: torch.Tensor,
    out: torch.Tensor,
    *,
    world_size: int,
) -> None:
    """Baseline: ``view -> permute(1,0,2) -> all_to_all_single`` for (1,0) layout."""
    rows, plocal = inp.shape
    lc = plocal // world_size
    x = inp.view(rows, world_size, lc).permute(1, 0, 2).contiguous()
    flat_in = x.reshape(-1)
    dist.all_to_all_single(out.reshape(-1), flat_in)


@requires_cuda_p2p_access()
@skip_but_pass_in_sandcastle_if(not TEST_CUDA, "CUDA not available")
class AllToAllPermuteBenchmark(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str | None:
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_nccl_symm(self) -> None:
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.device)
        dist.all_reduce(torch.ones(1, device=self.device))

    @requires_nccl_version((2, 29, 0), "nccl_all_to_all_permute requires nccl 2.29.0")
    @skip_if_lt_x_gpu(2)
    def test_bench_col_scatter_vs_permute_a2a(self) -> None:
        """
        Benchmark (scatter_dim=1, gather_dim=0): ``[seq_len/G, G*lc]`` -> ``[G, seq_len/G, lc]``.
        """
        self._init_nccl_symm()
        group_name = dist.group.WORLD.group_name
        g = dist.get_world_size()
        dtype = torch.float16
        seq_len = _COL_SCATTER_BENCH_SEQ_LEN
        assert seq_len % g == 0, f"seq_len ({seq_len}) must be divisible by G ({g})"  # noqa: S101
        rows = seq_len // g
        local_cols = (
            1024  # row_bytes = local_cols * esize; must be divisible by 16 for the op
        )

        warmup = 20
        bench = 40

        # --- symmetric-memory path ---
        symm_in = symm_mem.empty(
            rows, g * local_cols, dtype=dtype, device=self.device
        ).fill_(float(self.rank))
        symm_mem.rendezvous(symm_in, group=group_name)
        symm_out = torch.empty(g, rows, local_cols, dtype=dtype, device=self.device)

        for _ in range(warmup):
            symm_mem.all_to_all_permute(
                symm_in,
                symm_out,
                scatter_dim=1,
                gather_dim=0,
                group=group_name,
            )
        dist.barrier()

        t_symm = _mean_cuda_ms_per_iter(
            self.device,
            bench,
            lambda: symm_mem.all_to_all_permute(
                symm_in,
                symm_out,
                scatter_dim=1,
                gather_dim=0,
                group=group_name,
            ),
        )

        # --- permute + all_to_all_single (regular CUDA tensors) ---
        base_in = torch.empty(rows, g * local_cols, dtype=dtype, device=self.device)
        base_in.copy_(symm_in)
        base_out = torch.empty(g, rows, local_cols, dtype=dtype, device=self.device)

        for _ in range(warmup):
            _permute_all_to_all_single_col_scatter(base_in, base_out, world_size=g)
        dist.barrier()

        t_base = _mean_cuda_ms_per_iter(
            self.device,
            bench,
            lambda: _permute_all_to_all_single_col_scatter(
                base_in, base_out, world_size=g
            ),
        )

        elems = rows * g * local_cols
        nbytes = elems * dtype.itemsize

        if self.rank == 0:
            print()
            print("bench_all_to_all_permute: (scatter_dim=1, gather_dim=0)")
            print(
                f"  seq_len={seq_len}, rows=seq_len/G={rows}, G={g}, "
                f"local_cols={local_cols}, dtype={dtype}, "
                f"elements/rank={elems}, bytes/rank={nbytes / 1e6:.2f} MB"
            )
            print(f"  all_to_all_permute (symm): {t_symm:.4f} ms/iter")
            print(f"  permute + all_to_all_single: {t_base:.4f} ms/iter")
            print(f"  ratio (baseline / symm): {t_base / t_symm:.3f}x")
            print()

    @requires_nccl_version((2, 29, 0), "nccl_all_to_all_permute requires nccl 2.29.0")
    @skip_if_lt_x_gpu(2)
    def test_profile_col_scatter_implementations(self) -> None:
        """Profile both paths in one session: each step runs symm then baseline (rank 0 prints + traces)."""
        self._init_nccl_symm()
        group_name = dist.group.WORLD.group_name
        g = dist.get_world_size()
        dtype = torch.float16
        seq_len = _COL_SCATTER_BENCH_SEQ_LEN
        assert seq_len % g == 0, f"seq_len ({seq_len}) must be divisible by G ({g})"  # noqa: S101
        rows = seq_len // g
        local_cols = 1024

        symm_in = symm_mem.empty(
            rows, g * local_cols, dtype=dtype, device=self.device
        ).fill_(float(self.rank))
        symm_mem.rendezvous(symm_in, group=group_name)
        symm_out = torch.empty(g, rows, local_cols, dtype=dtype, device=self.device)
        base_in = torch.empty(rows, g * local_cols, dtype=dtype, device=self.device)
        base_in.copy_(symm_in)
        base_out = torch.empty(g, rows, local_cols, dtype=dtype, device=self.device)

        for _ in range(3):
            symm_mem.all_to_all_permute(
                symm_in,
                symm_out,
                scatter_dim=1,
                gather_dim=0,
                group=group_name,
            )
            _permute_all_to_all_single_col_scatter(base_in, base_out, world_size=g)
        dist.barrier()
        torch.cuda.synchronize(self.device)

        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        # One cycle: wait=0, warmup=1, active=N → (1 + N) steps (see torch.profiler.schedule).
        active_steps = 12
        step_schedule = torch.profiler.schedule(
            wait=0,
            warmup=1,
            active=active_steps,
            repeat=1,
        )
        profile_steps = 1 + active_steps

        with torch.profiler.profile(
            activities=activities,
            schedule=step_schedule,
            record_shapes=True,
            with_stack=True,
            with_modules=True,
        ) as prof:
            for _ in range(profile_steps):
                symm_mem.all_to_all_permute(
                    symm_in,
                    symm_out,
                    scatter_dim=1,
                    gather_dim=0,
                    group=group_name,
                )
                _permute_all_to_all_single_col_scatter(base_in, base_out, world_size=g)
                prof.step()
        torch.cuda.synchronize(self.device)
        dist.barrier()

        if self.rank == 0:
            trace_root = os.path.join(os.getcwd(), "bench_a2a_permute_traces")
            os.makedirs(trace_root, exist_ok=True)
            trace_dir = tempfile.mkdtemp(prefix="run_", dir=trace_root)
            trace_json = os.path.join(trace_dir, "col_scatter_both_impls.json")
            trace_stacks = os.path.join(trace_dir, "col_scatter_both_impls.stacks")
            prof.export_chrome_trace(trace_json)
            prof.export_stacks(trace_stacks, metric="self_cuda_time_total")

            table_kwargs = dict(sort_by="self_cuda_time_total", row_limit=30)
            print()
            print(
                "profile: both per step (all_to_all_permute then permute+all_to_all_single)"
            )
            print(prof.key_averages().table(**table_kwargs))
            print(f"Chrome trace directory: {trace_dir}")
            print(f"  (chrome://tracing) {trace_json}")
            print(f"  (export_stacks)    {trace_stacks}")
            print()


if __name__ == "__main__":
    run_tests()
