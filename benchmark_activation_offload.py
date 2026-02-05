"""
Benchmark: Pipeline Parallelism with Activation Offloading

Compares memory usage and throughput for a ~150M parameter transformer model
with and without activation offloading enabled.

Usage:
    # Single node, 4 GPUs (adjust CUDA_VISIBLE_DEVICES as needed)
    torchrun --nproc_per_node=4 benchmark_activation_offload.py

    # With specific schedule
    torchrun --nproc_per_node=4 benchmark_activation_offload.py --schedule interleaved_1f1b

    # Memory profile only (skip throughput benchmark)
    torchrun --nproc_per_node=4 benchmark_activation_offload.py --memory-only

    # View memory snapshots:
    # Open https://pytorch.org/memory_viz and drag the .pickle files
"""

import argparse
import gc
import os
import json
import pickle
from dataclasses import dataclass, asdict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage, ScheduleInterleaved1F1B


MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000


@dataclass
class BenchmarkResult:
    schedule: str
    offload_enabled: bool
    n_microbatches: int
    batch_size: int
    seq_len: int
    peak_memory_gb: float
    allocated_memory_gb: float
    throughput_samples_per_sec: float | None
    time_per_step_ms: float | None


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


class TransformerModel(nn.Module):
    """
    ~1B parameter transformer model.

    With d_model=2048, n_heads=16, d_ff=8192, n_layers=24:
    - Embedding: vocab_size * d_model = 50k * 2048 = 102M params
    - Per layer: ~4 * d_model^2 (attn) + 2 * d_model * d_ff (ff) = ~50M params
    - Total: ~102M + 24 * 50M = ~1.3B params
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 2048,
        n_heads: int = 16,
        d_ff: int = 8192,
        n_layers: int = 24,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class TransformerStage(nn.Module):
    """A pipeline stage containing a subset of transformer layers."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
        is_first: bool = False,
        is_last: bool = False,
        vocab_size: int = 50000,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        self.d_model = d_model

        if is_first:
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        if is_last:
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_first:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = self.embedding(x) + self.pos_embedding(positions)

        for layer in self.layers:
            x = layer(x)

        if self.is_last:
            x = self.norm(x)
            x = self.lm_head(x)

        return x


def create_pipeline_stages(
    rank: int,
    world_size: int,
    device: torch.device,
    d_model: int = 2048,
    n_heads: int = 32,
    d_ff: int = 8192,
    total_layers: int = 32,
    vocab_size: int = 32000,
    max_seq_len: int = 512,
    stages_per_rank: int = 2,
) -> list[TransformerStage]:
    """
    Create pipeline stages for interleaved scheduling.

    For ScheduleInterleaved1F1B, stages are assigned in looped fashion:
    - Rank 0: stages 0, world_size, 2*world_size, ...
    - Rank 1: stages 1, world_size+1, 2*world_size+1, ...
    - etc.

    With 4 ranks and 2 stages per rank:
    - Rank 0: stages 0 and 4
    - Rank 1: stages 1 and 5
    - Rank 2: stages 2 and 6
    - Rank 3: stages 3 and 7
    """
    num_stages = world_size * stages_per_rank
    layers_per_stage = total_layers // num_stages

    # Looped stage assignment: rank i owns stages [i, i + world_size, i + 2*world_size, ...]
    stage_indices = [rank + k * world_size for k in range(stages_per_rank)]

    stages = []
    for stage_idx in stage_indices:
        is_first = (stage_idx == 0)
        is_last = (stage_idx == num_stages - 1)

        stage = TransformerStage(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=layers_per_stage,
            is_first=is_first,
            is_last=is_last,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        ).to(device)
        stages.append(stage)

    return stages, stage_indices


def get_memory_stats(device: torch.device) -> tuple[float, float]:
    """Get peak and currently allocated memory in GB."""
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    return peak, allocated


def reset_memory_stats(device: torch.device):
    """Reset peak memory tracking."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()


def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable. Not recording memory history")
        return
    print("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )


def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable. Not stopping memory history")
        return
    print("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot(file_prefix: str) -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable. Not exporting memory snapshot")
        return
    try:
        print(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")


def run_benchmark(
    rank: int,
    world_size: int,
    device: torch.device,
    offload_enabled: bool,
    n_microbatches: int = 8,
    batch_size: int = 8,
    seq_len: int = 512,
    warmup_steps: int = 2,
    benchmark_steps: int = 3,
    memory_only: bool = False,
    snapshot_dir: str | None = None,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    reset_memory_stats(device)

    d_model = 2048
    vocab_size = 32000
    stages_per_rank = 2
    num_stages = world_size * stages_per_rank
    stage_modules, stage_indices = create_pipeline_stages(
        rank, world_size, device, stages_per_rank=stages_per_rank
    )

    n_params = sum(p.numel() for m in stage_modules for p in m.parameters())
    if rank == 0:
        print(f"  Stages per rank: {stages_per_rank}, Total stages: {num_stages}")
        print(f"  Stage params (this rank): {n_params / 1e6:.1f}M")

    pipe_stages = []
    for stage_module, stage_idx in zip(stage_modules, stage_indices):
        pipe_stage = PipelineStage(
            stage_module,
            stage_index=stage_idx,
            num_stages=num_stages,
            device=device,
        )
        pipe_stages.append(pipe_stage)

    # Define loss_fn for the last stage
    def loss_fn(output, target):
        return nn.functional.cross_entropy(
            output.view(-1, output.size(-1)), target.view(-1)
        )

    schedule = ScheduleInterleaved1F1B(
        stages=pipe_stages,
        n_microbatches=n_microbatches,
        enable_activation_offload=offload_enabled,
        loss_fn=loss_fn,
    )

    all_params = [p for m in stage_modules for p in m.parameters()]
    optimizer = torch.optim.AdamW(all_params, lr=1e-4)

    # With looped interleaved, rank 0 has first stage, rank (world_size-1) has last stage
    has_first_stage = 0 in stage_indices
    has_last_stage = (num_stages - 1) in stage_indices

    def run_step():
        optimizer.zero_grad()
        if has_first_stage and has_last_stage:
            # Edge case: single rank has both (shouldn't happen with world_size > 1)
            inputs = torch.randint(
                0, vocab_size, (batch_size * n_microbatches, seq_len), device=device
            )
            targets = torch.randint(
                0, vocab_size, (batch_size * n_microbatches, seq_len), device=device
            )
            losses = []
            schedule.step(inputs, target=targets, losses=losses)
        elif has_first_stage:
            # Rank 0: provides inputs
            inputs = torch.randint(
                0, vocab_size, (batch_size * n_microbatches, seq_len), device=device
            )
            schedule.step(inputs)
        elif has_last_stage:
            # Last rank: computes loss
            targets = torch.randint(
                0, vocab_size, (batch_size * n_microbatches, seq_len), device=device
            )
            losses = []
            schedule.step(target=targets, losses=losses)
        else:
            schedule.step()
        optimizer.step()

    # Warmup
    for _ in range(warmup_steps):
        run_step()

    torch.cuda.synchronize(device)
    reset_memory_stats(device)

    # Start memory recording if snapshot_dir is provided
    if snapshot_dir:
        start_record_memory_history()

    throughput = None
    time_per_step = None

    if not memory_only:
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(benchmark_steps):
            run_step()
        end_event.record()

        torch.cuda.synchronize(device)
        elapsed_ms = start_event.elapsed_time(end_event)
        time_per_step = elapsed_ms / benchmark_steps
        samples_per_step = batch_size * n_microbatches
        throughput = samples_per_step / (time_per_step / 1000)
    else:
        # Run at least one step for memory profiling
        run_step()

    peak_memory, allocated_memory = get_memory_stats(device)

    # Dump memory snapshot after all benchmark steps
    if snapshot_dir:
        torch.cuda.synchronize(device)
        offload_str = "with_offload" if offload_enabled else "no_offload"
        file_prefix = os.path.join(snapshot_dir, f"memory_snapshot_rank{rank}_{offload_str}")
        export_memory_snapshot(file_prefix)
        stop_record_memory_history()

    return BenchmarkResult(
        schedule="ScheduleInterleaved1F1B",
        offload_enabled=offload_enabled,
        n_microbatches=n_microbatches,
        batch_size=batch_size,
        seq_len=seq_len,
        peak_memory_gb=peak_memory,
        allocated_memory_gb=allocated_memory,
        throughput_samples_per_sec=throughput,
        time_per_step_ms=time_per_step,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-microbatches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--benchmark-steps", type=int, default=5)
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--snapshot-dir", type=str, default="./snapshots",
                        help="Directory to save memory snapshots (view at https://pytorch.org/memory_viz)")
    parser.add_argument("--output", type=str, default="activation_offload_benchmark.json")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create snapshot directory if specified
    if args.snapshot_dir and rank == 0:
        os.makedirs(args.snapshot_dir, exist_ok=True)
    dist.barrier()

    if rank == 0:
        print("=" * 70)
        print("Activation Offloading Benchmark")
        print("=" * 70)
        print(f"World size: {world_size}")
        print(f"Microbatches: {args.n_microbatches}")
        print(f"Batch size per microbatch: {args.batch_size}")
        print(f"Sequence length: {args.seq_len}")
        if args.snapshot_dir:
            print(f"Memory snapshots: {args.snapshot_dir}")
        print()

    results = []

    # Run without offloading
    if rank == 0:
        print("Running WITHOUT activation offloading...")
    dist.barrier()
    result_no_offload = run_benchmark(
        rank=rank,
        world_size=world_size,
        device=device,
        offload_enabled=False,
        n_microbatches=args.n_microbatches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        memory_only=args.memory_only,
        snapshot_dir=args.snapshot_dir,
    )
    results.append(result_no_offload)

    # Clean up before next run
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    # Run with offloading
    if rank == 0:
        print("\nRunning WITH activation offloading...")
    dist.barrier()
    result_with_offload = run_benchmark(
        rank=rank,
        world_size=world_size,
        device=device,
        offload_enabled=True,
        n_microbatches=args.n_microbatches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        memory_only=args.memory_only,
        snapshot_dir=args.snapshot_dir,
    )
    results.append(result_with_offload)

    # Gather and print results
    if rank == 0:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\n{'Configuration':<30} {'Peak Mem (GB)':<15} {'Throughput (samples/s)':<25}")
        print("-" * 70)

        for r in results:
            offload_str = "WITH offload" if r.offload_enabled else "WITHOUT offload"
            throughput_str = f"{r.throughput_samples_per_sec:.1f}" if r.throughput_samples_per_sec else "N/A"
            print(f"{offload_str:<30} {r.peak_memory_gb:<15.2f} {throughput_str:<25}")

        # Comparison
        print("\n" + "-" * 70)
        mem_saved = result_no_offload.peak_memory_gb - result_with_offload.peak_memory_gb
        mem_pct = (mem_saved / result_no_offload.peak_memory_gb) * 100 if result_no_offload.peak_memory_gb > 0 else 0

        print(f"\nMemory savings: {mem_saved:.2f} GB ({mem_pct:.1f}%)")

        if result_no_offload.throughput_samples_per_sec and result_with_offload.throughput_samples_per_sec:
            throughput_diff = result_with_offload.throughput_samples_per_sec - result_no_offload.throughput_samples_per_sec
            throughput_pct = (throughput_diff / result_no_offload.throughput_samples_per_sec) * 100
            print(f"Throughput change: {throughput_diff:+.1f} samples/s ({throughput_pct:+.1f}%)")

        # Save results to JSON
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
