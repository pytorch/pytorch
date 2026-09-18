#!/usr/bin/env python3
"""2-GPU FSDP + GDN context-parallel smoke.

Hard-caps this process at 500 MiB CUDA memory per visible device so a
failure OOMs *this* job instead of the JEPA trainer that already owns
the cards.

Usage (from an isolated venv, inside tmux session `torch`):

    torchrun --standalone --nproc_per_node=2 tools/gdn_cp/smoke_fsdp_cp.py
"""
from __future__ import annotations

import os
import sys
import traceback

# Keep NCCL / CUDA from grabbing extra pools before we set the cap.
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:32,garbage_collection_threshold:0.6")

MEM_CAP_MIB = 500


def _cap_device(device: int) -> float:
    import torch

    props = torch.cuda.get_device_properties(device)
    cap_bytes = MEM_CAP_MIB * 1024 * 1024
    frac = min(1.0, cap_bytes / float(props.total_memory))
    torch.cuda.set_per_process_memory_fraction(frac, device=device)
    return frac


def main() -> int:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._context_parallel._attention import (
        _cp_options,
    )
    from torch.nn.modules.gated_delta import TinyGatedDeltaModel

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    frac = _cap_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"this smoke expects 2 ranks, got {world}")

    if rank == 0:
        print(
            f"[gdn-smoke] torch={torch.__version__} git={getattr(torch.version, 'git_version', '?')}",
            flush=True,
        )
        print(f"[gdn-smoke] mem cap {MEM_CAP_MIB} MiB/GPU fraction={frac:.6f}", flush=True)
        try:
            import gated_delta_ext  # noqa: F401

            print("[gdn-smoke] CUDA extension: loaded", flush=True)
        except Exception as exc:
            print(f"[gdn-smoke] CUDA extension: python fallback ({exc.__class__.__name__})", flush=True)

    mesh = init_device_mesh("cuda", (1, 2), mesh_dim_names=("fsdp", "cp"))
    model = TinyGatedDeltaModel(
        vocab_size=32,
        hidden_size=32,
        num_heads=4,
        head_dim=8,
        num_layers=1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    n_params = model.n_parameters()
    fully_shard(model, mesh=mesh["fsdp"])
    _cp_options.enable_load_balance = False

    torch.manual_seed(0)
    # seq=32 is divisible by cp=2; tiny activations.
    tokens = torch.randint(0, 32, (1, 32), device="cuda")
    with context_parallel(mesh["cp"], buffers=[tokens], buffer_seq_dims=[1]):
        logits = model(tokens)
        loss = logits.float().pow(2).mean()
        loss.backward()
        in_proj_grad = model.layers[0].in_proj.weight.grad
        if in_proj_grad is None:
            raise RuntimeError(
                "GatedDeltaNet in_proj.grad is None; all-to-all detached the graph"
            )

    alloc = torch.cuda.memory_allocated(local_rank) / (1024 * 1024)
    peak = torch.cuda.max_memory_allocated(local_rank) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(local_rank) / (1024 * 1024)
    print(
        f"[gdn-smoke] rank={rank} params={n_params} loss={loss.item():.6f} "
        f"alloc={alloc:.1f}MiB peak={peak:.1f}MiB reserved={reserved:.1f}MiB "
        f"local_seq={tuple(tokens.shape)} logits={tuple(logits.shape)}",
        flush=True,
    )
    if peak > MEM_CAP_MIB + 1:
        raise RuntimeError(f"peak {peak:.1f} MiB exceeded cap {MEM_CAP_MIB} MiB")
    dist.barrier()
    if rank == 0:
        print("[gdn-smoke] PASS FSDP+CP GDN 2-GPU", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        raise SystemExit(1)
