#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import datetime
import os
import unittest

import torch
import torch.comms
import torch.nn as nn
import torch.nn.functional as F
from torch.comms.device_mesh import init_device_mesh
from torch.distributed._composable import checkpoint
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
    SequenceParallel,
)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        device: torch.device | None = None,
        *,
        bias: bool = True,
        dim_multiplier: int = 4,
    ):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim_multiplier * dim, device=device, bias=bias)
        self.out_proj = nn.Linear(dim_multiplier * dim, dim, device=device, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = F.relu(z)
        z = self.out_proj(z)
        z = F.relu(z)
        return z


class MLPStack(nn.Sequential):
    def __init__(self, mlp_dim: int, *, with_seq_parallel: bool = False):
        modules: list[nn.Module] = [
            MLP(mlp_dim, dim_multiplier=4),
            MLP(mlp_dim),
            MLP(mlp_dim, dim_multiplier=4),
        ]
        if with_seq_parallel:
            modules.append(nn.LayerNorm(mlp_dim, bias=False))
        super().__init__(*modules)
        self.with_seq_parallel = with_seq_parallel

    def parallelize(
        self,
        tp_mesh: DeviceMesh,
        dp_mesh: DeviceMesh,
        use_activation_checkpointing: bool,
        **fsdp_kwargs,
    ) -> "MLPStack":
        parallelize_plan = {
            # Pass `use_local_output=False` to keep as DTensor to preserve
            # uneven activation dims
            "0.in_proj": ColwiseParallel(use_local_output=False),
            "0.out_proj": RowwiseParallel(use_local_output=False),
            "1.in_proj": ColwiseParallel(use_local_output=False),
            "1.out_proj": RowwiseParallel(use_local_output=False),
            "2.in_proj": ColwiseParallel(use_local_output=False),
            "2.out_proj": RowwiseParallel(output_layouts=Shard(1))
            if self.with_seq_parallel
            else RowwiseParallel(),
        }
        if self.with_seq_parallel:
            parallelize_plan["3"] = SequenceParallel(sequence_dim=1)  # pyre-ignore[6]
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=parallelize_plan,  # pyre-ignore[6]
        )
        for module in self:
            if isinstance(module, nn.LayerNorm):
                continue
            if use_activation_checkpointing:
                checkpoint(module)
            fully_shard(module, mesh=dp_mesh, **fsdp_kwargs)
        fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)
        return self


class DPTPCommTest(unittest.TestCase):
    @unittest.skipIf(
        torch.accelerator.device_count() < 4, "Skipping non GPU situations for now"
    )
    def test_training(self) -> None:
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        comm = torch.comms.new_comm(
            backend,
            device,
            name="comms_test_global",
            timeout=datetime.timedelta(seconds=360),
        )
        world_size = comm.get_size()
        dp_degree = 2
        tp_degree = world_size // dp_degree
        mesh = torch.arange(world_size, dtype=torch.int, device="cpu").view(
            dp_degree, tp_degree
        )
        # Get current rank to determine which groups this rank belongs to
        cur_rank = comm.get_rank()

        # For TP communication: find which row contains current rank
        tp_ranks = None
        for row in mesh.tolist():
            if cur_rank in row:
                tp_ranks = row
                break

        # For DP communication: find which column contains current rank
        dp_ranks = None
        mesh_transposed = mesh.transpose(0, 1)
        for col in mesh_transposed.tolist():
            if cur_rank in col:
                dp_ranks = col
                break

        # Create communicators using the new single-list API
        tp_comm = comm.split(tp_ranks, "tp")
        dp_comm = comm.split(dp_ranks, "dp")

        try:
            device_mesh_2d = init_device_mesh(
                mesh_dim_comms=(dp_comm, tp_comm),
                mesh_dim_names=("dp", "tp"),
                _global_comm=comm,
            )
            dp_pg = device_mesh_2d.get_group("dp")
        except TypeError as e:
            # TODO: remove this once PT 2.10 is released
            if "_rank" in str(e):
                tp_comm.finalize()
                dp_comm.finalize()
                comm.finalize()
                return
            raise

        mlp_dim = 16
        LR = 1e-4
        torch.manual_seed(42)
        model = MLPStack(mlp_dim).to(device)

        ref_model = copy.deepcopy(model).to(device)
        model.parallelize(
            device_mesh_2d["tp"],
            device_mesh_2d["dp"],
            False,
            reshard_after_forward=False,
        )
        # Need data parallel wrapper to sync gradients for the ref model.
        for layer in ref_model:
            fully_shard(layer, mesh=device_mesh_2d["dp"])
        fully_shard(ref_model, mesh=device_mesh_2d["dp"])
        optim = torch.optim.Adam(model.parameters(), lr=LR, foreach=False)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=LR, foreach=False)

        torch.manual_seed(42 + dp_pg.rank() + 1)
        for iter_idx in range(10):
            inp = torch.randn((8, mlp_dim), device=device)
            losses: list[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            assert torch.allclose(losses[0], losses[1], atol=1e-7, rtol=1e-5)
        # Somehow if not sync here, calling finalize will see abort reason.
        torch.accelerator.synchronize()
        tp_comm.finalize()
        dp_comm.finalize()
        comm.finalize()


if __name__ == "__main__":
    unittest.main()
