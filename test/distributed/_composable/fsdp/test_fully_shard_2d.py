# Owner(s): ["oncall: distributed"]

import copy

from typing import List

import torch
import torch.nn as nn
from _test_fully_shard_common import MLP
from torch.distributed._composable import checkpoint, replicate

from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests


class TestFullyShard2D(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_parity_2d(self):
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [False, True],
                "mlp_dim": [3, 16, 17],
            },
            self._test_train_parity_2d,
        )

    def _test_train_parity_2d(
        self,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
        mlp_dim: int,
    ):
        # Prefer to test with >=4 GPUs, but for 2 GPUs, use 2-way TP
        dp_size = 2 if self.world_size > 2 else 1
        global_mesh = init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )
        dp_mesh = global_mesh["dp"]
        tp_mesh = global_mesh["tp"]
        dp_pg = dp_mesh.get_group()  # used for `replicate()`
        if self.rank == 0:
            print(f"global mesh: {global_mesh}")
            print(
                f"dp mesh size: {dp_mesh.size()} "
                f"ranks: {torch.distributed.distributed_c10d.get_process_group_ranks(dp_pg)}"
            )

        torch.manual_seed(42)
        model = nn.Sequential(
            # Use multiplier of 3 to exercise uneven case
            MLP(mlp_dim, torch.device("cpu"), dim_multiplier=3),
            MLP(mlp_dim, torch.device("cpu")),
            MLP(mlp_dim, torch.device("cpu"), dim_multiplier=3),
        )
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank], process_group=dp_pg)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        model = parallelize_module(
            model,
            device_mesh=tp_mesh,
            parallelize_plan={
                # Pass `use_local_output=False` to keep as DTensor to preserve
                # uneven activation dims
                "0.in_proj": ColwiseParallel(use_local_output=False),
                "0.out_proj": RowwiseParallel(use_local_output=False),
                "1.in_proj": ColwiseParallel(use_local_output=False),
                "1.out_proj": RowwiseParallel(use_local_output=False),
                "2.in_proj": ColwiseParallel(use_local_output=False),
                "2.out_proj": RowwiseParallel(),
            },
        )
        for mlp in model:
            if use_activation_checkpointing:
                checkpoint(mlp)
            fully_shard(mlp, mesh=dp_mesh, reshard_after_forward=reshard_after_forward)
        fully_shard(model, mesh=dp_mesh, reshard_after_forward=reshard_after_forward)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + dp_pg.rank() + 1)
        device = torch.device("cuda")
        for iter_idx in range(10):
            inp = torch.randn((8, mlp_dim), device=device)
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    run_tests()
