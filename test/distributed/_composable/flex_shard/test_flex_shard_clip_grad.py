# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._flex_shard import BucketSpec, clip_grad_norm_, flex_shard
from torch.distributed.fsdp._flex_shard.placement_contract import (
    GlobalLayout,
    set_global_layout,
)
from torch.distributed.fsdp._flex_shard.placements.block_shard import (
    make_bucketed_block_placement_fn,
)
from torch.distributed.fsdp._flex_shard.placements.owned import (
    make_bucketed_owned_full_param_placement_fn,
)
from torch.distributed.fsdp._flex_shard.placements.shard import per_param_placements
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import HardwareClassification, run_tests


device_type = torch.device(get_devtype())


def _net(hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(16, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
    )


class TestFlexShardClipGradNorm(FSDPTest):
    """``clip_grad_norm_`` on ``flex_shard`` parameters matches
    ``torch.nn.utils.clip_grad_norm_`` on the unsharded model."""

    hw_classification = HardwareClassification.CUDA

    @property
    def world_size(self) -> int:
        return min(4, torch.get_device_module(device_type).device_count())

    def _placement_fn(self, name: str):
        if name == "shard":
            return per_param_placements
        if name == "block":
            return make_bucketed_block_placement_fn(
                dims=(0,), blocks_per_rank=(1,) * self.world_size
            )
        return make_bucketed_owned_full_param_placement_fn()

    @skip_if_lt_x_gpu(2)
    def test_matches_unsharded_reference(self):
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        for placement in ("shard", "block", "owned"):
            for norm_type in (2.0, 1.0, float("inf")):
                with self.subTest(placement=placement, norm_type=norm_type):
                    torch.manual_seed(0)
                    # Shard and BucketedOwned get uneven rows (and owners get
                    # empty local grads); BucketedBlockShard needs data on
                    # every rank
                    model = _net(32 if placement == "block" else 33).to(device_type)
                    reference = copy.deepcopy(model)
                    # A parameter that flex_shard does not shard counts once
                    scale = nn.Parameter(torch.randn(16, device=device_type))
                    ref_scale = nn.Parameter(scale.detach().clone())
                    placement_fn = self._placement_fn(placement)
                    flex_shard(
                        model,
                        buckets=[
                            BucketSpec([f"{idx}.*"], placement_fn, mesh=mesh)
                            for idx in (0, 2, 4)
                        ],
                    )
                    # The same input on every rank, so the averaged sharded
                    # grads are the reference's
                    inp = torch.randn(4, 16, device=device_type)
                    (model(inp) * scale).sum().backward()
                    (reference(inp) * ref_scale).sum().backward()
                    params = [*model.parameters(), scale]
                    grads = [p.grad.clone() for p in params]
                    max_norm = 0.1
                    ref_norm = torch.nn.utils.clip_grad_norm_(
                        [*reference.parameters(), ref_scale], max_norm, norm_type
                    )
                    norm = clip_grad_norm_(params, max_norm, norm_type)
                    self.assertEqual(norm, ref_norm)
                    # Every rank scales its local grads by the reference's factor
                    coef = max_norm / (ref_norm + 1e-6)
                    for grad, param in zip(grads, params, strict=True):
                        self.assertEqual(param.grad, grad * coef)

    @skip_if_lt_x_gpu(2)
    def test_rejects_outer_layouts(self):
        # A parameter that is itself a part of a bigger one (tensor parallelism)
        # needs its norm combined over ranks outside the bucket's mesh
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        model = nn.Linear(8, 8, bias=False, device=device_type)
        set_global_layout(
            model.weight,
            GlobalLayout(
                global_shape=(16, 8),
                global_offsets=((0, 0),),
                local_offsets=((0, 0),),
                local_sizes=((8, 8),),
            ),
        )
        flex_shard(model, buckets=[BucketSpec(["*"], per_param_placements, mesh=mesh)])
        model(torch.randn(2, 8, device=device_type)).sum().backward()
        with self.assertRaisesRegex(NotImplementedError, "outer"):
            clip_grad_norm_(model.parameters(), 1.0)


if __name__ == "__main__":
    run_tests()
