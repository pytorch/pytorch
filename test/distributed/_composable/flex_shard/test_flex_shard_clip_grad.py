# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._flex_shard import (
    BucketSpec,
    clip_grad_norm_,
    flex_shard,
    get_flex_shard_global_layouts,
)
from torch.distributed.fsdp._flex_shard.layout_adapters import dtensor_to_global_layout
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
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
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


class _OuterParallel(nn.Module):
    """Parameters as tensor and expert parallelism split them: column- and
    row-wise linear weights, stacked expert weights and a replicated bias."""

    def __init__(self) -> None:
        super().__init__()
        self.colwise = nn.Parameter(torch.randn(8, 6))
        self.rowwise = nn.Parameter(torch.randn(6, 8))
        self.experts = nn.Parameter(torch.randn(4, 3, 5))
        self.bias = nn.Parameter(torch.randn(6))


def _slices(offsets: tuple[int, ...], sizes: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(o, o + s) for o, s in zip(offsets, sizes, strict=True))


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

    @skip_if_lt_x_gpu(4)
    def test_matches_unsharded_reference_with_outer_layouts(self):
        # Tensor and expert parallelism over "outer", flex_shard over "dp":
        # each rank holds the dp shard of its outer part, so the norm combines
        # over both, but not over the replicas of the bias
        mesh = init_device_mesh(
            device_type.type, (2, 2), mesh_dim_names=("outer", "dp")
        )
        outer_placements = {
            "colwise": Shard(0),
            "rowwise": Shard(1),
            "experts": Shard(0),
            "bias": Replicate(),
        }
        for norm_type in (2.0, 1.0, float("inf")):
            with self.subTest(norm_type=norm_type):
                torch.manual_seed(0)
                model = _OuterParallel().to(device_type)
                full_grads = {
                    name: torch.randn_like(param)
                    for name, param in model.named_parameters()
                }
                for name, placement in outer_placements.items():
                    full = getattr(model, name).detach()
                    dtensor = distribute_tensor(full, mesh["outer"], [placement])
                    setattr(model, name, nn.Parameter(dtensor))
                dtensor_to_global_layout(model)
                flex_shard(
                    model,
                    buckets=[BucketSpec(["*"], per_param_placements, mesh=mesh["dp"])],
                )
                # Each local grad is the part of the full grad that the
                # parameter's local shard holds
                layouts = get_flex_shard_global_layouts(model)
                grads = {}
                for name, param in model.named_parameters():
                    grad = torch.zeros_like(param)
                    layout = layouts[name]
                    for global_offsets, local_offsets, sizes in zip(
                        layout.global_offsets,
                        layout.local_offsets,
                        layout.local_sizes,
                        strict=True,
                    ):
                        grad[_slices(local_offsets, sizes)] = full_grads[name][
                            _slices(global_offsets, sizes)
                        ]
                    param.grad = grad
                    grads[name] = grad.clone()
                max_norm = 0.1
                ref_norm = torch.nn.utils.get_total_norm(
                    list(full_grads.values()), norm_type
                )
                norm = clip_grad_norm_(model.parameters(), max_norm, norm_type)
                self.assertEqual(norm, ref_norm)
                coef = max_norm / (ref_norm + 1e-6)
                for name, param in model.named_parameters():
                    self.assertEqual(param.grad, grads[name] * coef)

    @skip_if_lt_x_gpu(2)
    def test_outer_layout_needs_its_shard_groups(self):
        # A part of a bigger parameter whose outer groups are not declared
        # cannot be combined with the parts of the other outer ranks
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
        with self.assertRaisesRegex(ValueError, "set_outer_shard_groups"):
            clip_grad_norm_(model.parameters(), 1.0)


if __name__ == "__main__":
    run_tests()
