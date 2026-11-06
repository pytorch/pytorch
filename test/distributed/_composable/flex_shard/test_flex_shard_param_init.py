# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp._flex_shard import (
    BucketSpec,
    flex_shard,
    get_flex_shard_global_layouts,
    GlobalLayout,
    init_from_full,
    PlacementFn,
)
from torch.distributed.fsdp._flex_shard.layout_adapters import dtensor_to_global_layout
from torch.distributed.fsdp._flex_shard.placements import (
    make_bucketed_block_placement_fn,
    make_bucketed_owned_expert_block_placement_fn,
    per_param_placements,
)
from torch.distributed.tensor import DTensor, Shard as DTensorShard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.common_flex_shard import expected_shard


device_type = torch.device(get_devtype())
SEED = 1234

InitFns = dict[str, Callable[[torch.Tensor], object]]


class _MixedInitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.even = nn.Linear(64, 256)
        # 30 rows over 4 ranks: 8, 8, 8, 6.
        self.uneven = nn.Linear(256, 30, bias=False)
        # 3 rows over 4 ranks leaves rank 3 an empty shard.
        self.tiny = nn.Parameter(torch.empty(3, 30))


_MIXED_INITS: InitFns = {
    # nn.Linear.reset_parameters() as tensor functions; kaiming reads the
    # fan-in from the full shape.
    "even.weight": partial(nn.init.kaiming_uniform_, a=math.sqrt(5)),
    "even.bias": partial(nn.init.uniform_, a=-(64**-0.5), b=64**-0.5),
    "uneven.weight": partial(nn.init.trunc_normal_, std=0.02),
    "tiny": nn.init.normal_,
}


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(8, 32, 16))
        self.w3 = nn.Parameter(torch.empty(8, 32, 16))
        self.w2 = nn.Parameter(torch.empty(8, 16, 32))


class _ExpertModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _Experts()


_EXPERT_INITS: InitFns = {
    "experts.w1": partial(nn.init.trunc_normal_, std=0.02),
    "experts.w3": partial(nn.init.trunc_normal_, std=0.02),
    "experts.w2": partial(nn.init.normal_, std=0.01),
}


class _TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(30, 16)
        self.head = nn.Linear(16, 30, bias=False)
        self.head.weight = self.embed.weight


def _one_bucket(
    mesh: DeviceMesh, placement_fn: PlacementFn = per_param_placements
) -> list[BucketSpec]:
    return [BucketSpec(["*"], placement_fn=placement_fn, mesh=mesh)]


def _single_device_init(model_cls, inits: InitFns) -> nn.Module:
    torch.manual_seed(SEED)
    with torch.device("meta"):
        model = model_cls()
    model.to_empty(device=device_type)
    with torch.no_grad():
        for fqn, param in model.named_parameters():
            inits[fqn](param)
    return model


def _init_from_full(model: nn.Module, inits: InitFns) -> None:
    torch.manual_seed(SEED)
    layouts = get_flex_shard_global_layouts(model)
    for fqn, param in model.named_parameters():
        init_from_full(param, layouts[fqn], inits[fqn])


class InitFromFullTest(TestCase):
    def test_copies_regions_of_one_full_draw(self):
        # Rows 0 and 2-3 of a [4, 6] parameter, stacked into one local tensor.
        layout = GlobalLayout(
            (4, 6), ((0, 0), (2, 0)), ((0, 0), (1, 0)), ((1, 6), (2, 6))
        )
        torch.manual_seed(SEED)
        full = torch.empty(4, 6).normal_()
        next_draw = torch.rand(4)

        torch.manual_seed(SEED)
        local = torch.empty(3, 6)
        init_from_full(local, layout, nn.init.normal_)

        self.assertEqual(local, torch.cat([full[0:1], full[2:4]]))
        self.assertEqual(torch.rand(4), next_draw)


class InitFromFullDistributedTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return 4

    def _mesh(self) -> DeviceMesh:
        return init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )

    @skip_if_lt_x_gpu(4)
    def test_init_matches_single_device(self):
        mesh = self._mesh()
        # init_from_full writes through each placement's global_layout, while
        # the expected shards are packed by its copy_param_to_storage.
        cases = {
            "shard0": (_MixedInitModel, _MIXED_INITS, per_param_placements),
            "bucketed_block": (
                _ExpertModel,
                _EXPERT_INITS,
                make_bucketed_block_placement_fn(
                    dims=(0,), blocks_per_rank=(3, 2, 2, 1)
                ),
            ),
            "bucketed_owned": (
                _ExpertModel,
                _EXPERT_INITS,
                make_bucketed_owned_expert_block_placement_fn(),
            ),
        }
        for name, (model_cls, inits, placement_fn) in cases.items():
            with self.subTest(placement=name):
                reference = _single_device_init(model_cls, inits)
                reference_next_draw = torch.rand(8, device=device_type)

                with torch.device("meta"):
                    model = model_cls()
                flex_shard(model, buckets=_one_bucket(mesh, placement_fn))
                model.to_empty(device=device_type)
                _init_from_full(model, inits)

                # Every rank drew the full tensors, so its RNG is where one
                # device's is.
                self.assertEqual(torch.rand(8, device=device_type), reference_next_draw)
                expected = flex_shard(
                    copy.deepcopy(reference), buckets=_one_bucket(mesh, placement_fn)
                )
                for fqn, param in model.named_parameters():
                    self.assertEqual(param, expected.get_parameter(fqn), msg=fqn)

    @skip_if_lt_x_gpu(4)
    def test_shared_param_init_through_later_name(self):
        mesh = self._mesh()
        torch.manual_seed(SEED)
        with torch.device("meta"):
            reference = _TiedModel()
        reference.to_empty(device=device_type)
        with torch.no_grad():
            nn.init.trunc_normal_(reference.head.weight, std=0.02)

        with torch.device("meta"):
            model = _TiedModel()
        flex_shard(model, buckets=_one_bucket(mesh))
        model.to_empty(device=device_type)
        torch.manual_seed(SEED)
        init_from_full(
            model.head.weight,
            get_flex_shard_global_layouts(model)["head.weight"],
            partial(nn.init.trunc_normal_, std=0.02),
        )

        self.assertIs(model.head.weight, model.embed.weight)
        self.assertEqual(
            model.embed.weight,
            expected_shard(
                reference.head.weight,
                rank=mesh.get_local_rank(),
                world_size=self.world_size,
            ),
        )

    @skip_if_lt_x_gpu(4)
    def test_outer_layout_init_matches_single_device(self):
        mesh = init_device_mesh(
            device_type.type, (2, 2), mesh_dim_names=("ep", "efsdp")
        )
        ep_mesh, efsdp_mesh = mesh["ep"], mesh["efsdp"]
        reference = _single_device_init(_ExpertModel, _EXPERT_INITS)

        with torch.device("meta"):
            model = _ExpertModel()
        for name, param in list(model.experts.named_parameters()):
            local = torch.empty(param.shape[0] // 2, *param.shape[1:], device="meta")
            ep_param = DTensor.from_local(
                local,
                ep_mesh,
                [DTensorShard(0)],
                run_check=False,
                shape=param.shape,
                stride=param.stride(),
            )
            setattr(model.experts, name, nn.Parameter(ep_param))
        dtensor_to_global_layout(model)
        flex_shard(model, buckets=_one_bucket(efsdp_mesh))
        model.to_empty(device=device_type)
        _init_from_full(model, _EXPERT_INITS)

        # The full tensor has all 8 experts: EP slices it, then FlexShard does.
        for fqn, ref_param in reference.named_parameters():
            ep_local = expected_shard(
                ref_param, rank=ep_mesh.get_local_rank(), world_size=2
            )
            self.assertEqual(
                model.get_parameter(fqn),
                expected_shard(
                    ep_local, rank=efsdp_mesh.get_local_rank(), world_size=2
                ),
                msg=fqn,
            )


if __name__ == "__main__":
    run_tests()
