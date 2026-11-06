# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._flex_shard import (
    BucketSpec,
    flex_shard,
    get_global_layout,
    get_outer_layout,
    GlobalLayout,
    MixedPrecisionPolicy,
    set_global_layout,
    set_partial_grad_group,
)
from torch.distributed.fsdp._flex_shard.bucket_storage import ShardedBucketStorage
from torch.distributed.fsdp._flex_shard.layout_adapters import dtensor_to_global_layout
from torch.distributed.fsdp._flex_shard.placements.shard import (
    per_param_placements,
    Shard as FlexShardShard,
)
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.common_flex_shard import (
    expected_shard,
    single_rank_cpu_mesh,
)


device_type = torch.device(get_devtype())


class TestFlexShardOuterLayout(TestCase):
    def test_dtensor_adapter_declares_layout_and_preserves_grad_dtype(self) -> None:
        model = nn.Module()
        grad_dtypes = {"fp32_grad": torch.float32, "none_grad": None}
        with single_rank_cpu_mesh() as mesh:
            for name, grad_dtype in grad_dtypes.items():
                param = nn.Parameter(
                    DTensor.from_local(
                        torch.ones(4, 3, dtype=torch.bfloat16), mesh, [Shard(0)]
                    )
                )
                param.grad_dtype = grad_dtype
                setattr(model, name, param)
            dtensor_to_global_layout(model)
            param_infos, _ = ShardedBucketStorage.create_param_infos(
                list(model.named_parameters()),
                mesh,
                {name: (FlexShardShard(0),) for name in grad_dtypes},
            )
        for name, grad_dtype in grad_dtypes.items():
            param = model._parameters[name]
            self.assertNotIsInstance(param, DTensor)
            self.assertEqual(
                get_global_layout(param),
                GlobalLayout((4, 3), ((0, 0),), ((0, 0),), ((4, 3),)),
            )
            self.assertEqual(param_infos[name].unsharded_grad_dtype, grad_dtype)


class _PartialGradGroup(FSDPTest):
    def _check_partial_grad_group_sums_before_reduce_scatter(
        self, dp: int, tp: int
    ) -> None:
        # The weight's grad is partial over tp, as a norm weight's is under
        # sequence parallelism: each rank feeds its own inputs. The bias
        # declares nothing, so only dp sums it. With bf16 compute and fp32
        # grads, the weight's grad is summed over tp in fp32, as FSDP2 does.
        mesh = init_device_mesh(device_type.type, (dp, tp), mesh_dim_names=("dp", "tp"))
        dp_rank, tp_rank = mesh["dp"].get_local_rank(), mesh["tp"].get_local_rank()
        for compute_dtype in (torch.float32, torch.bfloat16):
            with self.subTest(compute_dtype=compute_dtype):
                self._check_partial_grad_sums(
                    mesh, dp_rank, tp_rank, dp, tp, compute_dtype
                )

    def _check_partial_grad_sums(
        self, mesh, dp_rank: int, tp_rank: int, dp: int, tp: int, compute_dtype
    ) -> None:
        torch.manual_seed(0)
        model = nn.Linear(8, 6, device=device_type)
        # The unsharded params and their local grads, in the compute dtype.
        reference = copy.deepcopy(model).to(compute_dtype)
        set_partial_grad_group(model.weight, mesh["tp"].get_group())
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=per_param_placements,
                    mesh=mesh["dp"],
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=compute_dtype, reduce_dtype=torch.float32
                    ),
                    gradient_divide_factor=1.0,
                )
            ],
        )
        generator = torch.Generator().manual_seed(self.rank)
        x = torch.randn(4, 8, generator=generator).to(device_type, compute_dtype)
        model(x).sum().backward()
        reference(x).sum().backward()

        local_grads = [reference.weight.grad.float(), reference.bias.grad.float()]
        gathered = [
            [torch.empty_like(grad) for _ in range(self.world_size)]
            for grad in local_grads
        ]
        for grads, grad in zip(gathered, local_grads, strict=True):
            dist.all_gather(grads, grad)
        weight_grads, bias_grads = (
            [grads[d * tp : (d + 1) * tp] for d in range(dp)] for grads in gathered
        )
        # Each sum has at most two terms, so it is exact in any order.
        expected_weight = sum(sum(grads[1:], grads[0]) for grads in weight_grads)
        expected_bias = sum(grads[tp_rank] for grads in bias_grads)
        for param, expected in (
            (model.weight, expected_weight),
            (model.bias, expected_bias),
        ):
            self.assertTrue(
                torch.equal(
                    param.grad,
                    expected_shard(expected, rank=dp_rank, world_size=dp),
                )
            )


class TestFlexShardPartialGradGroupTwoRanks(_PartialGradGroup):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_partial_grad_group_sums_before_reduce_scatter(self) -> None:
        self._check_partial_grad_group_sums_before_reduce_scatter(dp=1, tp=2)

    @skip_if_lt_x_gpu(2)
    def test_outer_layout_survives_to_empty(self) -> None:
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        with torch.device("meta"):
            model = nn.Linear(8, 6)
        layout = GlobalLayout((12, 8), ((6, 0),), ((0, 0),), ((6, 8),))
        set_global_layout(model.weight, layout)
        flex_shard(
            model,
            buckets=[BucketSpec(["*"], placement_fn=per_param_placements, mesh=mesh)],
        )
        self.assertEqual(get_outer_layout(model.weight), layout)
        model.to_empty(device=device_type)
        self.assertEqual(model.weight.device.type, device_type.type)
        self.assertEqual(get_outer_layout(model.weight), layout)
        self.assertIsNone(get_outer_layout(model.bias))


class TestFlexShardPartialGradGroupFourRanks(_PartialGradGroup):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_partial_grad_group_sums_before_reduce_scatter(self) -> None:
        self._check_partial_grad_group_sums_before_reduce_scatter(dp=2, tp=2)


if __name__ == "__main__":
    run_tests()
