# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._flex_shard import BucketSpec, flex_shard, GradientReduction
from torch.distributed.fsdp._flex_shard.bucket_storage import (
    ParamInfo,
    ShardedBucketStorage,
)
from torch.distributed.fsdp._flex_shard.placements import (
    BlockShard,
    BucketedBlockShard,
    make_bucketed_block_placement_fn,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    FSDPTestMultiThread,
    get_devtype,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed.common_flex_shard import single_rank_cpu_mesh


device_type = torch.device(get_devtype())


def _expected_bucketed_local(tensor: torch.Tensor, info: ParamInfo) -> torch.Tensor:
    if info.bucket_layout is None:
        raise AssertionError(f"Expected a bucket layout for {info.fqn}")
    param_layout = info.bucket_layout.param_layouts[info.fqn]
    start = param_layout.local_global_offset - param_layout.param_offset
    return (
        tensor.contiguous()
        .view(-1)[start : start + info.local_numel]
        .view(info.local_shape)
    )


class _TinyBlockModule(nn.Module):
    def __init__(self, device: torch.device | str) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(16, dtype=torch.float32, device=device).view(4, 4)
        )
        self.bias = nn.Parameter(torch.arange(4, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t() + self.bias


class TestBucketedBlockShardPlacement(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_equality_hash_and_repr_include_bucketed_block_semantics(self):
        placement = BucketedBlockShard(dims=(0,), blocks_per_rank=(1, 2))

        self.assertEqual(
            placement,
            BucketedBlockShard(dims=(0,), blocks_per_rank=(1, 2)),
        )
        self.assertNotEqual(
            placement,
            BucketedBlockShard(dims=(0,), blocks_per_rank=(2, 1)),
        )
        self.assertEqual(hash(placement), hash(BucketedBlockShard((0,), (1, 2))))
        self.assertEqual(
            repr(placement),
            "BucketedBlockShard(dims=(0,), blocks_per_rank=(1, 2))",
        )

    def test_rejects_invalid_config_and_shapes(self):
        with self.assertRaisesRegex(ValueError, "prefix dims"):
            BucketedBlockShard(dims=(1,), blocks_per_rank=(1, 1))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            BucketedBlockShard(dims=(0,), blocks_per_rank=(1, -1))
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            BucketedBlockShard(dims=(0,), blocks_per_rank=(0, 0))

        with single_rank_cpu_mesh() as mesh:
            placement = BucketedBlockShard(dims=(0, 1), blocks_per_rank=(1,))
            weight = nn.Parameter(torch.empty(2))
            with self.assertRaisesRegex(ValueError, "invalid for parameter shape"):
                placement.bucket_storage_layout(
                    [("weight", weight)],
                    {"weight": (placement,)},
                    mesh,
                )

            placement = BucketedBlockShard(dims=(0,), blocks_per_rank=(1, 1))
            weight = nn.Parameter(torch.empty(2, 2))
            with self.assertRaisesRegex(ValueError, "world size"):
                placement.bucket_storage_layout(
                    [("weight", weight)],
                    {"weight": (placement,)},
                    mesh,
                )

    def test_bucketed_block_placement_fn_uses_mesh_size(self):
        with single_rank_cpu_mesh() as mesh:
            weight = nn.Parameter(torch.empty(2, 2))
            placements = make_bucketed_block_placement_fn(
                dims=(0,),
                blocks_per_rank=(1,),
            )([("weight", weight)], mesh)

        self.assertEqual(placements["weight"], (BucketedBlockShard((0,), (1,)),))

    def test_rejects_mixed_bucketed_block_placements_in_one_bucket(self):
        from torch.distributed.fsdp._flex_shard.utils import (
            _validate_bucket_uniform_dtype_and_placement,
        )

        assignments = [["weight", "bias"]]
        placements = {
            "weight": (BucketedBlockShard(dims=(0,), blocks_per_rank=(1, 3)),),
            "bias": (BucketedBlockShard(dims=(0,), blocks_per_rank=(2, 2)),),
        }
        with single_rank_cpu_mesh() as mesh:
            buckets = [
                BucketSpec(
                    ["*"],
                    placement_fn=make_bucketed_block_placement_fn(
                        dims=(0,),
                        blocks_per_rank=(1, 3),
                    ),
                    mesh=mesh,
                    reshard_after_forward=True,
                )
            ]
        named_params = [
            ("weight", nn.Parameter(torch.empty(4, 4))),
            ("bias", nn.Parameter(torch.empty(4))),
        ]

        with self.assertRaisesRegex(ValueError, "mixed placements"):
            _validate_bucket_uniform_dtype_and_placement(
                assignments,
                placements,
                buckets,
                named_params,
            )


class TestBucketedBlockShardDistributed(FSDPTestMultiThread):
    hw_classification = HardwareClassification.GENERIC

    @property
    def world_size(self) -> int:
        return 2

    def _mesh(self):
        return init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))

    def test_bucketed_block_bucket_materializes_bucket_global_local_views(self):
        mesh = self._mesh()
        placement = BucketedBlockShard(dims=(0,), blocks_per_rank=(1, 3))

        class TinyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.arange(8, dtype=torch.float32).view(4, 2)
                )
                self.bias = nn.Parameter(torch.arange(4, dtype=torch.float32))

        model = TinyModule()
        named_params = list(model.named_parameters())
        original_params = {fqn: param.detach().clone() for fqn, param in named_params}
        placements = {fqn: (placement,) for fqn, _ in named_params}
        bucket_spec = BucketSpec(
            ["*"],
            placement_fn=make_bucketed_block_placement_fn(
                dims=(0,),
                blocks_per_rank=(1, 3),
            ),
            mesh=mesh,
            reshard_after_forward=False,
        )

        bucket_storage = ShardedBucketStorage.from_bucket(
            model,
            named_params,
            placements,
            mesh,
            torch.device("cpu"),
            bucket_spec,
        )

        current_params = dict(model.named_parameters())
        infos = bucket_storage.param_infos
        for fqn, info in infos.items():
            expected = _expected_bucketed_local(original_params[fqn], info)
            self.assertEqual(current_params[fqn], expected)
            self.assertEqual(bucket_storage.get_local_view(fqn), expected)
            if expected.numel() > 0:
                self.assertEqual(
                    current_params[fqn].untyped_storage().data_ptr(),
                    bucket_storage.byte_storage.untyped_storage().data_ptr(),
                )

    def test_bucketed_block_rejects_padding_only_local_bucket_range(self):
        mesh = self._mesh()
        placement = BucketedBlockShard(dims=(0,), blocks_per_rank=(1, 1))

        class TinyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.arange(4, dtype=torch.float32).view(1, 4)
                )

        model = TinyModule()
        named_params = list(model.named_parameters())
        placements = {fqn: (placement,) for fqn, _ in named_params}
        bucket_spec = BucketSpec(
            ["*"],
            placement_fn=make_bucketed_block_placement_fn(
                dims=(0,),
                blocks_per_rank=(1, 1),
            ),
            mesh=mesh,
            reshard_after_forward=False,
        )

        if self.rank == 0:
            bucket_storage = ShardedBucketStorage.from_bucket(
                model,
                named_params,
                placements,
                mesh,
                torch.device("cpu"),
                bucket_spec,
            )
            self.assertEqual(bucket_storage.total_bytes, 4 * torch.float32.itemsize)
        else:
            with self.assertRaisesRegex(ValueError, "contains only padding"):
                ShardedBucketStorage.from_bucket(
                    model,
                    named_params,
                    placements,
                    mesh,
                    torch.device("cpu"),
                    bucket_spec,
                )

    def test_bucketed_block_unshard_is_view_in_and_view_out(self):
        mesh = self._mesh()
        placement = BucketedBlockShard(dims=(0,), blocks_per_rank=(1, 3))

        class TinyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.arange(8, dtype=torch.float32).view(4, 2)
                )
                self.bias = nn.Parameter(torch.arange(4, dtype=torch.float32))

        model = TinyModule()
        named_params = list(model.named_parameters())
        original_params = {fqn: param.detach().clone() for fqn, param in named_params}
        placements = {fqn: (placement,) for fqn, _ in named_params}
        bucket_spec = BucketSpec(
            ["*"],
            placement_fn=make_bucketed_block_placement_fn(
                dims=(0,),
                blocks_per_rank=(1, 3),
            ),
            mesh=mesh,
            reshard_after_forward=False,
        )
        bucket_storage = ShardedBucketStorage.from_bucket(
            model,
            named_params,
            placements,
            mesh,
            torch.device("cpu"),
            bucket_spec,
        )
        infos = [bucket_storage.param_infos[fqn] for fqn, _ in named_params]
        local_shards = [bucket_storage.get_local_view(fqn) for fqn, _ in named_params]

        prepared = placement.prepare_unshard_bucket(local_shards, infos, mesh, None)
        send_buf = prepared.buffers[0]
        rank_numels = infos[0].bucket_layout.rank_numels
        if rank_numels[mesh.get_local_rank()] == max(rank_numels):
            # Smaller ranges are sent zero-padded to the largest one.
            self.assertEqual(
                send_buf.untyped_storage().data_ptr(),
                bucket_storage.byte_storage.untyped_storage().data_ptr(),
            )
            self.assertEqual(
                send_buf.data_ptr(), bucket_storage.byte_storage.data_ptr()
            )

        placement.run_prepared_unshard(prepared)
        result = placement.finish_prepared_unshard(prepared).full_params
        gathered_bucket = prepared.buffers[1]

        for full_param, (fqn, original_param) in zip(
            result,
            named_params,
            strict=True,
        ):
            self.assertEqual(full_param, original_params[fqn])
            self.assertEqual(
                full_param.untyped_storage().data_ptr(),
                gathered_bucket.untyped_storage().data_ptr(),
            )
            self.assertNotEqual(full_param.data_ptr(), original_param.data_ptr())

    def test_bucketed_block_reduce_scatter_returns_local_grad_views(self):
        mesh = self._mesh()
        placement = BucketedBlockShard(dims=(0,), blocks_per_rank=(1, 3))

        class TinyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty(4, 2))
                self.bias = nn.Parameter(torch.empty(4))

        model = TinyModule()
        named_params = list(model.named_parameters())
        placements = {fqn: (placement,) for fqn, _ in named_params}
        bucket_spec = BucketSpec(
            ["*"],
            placement_fn=make_bucketed_block_placement_fn(
                dims=(0,),
                blocks_per_rank=(1, 3),
            ),
            mesh=mesh,
            reshard_after_forward=False,
        )
        bucket_storage = ShardedBucketStorage.from_bucket(
            model,
            named_params,
            placements,
            mesh,
            torch.device("cpu"),
            bucket_spec,
        )
        weight_grad = torch.arange(8, dtype=torch.float32).view(4, 2)
        bias_grad = torch.arange(4, dtype=torch.float32)
        grads = [weight_grad, bias_grad]
        infos = [bucket_storage.param_infos[fqn] for fqn, _ in named_params]

        prepared = placement.prepare_reduce_grad(grads, infos, mesh, None)
        reduce_result = placement.reduce_prepared_grad(prepared, GradientReduction())
        sharded_grads = reduce_result.sharded_grads
        recv_buf = reduce_result.buffers[0]

        for grad, sharded_grad, info in zip(grads, sharded_grads, infos, strict=True):
            self.assertEqual(sharded_grad, _expected_bucketed_local(grad, info))
            if sharded_grad.numel() > 0:
                self.assertEqual(
                    sharded_grad.untyped_storage().data_ptr(),
                    recv_buf.untyped_storage().data_ptr(),
                )


class TestBlockShardPlacement(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_local_shape_preserves_parameter_rank(self):
        placement = BlockShard(blocks_per_rank=(2, 1))

        self.assertEqual(
            placement.compute_local_shape(torch.Size([18, 4]), 0, 2),
            (12, 4),
        )
        self.assertEqual(
            placement.compute_local_shape(torch.Size([18, 4]), 1, 2),
            (6, 4),
        )


class TestBlockShardRuntime(FSDPTest):
    hw_classification = HardwareClassification.CUDA

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_uneven_adamw_step_matches_unsharded_reference(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        x = torch.arange(8, dtype=torch.float32, device=device_type).view(2, 4)
        for blocks_per_rank in ((1, 0), (1, 3)):
            with self.subTest(blocks_per_rank=blocks_per_rank):
                placement = BlockShard(blocks_per_rank=blocks_per_rank)
                reference = _TinyBlockModule(device_type.type)
                model = _TinyBlockModule(device_type.type)

                def placement_fn(named_params, mesh, placement=placement):
                    del mesh
                    return {fqn: (placement,) for fqn, _ in named_params}

                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=placement_fn,
                            mesh=mesh,
                            reshard_after_forward=True,
                        )
                    ],
                )
                reference_optimizer = torch.optim.AdamW(
                    reference.parameters(), lr=0.01, foreach=True
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=True)

                reference_output = reference(x)
                output = model(x)
                self.assertEqual(output, reference_output)
                reference_output.sum().backward()
                output.sum().backward()

                reference_optimizer.step()
                with mock.patch.multiple(
                    dist,
                    all_gather_single=mock.DEFAULT,
                    all_gather_into_tensor=mock.DEFAULT,
                    reduce_scatter_tensor=mock.DEFAULT,
                    reduce_scatter_single=mock.DEFAULT,
                ) as collectives:
                    optimizer.step()
                self.assertTrue(
                    all(not collective.called for collective in collectives.values())
                )

                reference_params = dict(reference.named_parameters())
                for fqn, param in model.named_parameters():
                    reference_param = reference_params[fqn]
                    block_rows = reference_param.size(0) // sum(blocks_per_rank)
                    start = sum(blocks_per_rank[: self.rank]) * block_rows
                    rows = blocks_per_rank[self.rank] * block_rows
                    expected = reference_param.detach()[start : start + rows]
                    self.assertEqual(param.detach(), expected)

                    local_state = optimizer.state[param]
                    reference_state = reference_optimizer.state[reference_param]
                    for state_name in ("exp_avg", "exp_avg_sq"):
                        self.assertEqual(
                            local_state[state_name],
                            reference_state[state_name][start : start + rows],
                        )
                    self.assertEqual(local_state["step"], reference_state["step"])
                self.assertEqual(model(x), reference(x))


class TestBucketedBlockShardRuntime(FSDPTest):
    hw_classification = HardwareClassification.CUDA

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_bucketed_block_flex_shard_forward_backward_on_cuda_mesh(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        reference = _TinyBlockModule(device_type.type)
        model = _TinyBlockModule(device_type.type)
        for param in [*reference.parameters(), *model.parameters()]:
            dist.broadcast(param.data, src=0)
        x = torch.arange(8, dtype=torch.float32, device=device_type).view(2, 4)
        dist.broadcast(x, src=0)

        ref_output = reference(x)
        ref_output.sum().backward()
        for param in reference.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        original_params = {
            fqn: param.detach().clone() for fqn, param in model.named_parameters()
        }

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=make_bucketed_block_placement_fn(
                        dims=(0,),
                        blocks_per_rank=(1, 3),
                    ),
                    mesh=mesh,
                    reshard_after_forward=True,
                )
            ],
        )
        output = model(x)
        output.sum().backward()

        self.assertEqual(output, ref_output.detach())
        bucket_storage = model.sharded_bucket_storages[0]
        reference_params = dict(reference.named_parameters())
        for fqn, param in model.named_parameters():
            info = bucket_storage.param_infos[fqn]
            expected_param = _expected_bucketed_local(original_params[fqn], info)
            self.assertEqual(param.detach(), expected_param)
            param_grad = param.grad
            ref_grad = reference_params[fqn].grad
            self.assertIsNotNone(param_grad)
            self.assertIsNotNone(ref_grad)
            expected_grad = _expected_bucketed_local(ref_grad.detach(), info)
            self.assertEqual(param_grad.detach(), expected_grad)


if __name__ == "__main__":
    run_tests()
