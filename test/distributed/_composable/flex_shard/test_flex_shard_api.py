# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.fsdp._flex_shard import (
    BucketSpec,
    flex_shard,
    get_global_shape,
    get_placements,
    GradientReduction,
    is_flex_shard_param,
)
from torch.distributed.fsdp._flex_shard.placements.shard import per_param_placements
from torch.distributed.tensor import Shard
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed.common_flex_shard import (
    flex_shard_cuda,
    flex_shard_transformer_model,
    make_transformer_model,
    single_rank_cpu_mesh,
    single_rank_cuda_mesh,
)


class TestFlexShardAPI(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_reapplying_flex_shard_to_same_module_raises(self):
        with single_rank_cuda_mesh() as mesh:
            _, model = flex_shard_transformer_model(mesh)

            with self.assertRaisesRegex(ValueError, "Cannot apply flex_shard twice"):
                flex_shard_cuda(
                    model,
                    mesh,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                        )
                    ],
                )

    def test_flex_shard_rejects_child_then_root_nested_wrapping(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

            flex_shard(
                model[0],
                buckets=[
                    BucketSpec(
                        ["*"],
                        placement_fn=per_param_placements,
                        mesh=mesh,
                        reshard_after_forward=False,
                    )
                ],
            )

            with self.assertRaisesRegex(ValueError, "Nested flex_shard wrapping"):
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                        )
                    ],
                )

    def test_flex_shard_rejects_root_then_child_nested_wrapping(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

            flex_shard(
                model,
                buckets=[
                    BucketSpec(
                        ["*"],
                        placement_fn=per_param_placements,
                        mesh=mesh,
                        reshard_after_forward=False,
                    )
                ],
            )

            with self.assertRaisesRegex(ValueError, "Nested flex_shard wrapping"):
                flex_shard(
                    model[0],
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                        )
                    ],
                )

    def test_metadata_helpers_on_managed_and_unmanaged_tensors(self):
        with single_rank_cuda_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)
            raw_param = model.tok_embeddings._parameters["weight"]

            self.assertTrue(is_flex_shard_param(raw_param))
            self.assertEqual(get_placements(raw_param), (Shard(0),))
            self.assertEqual(
                get_global_shape(raw_param),
                torch.Size([args.vocab_size, args.dim]),
            )

            unmanaged = torch.randn(2, 3)
            self.assertFalse(is_flex_shard_param(unmanaged))
            self.assertIsNone(get_placements(unmanaged))
            self.assertIsNone(get_global_shape(unmanaged))

    def test_gradient_reduction_validates_its_factor(self):
        for factor in (0, -2.0, float("inf")):
            with self.assertRaisesRegex(ValueError, "positive number"):
                GradientReduction(divide_factor=factor)
        with self.assertRaisesRegex(ValueError, "requires gradient_reduce_op=AVG"):
            GradientReduction(dist.ReduceOp.SUM, 2)

    def test_offload_policy_is_rejected_until_supported(self):
        with single_rank_cuda_mesh() as mesh:
            _, model = make_transformer_model()
            with self.assertRaisesRegex(NotImplementedError, "CPUOffloadPolicy"):
                flex_shard_cuda(
                    model,
                    mesh,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            offload_policy=CPUOffloadPolicy(pin_memory=False),
                            reshard_after_forward=False,
                        )
                    ],
                )

    def test_param_dtype_override_fn(self):
        with single_rank_cuda_mesh() as mesh:

            def shard(override_dtype):
                model = nn.Sequential(
                    nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)
                )
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    param_dtype_override_fn=lambda param: (
                        override_dtype if param is model[1].weight else None
                    ),
                )
                bucket = BucketSpec(
                    ["*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=False,
                )
                return flex_shard_cuda(model, mesh, buckets=[bucket])

            model = shard(torch.float32)
            unsharded_dtypes = {
                fqn: info.unsharded_dtype
                for storage in model.sharded_bucket_storages
                for fqn, info in storage.param_infos.items()
            }
            self.assertEqual(
                unsharded_dtypes,
                {"0.weight": torch.bfloat16, "1.weight": torch.float32},
            )
            with self.assertRaisesRegex(ValueError, "param_dtype_override_fn"):
                shard(torch.float16)

    def test_mixed_precision_casts_at_hooked_modules(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, output_dtype=torch.float32
            )
            buckets = [
                BucketSpec([name], per_param_placements, mesh, mp_policy=mp_policy)
                for name in ("0", "1")
            ]
            flex_shard_cuda(model, mesh, buckets=buckets)
            input_dtypes = []
            for linear in model:
                linear.register_forward_pre_hook(
                    lambda module, args: input_dtypes.append(args[0].dtype)
                )
            output = model(torch.randn(2, 4, device="cuda"))
            self.assertEqual(input_dtypes, [torch.bfloat16] * 2)
            self.assertEqual(output.dtype, torch.float32)

            # Buckets that hook one module must cast alike
            no_cast = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, cast_forward_inputs=False
            )
            buckets = [
                BucketSpec(["weight"], per_param_placements, mesh, mp_policy=mp_policy),
                BucketSpec(["bias"], per_param_placements, mesh, mp_policy=no_cast),
            ]
            with self.assertRaisesRegex(ValueError, "must cast its forward inputs"):
                flex_shard_cuda(nn.Linear(4, 4), mesh, buckets=buckets)

    def test_cpu_mesh_is_rejected(self):
        with single_rank_cpu_mesh() as mesh:
            _, model = make_transformer_model()

            with self.assertRaisesRegex(NotImplementedError, "CUDA DeviceMesh"):
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                        )
                    ],
                )


if __name__ == "__main__":
    run_tests()
