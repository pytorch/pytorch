# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._flex_shard import flex_shard, relayout_bucket
from torch.distributed.fsdp._flex_shard.bucket_storage import (
    BucketSpec,
    MixedPrecisionPolicy,
)
from torch.distributed.fsdp._flex_shard.placements.owned import (
    BucketedOwned,
    make_bucketed_owned_full_param_placement_fn,
)
from torch.distributed.fsdp._flex_shard.placements.shard import (
    per_param_placements,
    Shard,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import HardwareClassification, run_tests
from torch.testing._internal.distributed.common_flex_shard import make_test_sgd


class TestRelayoutBucket(FSDPTest):
    hw_classification = HardwareClassification.CUDA

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_matches_owner_layout_from_the_start(self) -> None:
        # A model sharded row-wise, whose second bucket then moves to
        # whole-parameter owners, trains as one built with those owners from
        # the start: the same owners and local values bit for bit, then the
        # same outputs and sharded grads over 3 steps. In FP32, and with BF16
        # params reducing in FP32. The first bucket keeps its layout.
        mesh = init_device_mesh("cuda", (self.world_size,))
        for mp_policy in (
            MixedPrecisionPolicy(),
            MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        ):
            with self.subTest(param_dtype=mp_policy.param_dtype):
                moved = self._train(mesh, mp_policy, relayout=True)
                reference = self._train(mesh, mp_policy, relayout=False)
                for name in ("placements", "local_params"):
                    self.assertEqual(moved[name], reference[name], name)
                for step, (a, b) in enumerate(
                    zip(moved["outputs"], reference["outputs"], strict=True)
                ):
                    self.assertTrue(torch.equal(a, b), f"step {step} output")
                for fqn, grad in reference["grads"].items():
                    self.assertTrue(torch.equal(moved["grads"][fqn], grad), fqn)

    @skip_if_lt_x_gpu(2)
    def test_rejects_unsharded_bucket_and_foreign_storage(self) -> None:
        mesh = init_device_mesh("cuda", (self.world_size,))
        model, _ = self._model(mesh, MixedPrecisionPolicy(), owners_first=False)
        owned = make_bucketed_owned_full_param_placement_fn()
        other, _ = self._model(mesh, MixedPrecisionPolicy(), owners_first=False)
        with self.assertRaisesRegex(ValueError, "not one of the module's"):
            relayout_bucket(model, other.sharded_bucket_storages[1], owned)
        model(torch.randn(4, 64, device="cuda")).sum().backward()
        with self.assertRaisesRegex(RuntimeError, "before the bucket's first unshard"):
            relayout_bucket(model, model.sharded_bucket_storages[1], owned)

    def _model(self, mesh, mp_policy, *, owners_first: bool):
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(64, 64), nn.Linear(64, 16), nn.Linear(16, 16), nn.Linear(16, 8)
        ).cuda()
        full_params = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["0.*", "1.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    mp_policy=mp_policy,
                ),
                BucketSpec(
                    ["2.*", "3.*"],
                    placement_fn=(
                        make_bucketed_owned_full_param_placement_fn()
                        if owners_first
                        else per_param_placements
                    ),
                    mesh=mesh,
                    mp_policy=mp_policy,
                ),
            ],
        )
        return model, full_params

    def _train(self, mesh, mp_policy, *, relayout: bool) -> dict:
        model, full_params = self._model(mesh, mp_policy, owners_first=not relayout)
        if relayout:
            storage = model.sharded_bucket_storages[1]
            new_storage = relayout_bucket(
                model, storage, make_bucketed_owned_full_param_placement_fn()
            )
            self.assertIs(model.sharded_bucket_storages[1], new_storage)
            self.assertIsNot(new_storage, storage)
        placements = {
            fqn: type(info.placement).__name__
            for storage in model.sharded_bucket_storages
            for fqn, info in storage.param_infos.items()
        }
        self.assertEqual(
            placements,
            {
                **{
                    f"{i}.{n}": Shard.__name__
                    for i in (0, 1)
                    for n in ("weight", "bias")
                },
                **{
                    f"{i}.{n}": BucketedOwned.__name__
                    for i in (2, 3)
                    for n in ("weight", "bias")
                },
            },
        )
        local_params = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        # An owner holds its whole parameter, as initialized.
        for name, param in local_params.items():
            if name.startswith(("2.", "3.")) and param.numel():
                self.assertTrue(torch.equal(param, full_params[name]), name)
        optim = make_test_sgd(model.parameters(), lr=1e-2)
        torch.manual_seed(1 + self.rank)
        outputs, grads = [], {}
        dtype = mp_policy.param_dtype or torch.float32
        for step in range(3):
            output = model(torch.randn(4, 64, device="cuda", dtype=dtype))
            output.sum().backward()
            outputs.append(output.detach().clone())
            if step == 0:
                grads = {
                    name: param.grad.detach().clone()
                    for name, param in model.named_parameters()
                }
            optim.step()
            optim.zero_grad()
        return {
            "placements": placements,
            "local_params": {k: v.tolist() for k, v in local_params.items()},
            "outputs": outputs,
            "grads": grads,
        }


if __name__ == "__main__":
    run_tests()
