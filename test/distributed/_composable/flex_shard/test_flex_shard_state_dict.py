# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.nn as nn
from torch.distributed.fsdp._flex_shard import is_flex_shard_param
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed.common_flex_shard import (
    flex_shard_cuda,
    make_test_sgd,
    make_transformer_model,
    single_rank_cuda_mesh,
    transformer_bucket_specs,
    transformer_inputs,
)


def _make_flex_sharded_transformer(mesh, **kwargs):
    args, model = make_transformer_model(**kwargs)
    flex_shard_cuda(
        model,
        mesh,
        buckets=transformer_bucket_specs(
            args.n_layers,
            mesh,
            reshard_after_forward=False,
        ),
    )
    return args, model


class TestFlexShardStateDict(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_state_dict_load_round_trip_into_flex_sharded_model(self):
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            args, source = _make_flex_sharded_transformer(mesh)
            state_dict = {k: v.clone() for k, v in source.state_dict().items()}

            torch.manual_seed(1)
            _, target = _make_flex_sharded_transformer(mesh)
            load_result = target.load_state_dict(state_dict)

            self.assertEqual(load_result.missing_keys, [])
            self.assertEqual(load_result.unexpected_keys, [])

            x = transformer_inputs(args, batch_size=3, device="cuda")
            self.assertEqual(source(x), target(x))
            # Those forwards left the buckets unsharded: state_dict() reshards
            # first, and so does reshard().
            sharded = source.state_dict(keep_vars=True).values()
            self.assertTrue(all(is_flex_shard_param(value) for value in sharded))
            target.reshard()
            self.assertTrue(all(is_flex_shard_param(p) for p in target.parameters()))

            source_optim = make_test_sgd(source.parameters(), lr=0.05)
            target_optim = make_test_sgd(target.parameters(), lr=0.05)
            for model, optim in ((source, source_optim), (target, target_optim)):
                optim.zero_grad(set_to_none=True)
                model(x).sum().backward()
                optim.step()

            for key, value in source.state_dict().items():
                self.assertEqual(value, target.state_dict()[key])

    def test_load_state_dict_assign_after_forward(self):
        # load_state_dict(assign=True) replaces the sharded parameters, also
        # once the buckets have unsharded; the next unshard gathers the
        # loaded values
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            args, source = _make_flex_sharded_transformer(mesh)
            torch.manual_seed(1)
            _, target = _make_flex_sharded_transformer(mesh)
            x = transformer_inputs(args, batch_size=3, device="cuda")
            target(x).sum().backward()

            state_dict = {k: v.detach().clone() for k, v in source.state_dict().items()}
            target.load_state_dict(state_dict, assign=True)
            self.assertTrue(all(is_flex_shard_param(p) for p in target.parameters()))
            self.assertEqual(source(x), target(x))
            target.zero_grad(set_to_none=True)
            target(x).sum().backward()
            target.reshard()
            self.assertTrue(all(p.grad is not None for p in target.parameters()))

    def test_module_to_dtype_and_device_after_forward(self):
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            reference = nn.Sequential(nn.Linear(8, 12), nn.ReLU(), nn.Linear(12, 4))
            reference.cuda()
            model = copy.deepcopy(reference)
            flex_shard_cuda(model, mesh)
            x = torch.randn(2, 8, device="cuda")
            model(x).sum().backward()
            model.zero_grad(set_to_none=True)

            model.to(torch.float64)
            reference.to(torch.float64)
            self.assertEqual(model(x.double()), reference(x.double()))
            model.cpu()
            model.cuda()
            self.assertEqual(model(x.double()), reference(x.double()))
            model.reshard()
            for param in model.parameters():
                self.assertTrue(is_flex_shard_param(param))
                self.assertEqual(param.dtype, torch.float64)

    def test_params_left_on_meta_must_be_replaced_before_forward(self):
        def make() -> nn.Module:
            return nn.Sequential(
                nn.Linear(8, 8, device="cuda"), nn.Linear(8, 8, device="meta")
            )

        with single_rank_cuda_mesh() as mesh:
            x = torch.randn(2, 8, device="cuda")
            model = flex_shard_cuda(make(), mesh)
            with self.assertRaisesRegex(
                RuntimeError, r"still on meta device: \['1.weight', '1.bias'\]"
            ):
                model(x)

            torch.manual_seed(0)
            reference = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)).cuda()
            module = make()
            module[0].load_state_dict(reference[0].state_dict())
            model = flex_shard_cuda(module, mesh)
            model[1].load_state_dict(reference[1].state_dict(), assign=True)
            self.assertEqual(model(x), reference(x))

    def test_load_state_dict_rejects_incompatible_shapes(self):
        with single_rank_cuda_mesh() as mesh:
            _, source = _make_flex_sharded_transformer(mesh, dim=8, n_heads=2)
            _, target = _make_flex_sharded_transformer(mesh, dim=12, n_heads=3)

            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                target.load_state_dict(source.state_dict())


if __name__ == "__main__":
    run_tests()
