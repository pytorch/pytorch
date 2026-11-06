# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.fsdp._flex_shard import is_flex_shard_param
from torch.testing._internal.common_utils import run_tests, TestCase
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

    def test_load_state_dict_rejects_incompatible_shapes(self):
        with single_rank_cuda_mesh() as mesh:
            _, source = _make_flex_sharded_transformer(mesh, dim=8, n_heads=2)
            _, target = _make_flex_sharded_transformer(mesh, dim=12, n_heads=3)

            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                target.load_state_dict(source.state_dict())


if __name__ == "__main__":
    run_tests()
