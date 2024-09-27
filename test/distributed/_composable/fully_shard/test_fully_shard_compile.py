# Owner(s): ["oncall: distributed"]

import copy
import sys
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import checkpoint, fully_shard
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.inductor_utils import HAS_GPU


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestCompile(FSDPTest):
    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    def test_compile(self):
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                    ShardingStrategy.HYBRID_SHARD,
                    ShardingStrategy._HYBRID_SHARD_ZERO2,
                ],
                "skip_fsdp_guards": [True, False],
                "act_checkpoint": [True, False],
            },
            self._test_compile,
        )

    def _test_compile(
        self,
        sharding_strategy: ShardingStrategy,
        skip_fsdp_guards: bool,
        act_checkpoint: bool,
    ):
        torch._dynamo.config.skip_fsdp_guards = skip_fsdp_guards
        fsdp_kwargs = {
            "policy": ModuleWrapPolicy(
                {
                    nn.TransformerEncoderLayer,
                    nn.TransformerDecoderLayer,
                }
            ),
            "strategy": sharding_strategy,
        }
        base_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
            deterministic=True,
        )
        ref_model = fully_shard(copy.deepcopy(base_model), **fsdp_kwargs)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        model = fully_shard(copy.deepcopy(base_model), **fsdp_kwargs)
        if act_checkpoint:
            for module in model.modules():
                if isinstance(
                    module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)
                ):
                    checkpoint(module)
        model = torch.compile(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        for i in range(10):
            losses = []
            inp = ref_model.get_input(torch.device("cuda"))
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad()
                loss = _model(*inp).sum()
                losses.append(loss)
                loss.backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    run_tests()
