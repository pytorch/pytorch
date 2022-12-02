import copy
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import checkpoint, fully_shard
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 100)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )
        self.l2 = nn.Linear(100, 100)

    def forward(self, x):
        return self.l2(self.seq(self.l1(x)))


class TestFSDPCheckpoint(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_simple(self, use_reentrant):
        LR = 0.01
        device = torch.device("cuda")

        model = ToyModel().to(device)

        local_model = copy.deepcopy(model)
        local_optim = torch.optim.Adam(local_model.parameters(), lr=LR)

        combo_model = copy.deepcopy(model)
        combo_optim = torch.optim.Adam(combo_model.parameters(), lr=LR)
        combo_model.seq = checkpoint(
            combo_model.seq, use_reentrant=use_reentrant
        )
        combo_model.seq = fully_shard(
            combo_model.seq,
            policy=ModuleWrapPolicy({nn.Linear}),
        )

        x = torch.randn(2, 100, device=device)

        for _ in range(5):
            combo_loss = combo_model(x).sum()
            local_loss = local_model(x).sum()

            self.assertEqual(combo_loss, local_loss)

            combo_loss.backward()
            combo_optim.step()
            combo_optim.zero_grad()

            local_loss.backward()
            local_optim.step()
            local_optim.zero_grad()


instantiate_parametrized_tests(TestFSDPCheckpoint)


if __name__ == "__main__":
    run_tests()
