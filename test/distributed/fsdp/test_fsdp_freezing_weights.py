# Owner(s): ["oncall: distributed"]

import sys
from enum import Enum

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_full_params
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
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


class Model(nn.Module):
    def __init__(self, with_fsdp, freeze_after_wrap_fsdp):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, 10)
        if with_fsdp and freeze_after_wrap_fsdp:
            self.fsdp_wrap()

    def fsdp_wrap(self):
        self.trunk = FSDP(self.trunk)
        self.head = FSDP(self.head)

    def forward(self, x):
        return self.head(self.trunk(x))


class NestedTrunkModel(nn.Module):
    def __init__(self, with_fsdp, freeze_after_wrap_fsdp):
        super().__init__()
        self.trunk = nn.Sequential(
            self._create_block(3, 64, with_fsdp, freeze_after_wrap_fsdp),
            self._create_block(64, 64, with_fsdp, freeze_after_wrap_fsdp),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )
        if with_fsdp and freeze_after_wrap_fsdp:
            self.fsdp_wrap()

    def fsdp_wrap(self):
        for name, child in self.trunk.named_children():
            wrapped_child = FSDP(child)
            setattr(self.trunk, name, wrapped_child)
        self.trunk = FSDP(self.trunk)
        self.head = FSDP(self.head)

    def forward(self, x):
        return self.head(self.trunk(x))

    def _create_block(
        self, in_channels, out_channels, with_fsdp, freeze_after_wrap_fsdp
    ):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        return block


class FreezingMethod(str, Enum):
    GradToNone = "grad_to_none"
    RequiresGrad = "requires_grad"


class TestFreezingWeights(FSDPTest):
    def _create_model(self, with_fsdp, with_nested_trunk, freeze_after_wrap_fsdp):
        if with_nested_trunk:
            model = NestedTrunkModel(with_fsdp, freeze_after_wrap_fsdp)
        else:
            model = Model(with_fsdp, freeze_after_wrap_fsdp)
        return model

    def _dist_train(
        self, with_nested_trunk, freezing_method, freeze_after_wrap_fsdp, with_fsdp
    ):
        torch.manual_seed(0)
        batch = torch.randn(size=(2, 3, 224, 224)).cuda()

        model = self._create_model(with_fsdp, with_nested_trunk, freeze_after_wrap_fsdp)
        model = model.cuda()

        # freezing the trunk using requires_grad.
        if freezing_method == FreezingMethod.RequiresGrad:
            for param in model.trunk.parameters():
                param.requires_grad = False

        if with_fsdp:
            if not freeze_after_wrap_fsdp:
                model.fsdp_wrap()
            model = FSDP(model)
        else:
            model = DistributedDataParallel(model, device_ids=[self.rank])

        target = torch.tensor([0, 1], dtype=torch.long).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        for iteration in range(3):
            out = model(batch)
            fake_loss = criterion(out, target)
            optimizer.zero_grad()
            fake_loss.backward()
            if freezing_method == FreezingMethod.GradToNone:
                for param in model.module.trunk.parameters():
                    param.grad = None
            optimizer.step()

        if with_fsdp:
            return get_full_params(model)

        return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    @parametrize("with_nested_trunk", [True, False])
    @parametrize(
        "freezing_method", [FreezingMethod.RequiresGrad, FreezingMethod.GradToNone]
    )
    @parametrize("freeze_after_wrap_fsdp", [True, False])
    def test_freezing_weights(
        self, with_nested_trunk, freezing_method, freeze_after_wrap_fsdp
    ):
        # DDP
        ddp_state = self._dist_train(
            with_nested_trunk, freezing_method, freeze_after_wrap_fsdp, with_fsdp=False
        )

        # FSDP
        fsdp_state = self._dist_train(
            with_nested_trunk, freezing_method, freeze_after_wrap_fsdp, with_fsdp=True
        )

        self.assertEqual(
            ddp_state,
            fsdp_state,
            exact_device=True,
            msg="FullyShardedDataParallel states didn't match PyTorch DDP states",
        )


instantiate_parametrized_tests(TestFreezingWeights)

if __name__ == "__main__":
    run_tests()
