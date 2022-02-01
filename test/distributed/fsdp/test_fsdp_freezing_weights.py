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
from torch.testing._internal.commonfsdp import (
    FSDPTest,
    get_full_params,
)
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


class Model(nn.Module):
    def __init__(self, withfsdp, freeze_after_wrapfsdp):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, 10)
        if withfsdp and freeze_after_wrapfsdp:
            self.fsdp_wrap()

    def fsdp_wrap(self):
        self.trunk = FSDP(self.trunk)
        self.head = FSDP(self.head)

    def forward(self, x):
        return self.head(self.trunk(x))


class NestedTrunkModel(nn.Module):
    def __init__(self, withfsdp, freeze_after_wrapfsdp):
        super().__init__()
        self.trunk = nn.Sequential(
            self._create_block(3, 64, withfsdp, freeze_after_wrapfsdp),
            self._create_block(64, 64, withfsdp, freeze_after_wrapfsdp),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )
        if withfsdp and freeze_after_wrapfsdp:
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
        self, in_channels, out_channels, withfsdp, freeze_after_wrapfsdp
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
    def _create_model(self, withfsdp, with_nested_trunk, freeze_after_wrapfsdp):
        if with_nested_trunk:
            model = NestedTrunkModel(withfsdp, freeze_after_wrapfsdp)
        else:
            model = Model(withfsdp, freeze_after_wrapfsdp)
        return model

    def _dist_train(
        self, with_nested_trunk, freezing_method, freeze_after_wrapfsdp, withfsdp
    ):
        torch.manual_seed(0)
        batch = torch.randn(size=(2, 3, 224, 224)).cuda()

        model = self._create_model(withfsdp, with_nested_trunk, freeze_after_wrapfsdp)
        model = model.cuda()

        # freezing the trunk using requires_grad.
        if freezing_method == FreezingMethod.RequiresGrad:
            for param in model.trunk.parameters():
                param.requires_grad = False

        if withfsdp:
            if not freeze_after_wrapfsdp:
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
                if withfsdp:
                    for param in model.module.module.trunk.parameters():
                        param.grad = None
                else:
                    for param in model.module.trunk.parameters():
                        param.grad = None
            optimizer.step()

        if withfsdp:
            get_full_params(model)

        return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    @parametrize("with_nested_trunk", [True, False])
    @parametrize(
        "freezing_method", [FreezingMethod.RequiresGrad, FreezingMethod.GradToNone]
    )
    @parametrize("freeze_after_wrapfsdp", [True, False])
    def test_freezing_weights(
        self, with_nested_trunk, freezing_method, freeze_after_wrapfsdp
    ):
        # DDP
        ddp_state = self._dist_train(
            with_nested_trunk, freezing_method, freeze_after_wrapfsdp, withfsdp=False
        )

        # FSDP
        fsdp_state = self._dist_train(
            with_nested_trunk, freezing_method, freeze_after_wrapfsdp, withfsdp=True
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
