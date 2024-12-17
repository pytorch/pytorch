# Owner(s): ["oncall: distributed"]

import contextlib
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
    def __init__(
        self,
        with_fsdp,
        freeze_after_wrap_fsdp,
        disable_autograd,
        fsdp_kwargs,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        self.device = torch.cuda.current_device()
        self.head = nn.Linear(64, 10)
        if with_fsdp and freeze_after_wrap_fsdp:
            self.fsdp_wrap(fsdp_kwargs)
        self.autograd_ctx = (
            torch.no_grad if disable_autograd else contextlib.nullcontext
        )

    def fsdp_wrap(self, fsdp_kwargs):
        self.trunk = FSDP(self.trunk, **fsdp_kwargs)
        self.head = FSDP(self.head, **fsdp_kwargs)

    def forward(self, x):
        with self.autograd_ctx():
            x = self.trunk(x)
        return self.head(x)


class NestedTrunkModel(nn.Module):
    def __init__(
        self,
        with_fsdp,
        freeze_after_wrap_fsdp,
        disable_autograd,
        fsdp_kwargs,
    ):
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
            self.fsdp_wrap(fsdp_kwargs)
        self.autograd_ctx = (
            torch.no_grad if disable_autograd else contextlib.nullcontext
        )

    def fsdp_wrap(self, fsdp_kwargs):
        for name, child in self.trunk.named_children():
            wrapped_child = FSDP(child, **fsdp_kwargs)
            setattr(self.trunk, name, wrapped_child)
        self.trunk = FSDP(self.trunk, **fsdp_kwargs)
        self.head = FSDP(self.head, **fsdp_kwargs)

    def forward(self, x):
        with self.autograd_ctx():
            x = self.trunk(x)
        return self.head(x)

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
    def _create_model(
        self,
        with_fsdp,
        with_nested_trunk,
        freeze_after_wrap_fsdp,
        disable_autograd,
        fsdp_kwargs,
    ):
        if with_nested_trunk:
            model = NestedTrunkModel(
                with_fsdp, freeze_after_wrap_fsdp, disable_autograd, fsdp_kwargs
            )
        else:
            model = Model(
                with_fsdp, freeze_after_wrap_fsdp, disable_autograd, fsdp_kwargs
            )
        return model

    def _dist_train(
        self,
        with_nested_trunk,
        freezing_method,
        freeze_after_wrap_fsdp,
        with_fsdp,
        disable_autograd,
        forward_prefetch,
    ):
        torch.manual_seed(0)
        batch = torch.randn(size=(2, 3, 224, 224)).cuda()

        fsdp_kwargs = {
            "device_id": self.rank,
            "forward_prefetch": forward_prefetch,
        }

        ddp_kwargs = {
            "device_ids": [self.rank],
            "find_unused_parameters": True if disable_autograd else False,
        }

        model = self._create_model(
            with_fsdp,
            with_nested_trunk,
            freeze_after_wrap_fsdp,
            disable_autograd,
            fsdp_kwargs,
        )
        model = model.cuda()

        # freezing the trunk using requires_grad.
        if freezing_method == FreezingMethod.RequiresGrad:
            for param in model.trunk.parameters():
                param.requires_grad = False

        if with_fsdp:
            if not freeze_after_wrap_fsdp:
                model.fsdp_wrap(fsdp_kwargs)
            model = FSDP(model, **fsdp_kwargs)
        else:
            model = DistributedDataParallel(model, **ddp_kwargs)

        target = torch.tensor([0, 1], dtype=torch.long).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        for _ in range(3):
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
    @parametrize("disable_autograd", [True, False])
    @parametrize("forward_prefetch", [True, False])
    def test_freezing_weights(
        self,
        with_nested_trunk,
        freezing_method,
        freeze_after_wrap_fsdp,
        disable_autograd,
        forward_prefetch,
    ):
        # DDP
        ddp_state = self._dist_train(
            with_nested_trunk,
            freezing_method,
            freeze_after_wrap_fsdp,
            with_fsdp=False,
            disable_autograd=disable_autograd,
            forward_prefetch=False,  # does not apply to DDP
        )

        # FSDP
        fsdp_state = self._dist_train(
            with_nested_trunk,
            freezing_method,
            freeze_after_wrap_fsdp,
            with_fsdp=True,
            disable_autograd=disable_autograd,
            forward_prefetch=forward_prefetch,
        )

        self.assertEqual(
            ddp_state,
            fsdp_state,
            exact_device=True,
            msg="FullyShardedDataParallel states didn't match PyTorch DDP states",
        )

        if freezing_method == FreezingMethod.RequiresGrad:
            for ddp_param, fsdp_param in zip(ddp_state, fsdp_state):
                self.assertEqual(ddp_param.requires_grad, fsdp_param.requires_grad)


instantiate_parametrized_tests(TestFreezingWeights)

if __name__ == "__main__":
    run_tests()
