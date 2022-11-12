# Owner(s): ["oncall: distributed"]

import copy
import sys
from typing import Any, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import _is_fsdp_flattened
from torch.distributed.fsdp._runtime_utils import _root_pre_forward
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
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


class SubModel(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(5, 5, bias=False, device=device)
        self.lin2 = nn.Linear(5, 5, bias=False, device=device)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.relu(self.lin1(x))
        z = self.relu(self.lin2(z))
        return z


class Model(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.sub1 = SubModel(device=device)
        self.sub2 = SubModel(device=device)
        self.lin = nn.Linear(5, 5, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.sub1(x)
        z = self.sub2(z)
        z = self.lin(z)
        return z

    @staticmethod
    def policy():
        return ModuleWrapPolicy({SubModel})

    def get_input(self, device=torch.device) -> Tuple[Any, ...]:
        return (torch.randn((8, 5), device=device),)


class TestFSDPInitialization(FSDPTest):
    """Tests composable FSDP initialization."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_auto_wrap_policy(self):
        """Tests passing an ``auto_wrap_policy``."""

        local_model = Model(device=torch.device("cuda"))
        fsdp_wrapped_model = FSDP(
            copy.deepcopy(local_model),
            auto_wrap_policy=Model.policy(),
            use_orig_params=True,
        )
        composable_module = copy.deepcopy(local_model)
        fully_shard(
            composable_module,
            policy=Model.policy(),
        )

        # Check that the composable module has the same names as the local
        # model and the same sharded parameters as the FSDP-wrapped model
        for (
            (local_name, _),
            (composable_name, composable_param),
            (_, fsdp_wrapped_param),
        ) in zip(
            local_model.named_parameters(),
            composable_module.named_parameters(),
            fsdp_wrapped_model.named_parameters(),
        ):
            self.assertEqual(local_name, composable_name)
            self.assertEqual(fsdp_wrapped_param, composable_param)

        # Check that the composable module has the same  `FlatParameter`
        # construction as the FSDP-wrapped model
        composable_handles = fully_shard.state(composable_module)._handles
        fsdp_wrapped_handles = FSDP._fsdp_handles(fsdp_wrapped_model)
        self.assertEqual(len(composable_handles), len(fsdp_wrapped_handles))
        for (composable_handle, fsdp_wrapped_handle) in zip(
            composable_handles, fsdp_wrapped_handles
        ):
            self.assertEqual(
                composable_handle.flat_param.shape, fsdp_wrapped_handle.flat_param.shape
            )

        # Check that the composable module does not add any wrapper class
        local_module_classes = set()
        composable_module_classes = set()
        for submodule in local_model.modules():
            local_module_classes.add(type(submodule))
        for submodule in composable_module.modules():
            composable_module_classes.add(type(submodule))
        self.assertEqual(local_module_classes, composable_module_classes)

    @skip_if_lt_x_gpu(2)
    def test_device_id(self):
        """Tests passing a ``device_id``."""
        cpu_device = torch.device("cpu")
        composable_module = Model(device=cpu_device)
        for param in composable_module.parameters():
            assert param.device == cpu_device
        fully_shard(
            composable_module,
            policy=Model.policy(),
            device_id=self.rank,
        )
        for param in composable_module.parameters():
            self.assertEqual(param.device, torch.device("cuda", self.rank))

    @skip_if_lt_x_gpu(2)
    def test_sync_module_states(self):
        """Tests passing ``sync_module_states=True``."""
        local_model = Model(device=torch.device("cuda"))
        composable_module = copy.deepcopy(local_model)
        # Check that the parameters are broadcast from rank 0 by comparing
        # against an equivalent FSDP-wrapped module
        if self.rank != 0:
            for param in composable_module.parameters():
                with torch.no_grad():
                    param.zero_()
        fsdp_wrapped_model = FSDP(
            copy.deepcopy(local_model),
            auto_wrap_policy=Model.policy(),
            use_orig_params=True,
        )
        fully_shard(
            composable_module,
            policy=Model.policy(),
            sync_module_states=True,
        )
        for (composable_param, fsdp_wrapped_param) in zip(
            composable_module.parameters(),
            fsdp_wrapped_model.parameters(),
        ):
            self.assertEqual(composable_param, fsdp_wrapped_param)

    @skip_if_lt_x_gpu(2)
    def test_materialize_meta_module(self):
        """Tests materializing a meta-device module."""

        def _param_init_fn(module: nn.Module):
            """
            This is an example ``param_init_fn`` for composable FSDP.

            TODO: This function is not satisfactory because this requires
            guarding with ``_is_fsdp_flattened()``. This guard is needed to
            avoid re-initializing parameters for nested cases since some
            initialization methods strictly require non-1D shape (e.g.
            ``kaiming_uniform_()``), while FSDP replaces the original
            parameters with their 1D shards.
            """
            is_meta = any(param.is_meta for param in module.parameters())
            if is_meta:
                module.to_empty(device=torch.cuda.current_device())
            torch.manual_seed(0)
            for param in module.parameters():
                if not _is_fsdp_flattened(param):
                    nn.init.uniform_(param)

        composable_module = Model(device="meta")
        fsdp_wrapped_model = FSDP(
            Model(device="meta"),
            auto_wrap_policy=Model.policy(),
            param_init_fn=_param_init_fn,
            use_orig_params=True,
        )
        fully_shard(
            composable_module,
            policy=Model.policy(),
            param_init_fn=_param_init_fn,
        )
        for (composable_param, fsdp_wrapped_param) in zip(
            composable_module.parameters(),
            fsdp_wrapped_model.parameters(),
        ):
            self.assertEqual(composable_param, fsdp_wrapped_param)


class TestFSDPRuntime(FSDPTest):
    """Tests composable FSDP runtime."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_training(self):
        """Tests training (forward, backward, optimizer)."""
        device = torch.device("cuda")
        local_model = Model(device=device)
        fsdp_wrapped_model = FSDP(
            copy.deepcopy(local_model),
            auto_wrap_policy=Model.policy(),
            use_orig_params=True,
        )
        composable_module = copy.deepcopy(local_model)
        fully_shard(
            composable_module,
            policy=Model.policy(),
        )
        del local_model  # not needed anymore
        LR = 1e-2
        fsdp_wrapped_optim = torch.optim.Adam(fsdp_wrapped_model.parameters(), lr=LR)
        composable_optim = torch.optim.Adam(composable_module.parameters(), lr=LR)
        for _ in range(5):
            inp = composable_module.get_input(device)
            losses = []
            for model, optim in (
                (fsdp_wrapped_model, fsdp_wrapped_optim),
                (composable_module, composable_optim),
            ):
                optim.zero_grad(set_to_none=True)
                # TODO (awgu): Remove this after resolving the root pre-forward
                # hook registration, currently blocked by kwarg support
                if model is composable_module:
                    args, kwargs = _root_pre_forward(
                        fully_shard.state(composable_module), composable_module, *inp
                    )
                else:
                    args = inp
                    kwargs = {}
                out = model(*args, **kwargs)
                loss = out.sum()
                losses.append(loss)
                loss.backward()
                optim.step()
            self.assertEqual(losses[0], losses[1])


instantiate_parametrized_tests(TestFSDPInitialization)

if __name__ == "__main__":
    run_tests()
