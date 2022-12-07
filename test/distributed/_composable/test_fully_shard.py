# Owner(s): ["oncall: distributed"]

import copy
import sys
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import _is_fsdp_flattened
from torch.distributed.fsdp.wrap import _FSDPPolicy, ModuleWrapPolicy
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,
    UnitModule,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestFSDPInitialization(FSDPTest):
    """Tests composable FSDP initialization."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_policy(self):
        """Tests passing a ``policy`` for pseudo-auto-wrapping."""
        self.run_subtests(
            {"policy": [None, ModuleWrapPolicy({UnitModule})]},
            self._test_policy,
        )

    def _test_policy(self, policy: Optional[_FSDPPolicy]):
        local_model = CompositeParamModel(torch.device("cuda"))
        fsdp_wrapped_model = FSDP(
            copy.deepcopy(local_model),
            auto_wrap_policy=policy,
            use_orig_params=True,
        )
        composable_module = copy.deepcopy(local_model)
        fully_shard(
            composable_module,
            policy=policy,
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
        composable_module = CompositeParamModel(device=cpu_device)
        for param in composable_module.parameters():
            assert (
                param.device == cpu_device
            ), "Expects module to be initialized on CPU for this unit test"
        fully_shard(
            composable_module,
            policy=ModuleWrapPolicy({UnitModule}),
            device_id=self.rank,
        )
        for param in composable_module.parameters():
            self.assertEqual(param.device, torch.device("cuda", self.rank))

    @skip_if_lt_x_gpu(2)
    def test_sync_module_states(self):
        """Tests passing ``sync_module_states=True``."""
        local_model = CompositeParamModel(device=torch.device("cuda"))
        composable_module = copy.deepcopy(local_model)
        # Check that the parameters are broadcast from rank 0 by comparing
        # against an equivalent FSDP-wrapped module
        if self.rank != 0:
            for param in composable_module.parameters():
                with torch.no_grad():
                    param.zero_()
        policy = ModuleWrapPolicy({UnitModule})
        fsdp_wrapped_model = FSDP(
            copy.deepcopy(local_model),
            auto_wrap_policy=policy,
            use_orig_params=True,
        )
        fully_shard(
            composable_module,
            policy=policy,
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

            TODO: This function is not satisfactory because:
            (1) This requires guarding with ``_is_fsdp_flattened()``. This
            guard is needed to avoid re-initializing parameters for nested
            cases since some initialization methods strictly require non-1D
            shape (e.g. ``kaiming_uniform_()``), while FSDP replaces the
            original parameters with their 1D shards.
            (2) This requires module-by-module traversal and manual ``setattr``
            usage as opposed to first calling ``module.to_empty()`` and then
            initializing each parameter after. The latter will override the
            initialization of already-initialized nested parameters. In other
            words, this parameter initialization function must strictly modify
            only the parameters on meta device.
            """
            torch.manual_seed(0)
            for submodule in module.modules():
                for param_name, param in submodule.named_parameters(recurse=False):
                    if not _is_fsdp_flattened(param) and param.is_meta:
                        materialized_param = nn.Parameter(
                            torch.empty_like(param, device=torch.device("cuda"))
                        )
                        nn.init.uniform_(materialized_param)
                        setattr(submodule, param_name, materialized_param)

        composable_module = CompositeParamModel(device=torch.device("meta"))
        meta_model = CompositeParamModel(device=torch.device("meta"))
        fsdp_wrapped_model = FSDP(
            meta_model,
            auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
            param_init_fn=_param_init_fn,
            use_orig_params=True,
        )
        fully_shard(
            composable_module,
            policy=ModuleWrapPolicy({UnitModule}),
            param_init_fn=_param_init_fn,
        )
        for (
            (composable_param_name, composable_param),
            (fsdp_wrapped_param_name, fsdp_wrapped_param),
        ) in zip(
            composable_module.named_parameters(),
            fsdp_wrapped_model.named_parameters(),
        ):
            self.assertEqual(composable_param_name, fsdp_wrapped_param_name)
            self.assertEqual(
                composable_param.device,
                torch.device("cuda", torch.cuda.current_device()),
            )
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
        local_model = CompositeParamModel(device=device)
        fsdp_wrapped_model = FSDP(
            copy.deepcopy(local_model),
            auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
            use_orig_params=True,
        )
        composable_module = copy.deepcopy(local_model)
        fully_shard(
            composable_module,
            policy=ModuleWrapPolicy({UnitModule}),
        )
        del local_model  # not needed anymore
        LR = 1e-2
        fsdp_wrapped_optim = torch.optim.Adam(fsdp_wrapped_model.parameters(), lr=LR)
        composable_optim = torch.optim.Adam(composable_module.parameters(), lr=LR)
        for _ in range(5):
            inp = torch.randn(2, 100, device="cuda")
            losses = []
            for model, optim in (
                (fsdp_wrapped_model, fsdp_wrapped_optim),
                (composable_module, composable_optim),
            ):
                optim.zero_grad(set_to_none=True)
                out = model(inp)
                loss = out.sum()
                losses.append(loss)
                loss.backward()
                optim.step()
            self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    run_tests()
