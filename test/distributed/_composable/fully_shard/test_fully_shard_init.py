# Owner(s): ["oncall: distributed"]

import copy
import sys
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import _is_fsdp_flattened, clean_tensor_name
from torch.distributed.fsdp.wrap import _FSDPPolicy, LambdaWrapPolicy, ModuleWrapPolicy
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,
    FakeSequential,
    NestedSequentialModel,
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


class TestInitialization(FSDPTest):
    """Tests ``fully_shard`` initialization."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_policy(self):
        """Tests passing a ``policy`` for pseudo-auto-wrapping."""

        def lambda_fn(module: nn.Module):
            if isinstance(module, nn.Sequential):
                return True
            elif isinstance(module, FakeSequential):
                return {"backward_prefetch": BackwardPrefetch.BACKWARD_POST}
            return False

        self.run_subtests(
            {
                "policy": [
                    None,
                    ModuleWrapPolicy({UnitModule}),
                    ModuleWrapPolicy({nn.Sequential}),
                    LambdaWrapPolicy(lambda_fn),
                ],
            },
            self._test_policy,
        )

    def _test_policy(self, policy: Optional[_FSDPPolicy]):
        use_nested_sequential_model = "Sequential" in getattr(
            policy, "_module_classes_str", ""
        )
        local_model = (
            NestedSequentialModel(torch.device("cuda"))
            if use_nested_sequential_model
            else CompositeParamModel(torch.device("cuda"))
        )
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
        self._test_fully_shard_construction(
            local_model,
            fsdp_wrapped_model,
            composable_module,
        )

    @skip_if_lt_x_gpu(2)
    def test_manual_fully_shard(self):
        """Tests manually applying ``fully_shard``."""
        local_model = CompositeParamModel(torch.device("cuda"))
        fsdp_wrapped_model = copy.deepcopy(local_model)
        fsdp_wrapped_model.u2 = FSDP(fsdp_wrapped_model.u2, use_orig_params=True)
        fsdp_wrapped_model = FSDP(fsdp_wrapped_model, use_orig_params=True)
        composable_module = copy.deepcopy(local_model)
        fully_shard(composable_module.u2)
        fully_shard(composable_module)
        self._test_fully_shard_construction(
            local_model,
            fsdp_wrapped_model,
            composable_module,
        )

    def _test_fully_shard_construction(
        self,
        local_model: nn.Module,
        fsdp_wrapped_model: FSDP,
        composable_module: nn.Module,
    ):
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
        composable_handles = traversal_utils._get_fsdp_handles(composable_module)
        fsdp_wrapped_handles = traversal_utils._get_fsdp_handles(fsdp_wrapped_model)
        self.assertEqual(len(composable_handles), len(fsdp_wrapped_handles))
        for composable_handle, fsdp_wrapped_handle in zip(
            composable_handles, fsdp_wrapped_handles
        ):
            self.assertEqual(
                composable_handle.flat_param.shape, fsdp_wrapped_handle.flat_param.shape
            )
            self.assertEqual(
                composable_handle.flat_param._fqns,
                fsdp_wrapped_handle.flat_param._fqns,
            )

        # Check that the composable module does not add any wrapper class
        local_module_classes = set()
        composable_module_classes = set()
        for submodule in local_model.modules():
            local_module_classes.add(type(submodule))
        for submodule in composable_module.modules():
            composable_module_classes.add(type(submodule))
        self.assertEqual(local_module_classes, composable_module_classes)

        # Check that the composable module has the same FSDP states with the
        # same attributes (mainly checking backward prefetch since the lambda
        # wrap policy overrides it for `FakeSequential`)
        wrapper_states = traversal_utils._get_fsdp_states(fsdp_wrapped_model)
        composable_states = traversal_utils._get_fsdp_states(composable_module)
        self.assertEqual(len(wrapper_states), len(composable_states))
        for wrapper_state, composable_state in zip(wrapper_states, composable_states):
            self.assertEqual(
                wrapper_state.sharding_strategy, composable_state.sharding_strategy
            )
            self.assertEqual(
                wrapper_state.backward_prefetch, composable_state.backward_prefetch
            )

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
        for composable_param, fsdp_wrapped_param in zip(
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
            self.assertEqual(
                composable_param_name, clean_tensor_name(fsdp_wrapped_param_name)
            )
            self.assertEqual(
                composable_param.device,
                torch.device("cuda", torch.cuda.current_device()),
            )
            self.assertEqual(composable_param, fsdp_wrapped_param)

    @skip_if_lt_x_gpu(2)
    def test_nested_fully_shard_shared_state(self):
        """
        Tests that nested applications of ``fully_shard`` share the expected
        data structure state.
        """
        self.run_subtests(
            {"use_policy": [False, True]},
            self._test_nested_fully_shard_shared_state,
        )

    def _test_nested_fully_shard_shared_state(self, use_policy: bool):
        device = torch.device("cuda")
        composable_module = CompositeParamModel(device=device)
        if use_policy:
            fully_shard(composable_module, policy=ModuleWrapPolicy({UnitModule}))
        else:
            fully_shard(composable_module.u1)
            fully_shard(composable_module.u2)
            fully_shard(composable_module)

        # Run a forward pass to trigger lazy initialization
        inp = torch.randn((2, 100), device=device)
        composable_module(inp)

        # Check that all modules with `fully_shard` applied share the same data
        # structure state for the structures with the given names (there is no
        # need to check all of them to verify that the sharing worked).
        # NOTE: This check only requires that the data structure state is
        # shared. Namely, sharing the FSDP state object itself is sufficient
        # but not necessary.
        data_structure_names = [
            "_exec_order_data",
            "_free_event_queue",
            "_pre_unshard_stream",
            "_unshard_stream",
            "_post_backward_stream",
            "_default_stream",
        ]
        for data_structure_name in data_structure_names:
            all_structures = set()
            for module in (
                composable_module.u1,
                composable_module.u2,
                composable_module,
            ):
                all_structures.add(
                    id(getattr(fully_shard.state(module), data_structure_name))
                )
            self.assertEqual(len(all_structures), 1)


if __name__ == "__main__":
    run_tests()
