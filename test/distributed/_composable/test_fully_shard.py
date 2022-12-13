# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import sys
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import (
    _all_handles,
    _FSDPState,
    _is_fsdp_flattened,
)
from torch.distributed.fsdp.api import MixedPrecision
from torch.distributed.fsdp.flat_param import _HandlesKey, FlatParamHandle
from torch.distributed.fsdp.wrap import _FSDPPolicy, ModuleWrapPolicy
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,
    UnitModule,
)
from torch.testing._internal.common_distributed import (
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
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

    def _init_models_and_optims(
        self, device: torch.device
    ) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module, torch.optim.Optimizer]:
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
        LR = 1e-2
        fsdp_wrapped_optim = torch.optim.Adam(fsdp_wrapped_model.parameters(), lr=LR)
        composable_optim = torch.optim.Adam(composable_module.parameters(), lr=LR)
        return (
            composable_module,
            composable_optim,
            fsdp_wrapped_model,
            fsdp_wrapped_optim,
        )

    @skip_if_lt_x_gpu(2)
    def test_training(self):
        """Tests training (forward, backward, optimizer)."""
        device = torch.device("cuda")
        (
            composable_module,
            composable_optim,
            fsdp_wrapped_model,
            fsdp_wrapped_optim,
        ) = self._init_models_and_optims(device)
        for _ in range(5):
            inp = torch.randn(2, 100, device="cuda")
            losses: List[torch.Tensor] = []
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

    @skip_if_lt_x_gpu(2)
    def test_unshard_reshard_order(self):
        """
        Tests that the unshard/reshard order matches between ``fully_shard``
        and ``FullyShardedDataParallel`` for the same policy.

        NOTE: We use FQNs as the proxy for checking the order across the two
        versions. See ``_check_same_param_handles()`` for details.
        """
        device = torch.device("cuda")
        (
            composable_module,
            composable_optim,
            fsdp_wrapped_model,
            fsdp_wrapped_optim,
        ) = self._init_models_and_optims(device)
        # Before checking the unshard/reshard order, sanity check that the
        # assumption about wrapper FQN being a suffix of composable FQN holds
        all_composable_handles = _all_handles(fully_shard.state(composable_module))
        all_wrapped_handles = _all_handles(fsdp_wrapped_model)
        self._check_same_param_handles(all_composable_handles, all_wrapped_handles)
        num_handles = len(all_composable_handles)

        orig_unshard = torch.distributed.fsdp._runtime_utils._unshard
        orig_reshard = torch.distributed.fsdp._runtime_utils._reshard
        UnshardReshardEvent = Tuple[str, _HandlesKey]

        def patched_unshard(
            unshard_reshard_order: List[UnshardReshardEvent],
            state: _FSDPState,
            handles: List[FlatParamHandle],
            *args,
            **kwargs,
        ):
            handles_key = tuple(handles)
            unshard_reshard_order.append(("unshard", handles_key))
            return orig_unshard(state, handles, *args, **kwargs)

        def patched_reshard(
            unshard_reshard_order: List[UnshardReshardEvent],
            state: _FSDPState,
            handles: List[FlatParamHandle],
            *args,
            **kwargs,
        ):
            handles_key = tuple(handles)
            unshard_reshard_order.append(("reshard", handles_key))
            return orig_reshard(state, handles, *args, **kwargs)

        @contextlib.contextmanager
        def patch_unshard(_patched_unshard: Callable):
            _orig_unshard = torch.distributed.fsdp._runtime_utils._unshard
            torch.distributed.fsdp._runtime_utils._unshard = _patched_unshard
            try:
                yield
            finally:
                torch.distributed.fsdp._runtime_utils._unshard = _orig_unshard

        @contextlib.contextmanager
        def patch_reshard(_patched_reshard: Callable):
            _orig_reshard = torch.distributed.fsdp._runtime_utils._reshard
            torch.distributed.fsdp._runtime_utils._reshard = _patched_reshard
            try:
                yield
            finally:
                torch.distributed.fsdp._runtime_utils._unshard = _orig_reshard

        composable_order: List[UnshardReshardEvent] = []
        wrapped_order: List[UnshardReshardEvent] = []

        inp = torch.randn(2, 100, device="cuda")
        losses: List[torch.Tensor] = []

        for order, model, optim in (
            (composable_order, composable_module, composable_optim),
            (wrapped_order, fsdp_wrapped_model, fsdp_wrapped_optim),
        ):
            with patch_unshard(
                functools.partial(patched_unshard, order)
            ), patch_reshard(functools.partial(patched_reshard, order)):
                optim.zero_grad(set_to_none=True)
                out = model(inp)
                loss = out.sum()
                losses.append(loss)
                loss.backward()
                optim.step()
        self.assertEqual(losses[0], losses[1])

        # Sanity check that the unshard/reshard events were recorded, where we
        # expect one unshard/reshard pair for forward, one pair for backward,
        # and possibly some extra unshards from backward prefetching (in this
        # case, we expect exactly 2 extra since there are 3 handles)
        self.assertGreaterEqual(len(composable_order), 2 * 2 * num_handles)
        self.assertGreaterEqual(len(wrapped_order), 2 * 2 * num_handles)
        self.assertGreaterEqual(
            len([e for e in composable_order if e[0] == "unshard"]), 2 * num_handles
        )
        self.assertGreaterEqual(
            len([e for e in wrapped_order if e[0] == "unshard"]), 2 * num_handles
        )
        self.assertGreaterEqual(
            len([e for e in composable_order if e[0] == "reshard"]), 2 * num_handles
        )
        self.assertGreaterEqual(
            len([e for e in wrapped_order if e[0] == "reshard"]), 2 * num_handles
        )

        # Check that the unshard/reshard order matches
        self.assertEqual(len(composable_order), len(wrapped_order))
        for (
            (composable_event, composable_handles_key),
            (wrapped_event, wrapped_handles_key),
        ) in zip(composable_order, wrapped_order):
            self.assertEqual(composable_event, wrapped_event)
            self._check_same_param_handles(composable_handles_key, wrapped_handles_key)

    def _check_same_param_handles(
        self,
        composable_handles: Iterable[FlatParamHandle],
        wrapped_handles: Iterable[FlatParamHandle],
    ) -> None:
        """
        Checks that ``composable_handles`` matches ``wrapped_handles`` by
        checking FQNs.

        For ``fully_shard``, each ``FlatParamHandle`` 's saved FQNs are
        prefixed from the local FSDP root, while for wrapper FSDP, they are
        prefixed from its owning FSDP instance, which may not be the local FSDP
        root. Thus, we relax the check to only that the wrapper FQN is a suffix
        of the composable FQN.

        If this check passes for the entire model and we separately unit-test
        parity for wrapping policies, then we can be sure that the handles
        actually match.
        """
        self.assertEqual(len(composable_handles), len(wrapped_handles))
        for composable_handle, wrapped_handle in zip(
            composable_handles, wrapped_handles
        ):
            composable_fqns = composable_handle.flat_param._fqns
            wrapped_fqns = wrapped_handle.flat_param._fqns
            self.assertEqual(len(composable_fqns), len(wrapped_fqns))
            for composable_fqn, wrapped_fqn in zip(composable_fqns, wrapped_fqns):
                self.assertTrue(composable_fqn.endswith(wrapped_fqn))


class TestMixedPrecision(FSDPTest):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_float16_on_one_submodule(self):
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        float16 = MixedPrecision(param_dtype=torch.float16)

        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs,
            cast_forward_inputs=False,
        ).cuda()
        c1, c2 = model.c1, model.c2
        x = torch.zeros(2, 100, device="cuda")

        # float16 on one submodule and float32 on everything else
        model.c2 = fully_shard(model.c2, mixed_precision=float16)
        fsdp = fully_shard(model)

        with self.assertRaisesRegex(
            TypeError,
            "cannot assign 'torch.cuda.FloatTensor' as parameter 'weight'",
        ):
            # FIXME: fully_shard() does not support nested wrapping yet.
            fsdp(x).sum().backward()

            self.assertEqual(forward_inputs[model].dtype, torch.float32)
            self.assertEqual(forward_inputs[c1].dtype, torch.float32)
            self.assertEqual(forward_inputs[c2].dtype, torch.float16)


class TestFSDPModelCheckpointing(FSDPTest):
    """Tests composable FSDP model checkpointing."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_state_dict_root_fully_shard(self):
        """
        Tests that the full state dict saved from a module with ``fully_shard``
        applied to the global root matches that of an equivalent local module.
        """
        local_model = CompositeParamModel(device=torch.device("cuda"))
        composable_module = copy.deepcopy(local_model)
        fully_shard(composable_module, policy=ModuleWrapPolicy({UnitModule}))
        local_sd = local_model.state_dict()
        composable_sd = composable_module.state_dict()
        self._check_state_dict_parity(local_sd, composable_sd)

    @skip_if_lt_x_gpu(2)
    def test_state_dict_submodule_fully_shard(self):
        """
        Tests that the full state dict saved from a module with ``fully_shard``
        applied on submodules matches that of an equivalent local module.
        """
        local_model = CompositeParamModel(device=torch.device("cuda"))
        composable_module = copy.deepcopy(local_model)
        fully_shard(composable_module.u1)
        fully_shard(composable_module.u2)
        local_sd = local_model.state_dict()
        composable_sd = composable_module.state_dict()
        self._check_state_dict_parity(local_sd, composable_sd)

    def _check_state_dict_parity(self, local_sd: Dict, composable_sd: Dict):
        """Checks that ``local_sd`` and ``composable_sd`` are the same."""
        # Check that all keys match
        for k1, k2 in zip(composable_sd.keys(), local_sd.keys()):
            self.assertEqual(k1, k2)
        # Check that all values match
        for v1, v2 in zip(composable_sd.values(), local_sd.values()):
            self.assertEqual(v1.shape, v2.shape)
        # Check that all keys and values are aligned
        for (k1, v1), (k2, v2) in zip(composable_sd.items(), local_sd.items()):
            self.assertEqual(k1, k2)
            self.assertEqual(v1, v2)


if __name__ == "__main__":
    run_tests()
