# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import itertools
import sys
from enum import auto, Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import _FSDPState, _is_fsdp_flattened
from torch.distributed.fsdp.api import MixedPrecision
from torch.distributed.fsdp.flat_param import _HandlesKey, FlatParamHandle
from torch.distributed.fsdp.wrap import _FSDPPolicy, ModuleWrapPolicy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,
    NestedSequentialModel,
    UnitModule,
)
from torch.testing._internal.common_distributed import (
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    _zero_model,
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
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


class FSDPWrapMode(Enum):
    AUTO_WRAP = auto()
    MANUAL_WRAP = auto()


class TestFSDPInitialization(FSDPTest):
    """Tests composable FSDP initialization."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_policy(self):
        """Tests passing a ``policy`` for pseudo-auto-wrapping."""
        self.run_subtests(
            {
                "policy": [
                    None,
                    ModuleWrapPolicy({UnitModule}),
                    ModuleWrapPolicy({nn.Sequential}),
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
        for (composable_handle, fsdp_wrapped_handle) in zip(
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

    @skip_if_lt_x_gpu(2)
    def test_nested_fully_shard_shared_state(self):
        """
        Tests that nested applications of ``fully_shard`` share the expected
        data structure state.
        """
        device = torch.device("cuda")
        composable_module = CompositeParamModel(device=device)
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
        data_structure_names = ["_streams", "_exec_order_data", "_free_event_queue"]
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


class TestFSDPRuntime(FSDPTest):
    """Tests composable FSDP runtime."""

    @property
    def world_size(self) -> int:
        return 2

    def _init_models_and_optims(
        self,
        device: torch.device,
        fsdp_wrap_mode: FSDPWrapMode,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module, torch.optim.Optimizer]:
        local_model = CompositeParamModel(device=device)

        composable_module = copy.deepcopy(local_model)
        if fsdp_wrap_mode == FSDPWrapMode.AUTO_WRAP:
            fsdp_wrapped_model = FSDP(
                copy.deepcopy(local_model),
                auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
                use_orig_params=True,
            )
            fully_shard(
                composable_module,
                policy=ModuleWrapPolicy({UnitModule}),
            )
        elif fsdp_wrap_mode == FSDPWrapMode.MANUAL_WRAP:
            fsdp_wrapped_model = copy.deepcopy(local_model)
            fsdp_wrapped_model.u2 = FSDP(fsdp_wrapped_model.u2, use_orig_params=True)
            fsdp_wrapped_model = FSDP(fsdp_wrapped_model, use_orig_params=True)
            fully_shard(composable_module.u2)
            fully_shard(composable_module)
        else:
            raise ValueError(f"Unknown `fsdp_wrap_mode`: {fsdp_wrap_mode}")
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
        self.run_subtests(
            {
                "fsdp_wrap_mode": [
                    FSDPWrapMode.AUTO_WRAP,
                    FSDPWrapMode.MANUAL_WRAP,
                ]
            },
            self._test_training,
        )

    def _test_training(self, fsdp_wrap_mode: FSDPWrapMode):
        device = torch.device("cuda")
        (
            composable_module,
            composable_optim,
            fsdp_wrapped_model,
            fsdp_wrapped_optim,
        ) = self._init_models_and_optims(device, fsdp_wrap_mode)
        torch.manual_seed(self.rank + 1)
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
        self.run_subtests(
            {"fsdp_wrap_mode": [FSDPWrapMode.AUTO_WRAP, FSDPWrapMode.MANUAL_WRAP]},
            self._test_unshard_reshard_order,
        )

    def _test_unshard_reshard_order(self, fsdp_wrap_mode: FSDPWrapMode):
        device = torch.device("cuda")
        (
            composable_module,
            composable_optim,
            fsdp_wrapped_model,
            fsdp_wrapped_optim,
        ) = self._init_models_and_optims(device, fsdp_wrap_mode)
        # Before checking the unshard/reshard order, sanity check that the
        # assumption about wrapper FQN being a suffix of composable FQN holds
        all_composable_handles = traversal_utils._get_fsdp_handles(composable_module)
        all_wrapped_handles = traversal_utils._get_fsdp_handles(fsdp_wrapped_model)
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
    def test_state_dict_save_load_root_fully_shard(self):
        """
        Tests that the full state dict saved from a module with ``fully_shard``
        applied to the global root matches that of an equivalent local module. Also
        ensure that this state_dict can be reloaded into a composable module and
        is equivalent to the original composable module.
        """
        local_model = CompositeParamModel(device=torch.device("cuda"))
        save_composable = copy.deepcopy(local_model)
        fully_shard(save_composable, policy=ModuleWrapPolicy({UnitModule}))
        local_sd = local_model.state_dict()
        composable_sd = save_composable.state_dict()
        self._check_state_dict_parity(local_sd, composable_sd)

        # Validate load
        load_composable = fully_shard(
            copy.deepcopy(local_model), policy=ModuleWrapPolicy({UnitModule})
        )
        _zero_model(load_composable, summon_full=False)
        for p in load_composable.parameters():
            self.assertEqual(p.sum(), 0)

        sd = {k: v.clone() for k, v in composable_sd.items()}
        load_composable.load_state_dict(sd)
        self._check_model_parity(load_composable, save_composable)

    @skip_if_lt_x_gpu(2)
    def test_state_dict_save_load_submodule_fully_shard(self):
        """
        Tests that the full state dict saved from a module with ``fully_shard``
        applied on submodules matches that of an equivalent local module. Also
        ensures that this state_dict can be reloaded into a composable module and
        is equivalent to the original composable module.
        """
        local_model = CompositeParamModel(device=torch.device("cuda"))

        def _create_fully_shard_on_submodules(mod: nn.Module):
            fully_shard(mod.u1)
            fully_shard(mod.u2)
            return mod

        save_composable = copy.deepcopy(local_model)
        save_composable = _create_fully_shard_on_submodules(save_composable)
        local_sd = local_model.state_dict()
        composable_sd = save_composable.state_dict()
        self._check_state_dict_parity(local_sd, composable_sd)

        # Validate load
        load_composable = copy.deepcopy(local_model)
        load_composable = _create_fully_shard_on_submodules(load_composable)
        _zero_model(load_composable, summon_full=False)
        for p in load_composable.parameters():
            self.assertEqual(0, p.sum())

        sd = {k: v.clone() for k, v in composable_sd.items()}
        load_composable.load_state_dict(sd)
        self._check_model_parity(load_composable, save_composable)

    @skip_if_lt_x_gpu(2)
    def test_state_dict_save_load_flow(self):
        """
        E2E test of save + load with rank0_only + CPU offload for TransformerWithSharedParams
        on the composable path.
        """
        # TODO refactor to use self.run_subtests
        for ignore_modules in [True, False]:
            with self.subTest(ignore_modules=ignore_modules):
                local_model = TransformerWithSharedParams.init(
                    self.process_group,
                    FSDPInitMode.NO_FSDP,
                    CUDAInitMode.CUDA_BEFORE,
                    deterministic=True,
                )

                # force model parameters and buffers to be nonzero
                for tensor in itertools.chain(
                    local_model.parameters(), local_model.buffers()
                ):
                    if torch.count_nonzero(tensor) == 0:
                        with torch.no_grad():
                            tensor.add_(torch.ones_like(tensor))

                save_model = copy.deepcopy(local_model)
                save_model = fully_shard(
                    save_model,
                    policy=ModuleWrapPolicy(
                        {TransformerEncoderLayer, TransformerDecoderLayer}
                    ),
                    ignored_modules=(
                        save_model.get_ignored_modules() if ignore_modules else []
                    ),
                )

                # TODO: test state_dict_type after https://github.com/pytorch/pytorch/issues/90954 is resolved
                state_dict = save_model.state_dict()
                local_state_dict = local_model.state_dict()

                self._check_state_dict_parity(local_model.state_dict(), state_dict)

                load_model = TransformerWithSharedParams.init(
                    self.process_group,
                    FSDPInitMode.NO_FSDP,
                    CUDAInitMode.CUDA_BEFORE,
                )
                _zero_model(load_model, zero_buffers=True, summon_full=False)
                fully_shard(
                    load_model,
                    policy=ModuleWrapPolicy(
                        {TransformerDecoderLayer, TransformerEncoderLayer}
                    ),
                    ignored_modules=(
                        load_model.get_ignored_modules() if ignore_modules else []
                    ),
                )
                load_model.load_state_dict(state_dict)
                self._check_model_parity(load_model, save_model)

    def _check_state_dict_parity(self, local_sd: Dict, composable_sd: Dict):
        """Checks that ``local_sd`` and ``composable_sd`` are the same."""
        # Check that all keys match
        self.assertEqual(set(composable_sd.keys()), set(local_sd.keys()))
        # Check value shapes
        for k in composable_sd.keys():
            v1 = composable_sd[k]
            v2 = local_sd[k]
            self.assertEqual(
                v1.shape, v2.shape, f"Shape mismatch for {k} {v1.shape} vs {v2.shape}"
            )

        # Check actual values
        for k in composable_sd.keys():
            v1 = composable_sd[k]
            v2 = local_sd[k]
            self.assertEqual(v1, v2, f"Param mismatch for {k}: {v1} vs {v2}")

    def _check_model_parity(self, m1: nn.Module, m2: nn.Module):
        """
        Checks that m1 and m2 have equivalent named_parameters.
        """
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertEqual(p1, p2)


if __name__ == "__main__":
    run_tests()
