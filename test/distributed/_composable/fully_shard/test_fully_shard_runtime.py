# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import sys
from enum import auto, Enum
from typing import Callable, List, Tuple

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.fsdp.flat_param import FlatParamHandle
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
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


class FSDPWrapMode(Enum):
    AUTO_WRAP = auto()
    MANUAL_WRAP = auto()


class TestRuntime(FSDPTest):
    """Tests ``fully_shard`` runtime (forward/backward/optimizer)."""

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    def _init_models_and_optims(
        self,
        device: torch.device,
        fsdp_wrap_mode: FSDPWrapMode,
        sharding_strategy: ShardingStrategy,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, nn.Module, torch.optim.Optimizer]:
        local_model = CompositeParamModel(device=device)

        composable_module = copy.deepcopy(local_model)
        if fsdp_wrap_mode == FSDPWrapMode.AUTO_WRAP:
            fsdp_wrapped_model = FSDP(
                copy.deepcopy(local_model),
                auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
                use_orig_params=True,
                sharding_strategy=sharding_strategy,
            )
            fully_shard(
                composable_module,
                policy=ModuleWrapPolicy({UnitModule}),
                strategy=sharding_strategy,
            )
        elif fsdp_wrap_mode == FSDPWrapMode.MANUAL_WRAP:
            fsdp_wrapped_model = copy.deepcopy(local_model)
            fsdp_wrapped_model.u2 = FSDP(
                fsdp_wrapped_model.u2,
                use_orig_params=True,
                sharding_strategy=sharding_strategy,
            )
            fsdp_wrapped_model = FSDP(
                fsdp_wrapped_model,
                use_orig_params=True,
                sharding_strategy=sharding_strategy,
            )
            fully_shard(composable_module.u2, strategy=sharding_strategy)
            fully_shard(composable_module, strategy=sharding_strategy)
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
                ],
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                    ShardingStrategy.HYBRID_SHARD,
                ],
            },
            self._test_training,
        )

    def _test_training(
        self, fsdp_wrap_mode: FSDPWrapMode, sharding_strategy: ShardingStrategy
    ):
        if (
            sharding_strategy
            in [ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2]
            and fsdp_wrap_mode == FSDPWrapMode.MANUAL_WRAP
        ):
            return  # TODO: manual wrap + HSDP requires explicit specification of pg

        device = torch.device("cuda")
        (
            composable_module,
            composable_optim,
            fsdp_wrapped_model,
            fsdp_wrapped_optim,
        ) = self._init_models_and_optims(device, fsdp_wrap_mode, sharding_strategy)
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
        ) = self._init_models_and_optims(
            device, fsdp_wrap_mode, ShardingStrategy.FULL_SHARD
        )
        # Before checking the unshard/reshard order, sanity check that the
        # assumption about wrapper FQN being a suffix of composable FQN holds
        all_composable_handles = traversal_utils._get_fsdp_handles(composable_module)
        all_wrapped_handles = traversal_utils._get_fsdp_handles(fsdp_wrapped_model)
        for c_handle, w_handle in zip(all_composable_handles, all_wrapped_handles):
            self._check_same_param_handles(c_handle, w_handle)
        num_handles = len(all_composable_handles)

        orig_unshard = torch.distributed.fsdp._runtime_utils._unshard
        orig_reshard = torch.distributed.fsdp._runtime_utils._reshard
        UnshardReshardEvent = Tuple[str, FlatParamHandle]

        def patched_unshard(
            unshard_reshard_order: List[UnshardReshardEvent],
            state: _FSDPState,
            handle: FlatParamHandle,
            *args,
            **kwargs,
        ):
            unshard_reshard_order.append(("unshard", handle))
            return orig_unshard(state, handle, *args, **kwargs)

        def patched_reshard(
            unshard_reshard_order: List[UnshardReshardEvent],
            state: _FSDPState,
            handle: FlatParamHandle,
            *args,
            **kwargs,
        ):
            unshard_reshard_order.append(("reshard", handle))
            return orig_reshard(state, handle, *args, **kwargs)

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
        composable_handle: FlatParamHandle,
        wrapped_handle: FlatParamHandle,
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
        composable_fqns = composable_handle.flat_param._fqns
        wrapped_fqns = wrapped_handle.flat_param._fqns
        self.assertEqual(len(composable_fqns), len(wrapped_fqns))
        for composable_fqn, wrapped_fqn in zip(composable_fqns, wrapped_fqns):
            self.assertTrue(composable_fqn.endswith(wrapped_fqn))


if __name__ == "__main__":
    run_tests()
