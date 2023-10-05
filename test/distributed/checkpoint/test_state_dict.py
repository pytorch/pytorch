# Owner(s): ["oncall: distributed"]

import copy
import sys
from itertools import chain
from typing import Any, Callable, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import fully_shard, replicate
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
    DistributedStateDictOptions,
    load_state_dict,
    PG,
    STATE,
    state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._shard_utils import _gather_state_dict
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,
    UnitModule,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import TEST_WITH_DEV_DBG_ASAN


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestStateDict(FSDPTest):
    """Tests state_dict and load_state_dict"""

    @property
    def world_size(self) -> int:
        return 2

    def _compare_tensor(self, orig_tensor, dist_tensor):
        if isinstance(dist_tensor, (DTensor, ShardedTensor)):
            dist_tensor = _gather_state_dict({"mykey": dist_tensor}).pop("mykey")
        self.assertTrue(isinstance(dist_tensor, torch.Tensor))
        self.assertTrue(torch.allclose(orig_tensor, dist_tensor))

    def _verify_msd(
        self,
        model: nn.Module,
        msd: Dict[str, Any],
        dist_msd: Dict[str, Any],
        options: DistributedStateDictOptions,
    ) -> None:
        if options.save_frozen_params:
            self.assertEqual(len(msd), len(dist_msd))
        for fqn, param in msd.items():
            dist_param = dist_msd.get(fqn, None)
            if options.save_frozen_params:
                self.assertIsNotNone(dist_param)
                self._compare_tensor(param, dist_param)
            elif dist_param is None:
                self.assertFalse(param.requires_grad)

    def _verify_osd(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        osd: Dict[str, Any],
        dist_osd: Dict[str, Any],
    ) -> None:
        params = list(chain.from_iterable(g["params"] for g in optim.param_groups))
        param_pid_mapping = dict(zip(params, range(len(params))))
        fqn_pid_mapping = {}
        for fqn, param in model.named_parameters():
            pid = param_pid_mapping[param]
            fqn_pid_mapping[fqn] = pid
            fqn_pid_mapping[pid] = fqn
        # Check optimizer_state_dict state

        self.assertEqual(len(osd[STATE]), len(dist_osd[STATE]))
        for pid, states in osd[STATE].items():
            fqn = fqn_pid_mapping[pid]
            dist_states = dist_osd[STATE].get(fqn, None)
            self.assertIsNotNone(dist_states, fqn)
            self.assertEqual(len(states), len(dist_states))
            for key, state in states.items():
                dist_state = states.get(key, None)
                self.assertIsNotNone(dist_state)
                self._compare_tensor(state, dist_state)

        # Check optimizer_state_dict param_group
        old_dist_osd_pg = dist_osd[PG]
        if len(osd[PG]) != len(dist_osd[PG]):
            self.assertTrue(len(dist_osd[PG]) > len(osd[PG]))
            new_pg = copy.deepcopy(dist_osd[PG][0])
            new_pg["params"] = []
            for dist_group in dist_osd[PG]:
                new_pg["params"].extend(dist_group["params"])
            dist_osd[PG] = [new_pg]

        self.assertEqual(len(osd[PG]), len(dist_osd[PG]))
        for group, dist_group in zip(osd[PG], dist_osd[PG]):
            self.assertEqual(len(group), len(dist_group))
            for key, value in group.items():
                # Below doesn't work because param_groups can have None
                # values.
                # dist_value = dist_group.get(key, None)
                # self.assertIsNotNone(dist_value, (dist_group, group))
                dist_value = dist_group[key]
                if key == "params":
                    fqns = [fqn_pid_mapping[pid] for pid in value]
                    self.assertEqual(sorted(fqns), sorted(dist_value))
                else:
                    self.assertEqual(value, dist_value)
        dist_osd[PG] = old_dist_osd_pg

    def _verify_osd_by_load(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        new_optim: torch.optim.Optimizer,
        dist_osd: Dict[str, Any],
    ) -> None:
        new_dist_osd = _gather_state_dict(dist_osd)
        load_state_dict(
            model,
            [new_optim],
            model_state_dict={},
            optim_state_dict=new_dist_osd,
            optim_only=True,
        )
        self.assertEqual(optim.state_dict(), new_optim.state_dict())

    def _test_save_load(
        self,
        init_model_optim: Callable,
        test_frozen: bool = False,
    ) -> None:
        options = DistributedStateDictOptions(save_frozen_params=(not test_frozen))
        # Initialize original model and distributed model.
        model, optim, copy_optim, dist_model, dist_optim = init_model_optim()

        # Train 10 steps.
        for i in range(10):
            batch = torch.rand(8, 100, device="cuda")
            model(batch).sum().backward()
            optim.step()
            dist_model(batch).sum().backward()
            if not isinstance(dist_optim, list):
                dist_optim.step()
                dist_optim.zero_grad()
            else:
                for _dist_optim in dist_optim:
                    _dist_optim.zero_grad()
            optim.zero_grad()

        # Get the state_dict, and compare the result
        msd = model.state_dict()
        osd = optim.state_dict()
        if not isinstance(dist_optim, list):
            dist_optim = [dist_optim]
        dist_msd, dist_osd = state_dict(dist_model, dist_optim, options=options)
        self._verify_msd(model, msd, dist_msd, options)
        self._verify_osd_by_load(model, optim, copy_optim, dist_osd)
        self._verify_osd(model, optim, osd, dist_osd)

        # Initialize a completely new model to simulate checkpoint load.
        _, _, _, dist_model, dist_optim = init_model_optim()

        # Simulate DCP distributed load. We need to first get the state_dict and
        # pass them to DCP to load the saved state_dict from the storage.
        # Then finally we can call load_state_dict().
        if not isinstance(dist_optim, list):
            dist_optim = [dist_optim]
        curr_dist_msd, curr_dist_osd = state_dict(
            dist_model, dist_optim, options=options
        )
        if test_frozen:
            # We won't be able to load the partial state_dict back.
            return
        # Since we already have the state_dict saved before, no need to call DCP.
        # We can directly load them back. This asser is to ensure that optimizer
        # state storage are initialized.
        # self.assertEqual(len(curr_dist_osd[STATE]), len(dist_osd[STATE]))
        load_state_dict(
            dist_model,
            dist_optim,
            model_state_dict=dist_msd,
            optim_state_dict=dist_osd,
            options=options,
        )

        # Check if the new state_dict are the same
        dist_msd, dist_osd = state_dict(dist_model, dist_optim, options=options)
        self._verify_msd(model, msd, dist_msd, options)
        self._verify_osd_by_load(model, optim, copy_optim, dist_osd)
        self._verify_osd(model, optim, osd, dist_osd)

        # Test _patch_model_state_dict, and _patch_optimizer_state_dict
        _patch_model_state_dict(dist_model, options=options)
        _patch_optimizer_state_dict(dist_model, dist_optim, options=options)
        dist_msd = dist_model.state_dict()
        dist_osd = dist_optim[0].state_dict()
        self._verify_msd(model, msd, dist_msd, options)
        self._verify_osd_by_load(model, optim, copy_optim, dist_osd)
        self._verify_osd(model, optim, osd, dist_osd)

    def _test_fsdp(self, use_orig_params: bool, use_composable: bool) -> None:
        if not use_orig_params and use_composable:
            return

        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            orig_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            copy_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            if use_composable:
                dist_model = fully_shard(
                    copy.deepcopy(orig_model), policy=ModuleWrapPolicy({UnitModule})
                )
            else:
                dist_model = FSDP(
                    copy.deepcopy(orig_model),
                    auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
                    use_orig_params=use_orig_params,
                )
            dist_optim = torch.optim.Adam(dist_model.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim)

    @skip_if_lt_x_gpu(2)
    def test_fsdp(self) -> None:
        self.run_subtests(
            {"use_orig_params": [True, False], "use_composable": [True, False]},
            self._test_fsdp,
        )

    def _test_ddp(self, use_composable: bool) -> None:
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            orig_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            copy_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            if use_composable:
                dist_model = replicate(copy.deepcopy(orig_model))
            else:
                dist_model = DDP(copy.deepcopy(orig_model))
            dist_optim = torch.optim.Adam(dist_model.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim)

    @skip_if_lt_x_gpu(2)
    def test_ddp(self) -> None:
        self.run_subtests(
            {"use_composable": [True, False]},
            self._test_ddp,
        )

    def _test_fsdp_ddp(
        self,
        use_composable: bool,
        optim_in_backward: bool = False,
        test_frozen: bool = False,
    ) -> None:
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            if test_frozen:
                for param in chain(
                    orig_model.u1.parameters(), orig_model.u2.parameters()
                ):
                    param.requires_grad = False
            orig_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            copy_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            dist_model = copy.deepcopy(orig_model)
            if use_composable:
                replicate(dist_model.l)
                fully_shard(dist_model, policy=ModuleWrapPolicy({UnitModule}))
            else:
                dist_model.l = DDP(dist_model.l)
                dist_model = FSDP(
                    copy.deepcopy(orig_model),
                    auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
                    use_orig_params=optim_in_backward,
                    ignored_modules=[dist_model.l],
                )
            if optim_in_backward:
                _apply_optimizer_in_backward(
                    torch.optim.Adam, dist_model.parameters(), {"lr": 1e-3}
                )
                dist_optim = [
                    p._in_backward_optimizers[0] for p in dist_model.parameters()
                ]
            else:
                dist_optim = torch.optim.Adam(dist_model.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim, test_frozen)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_ddp(self) -> None:
        self.run_subtests(
            {"use_composable": [True, False]},
            self._test_fsdp_ddp,
        )

    @skip_if_lt_x_gpu(2)
    def test_frozen_parameters(self) -> None:
        self._test_fsdp_ddp(use_composable=False, test_frozen=True)

    # TODO: enable use_dtensor once 2D device_mesh support is fully landed.
    """
    @skip_if_lt_x_gpu(2)
    def test_use_dtensor(self) -> None:
        self._test_fsdp_ddp(use_composable=False, use_dtensor=True)
    """

    # TODO: enable the test after FSDP + apply_optimizer_in_backward works.
    # Disable this test as it is broken after
    # https://github.com/pytorch/pytorch/pull/108298.
    """
    @skip_if_lt_x_gpu(2)
    def test_apply_optimizer_in_backward(self) -> None:
        self.run_subtests(
            {"use_composable": [True, False]},
            self._test_fsdp_ddp,
            optim_in_backward=True,
        )
    """

    @skip_if_lt_x_gpu(1)
    def test_single_gpu(self) -> None:
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            orig_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            copy_optim = torch.optim.Adam(orig_model.parameters(), lr=1e-3)
            model_copy = copy.deepcopy(orig_model)
            optim_copy = torch.optim.Adam(model_copy.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, model_copy, optim_copy

        self._test_save_load(init_model_optim)
