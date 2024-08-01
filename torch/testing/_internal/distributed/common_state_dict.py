# mypy: allow-untyped-defs

# Owner(s): ["oncall: distributed"]

import copy
from itertools import chain
from typing import Any, Dict

import torch
import torch.nn as nn

from torch.distributed._sharded_tensor import ShardedTensor
from torch.distributed._state_dict_utils import _gather_state_dict
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    _PG,
    _STATE,
    set_state_dict,
    StateDictOptions,
)


class VerifyStateDictMixin:
    def _compare_tensor(self, orig_tensor, dist_tensor, offload_to_cpu=False):
        if isinstance(dist_tensor, (DTensor, ShardedTensor)):
            dist_tensor = _gather_state_dict({"mykey": dist_tensor}).pop("mykey")

        if offload_to_cpu:
            orig_tensor = orig_tensor.cpu()
            dist_tensor = dist_tensor.cpu()
        self.assertTrue(isinstance(dist_tensor, torch.Tensor))
        self.assertTrue(torch.allclose(orig_tensor, dist_tensor))

    def _verify_msd(
        self,
        msd: Dict[str, Any],
        dist_msd: Dict[str, Any],
        options: StateDictOptions = StateDictOptions(),
        offload_to_cpu=False,
    ) -> None:
        if not options.ignore_frozen_params:
            self.assertEqual(len(msd), len(dist_msd))
        for fqn, param in msd.items():
            dist_param = dist_msd.get(fqn, None)
            if not options.ignore_frozen_params:
                self.assertIsNotNone(dist_param, f"{fqn=}")
                self._compare_tensor(param, dist_param, offload_to_cpu)
            elif dist_param is None:
                self.assertFalse(param.requires_grad, f"{fqn=}")

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

        self.assertEqual(len(osd[_STATE]), len(dist_osd[_STATE]))
        for pid, states in osd[_STATE].items():
            fqn = fqn_pid_mapping[pid]
            dist_states = dist_osd[_STATE].get(fqn, None)
            self.assertIsNotNone(dist_states, fqn)
            self.assertEqual(len(states), len(dist_states))
            for key, state in states.items():
                dist_state = states.get(key, None)
                self.assertIsNotNone(dist_state)
                self._compare_tensor(state, dist_state)

        # Check optimizer_state_dict param_group
        old_dist_osd_pg = dist_osd[_PG]
        if len(osd[_PG]) != len(dist_osd[_PG]):
            self.assertTrue(len(dist_osd[_PG]) > len(osd[_PG]))
            new_pg = copy.deepcopy(dist_osd[_PG][0])
            new_pg["params"] = []
            for dist_group in dist_osd[_PG]:
                new_pg["params"].extend(dist_group["params"])
            dist_osd[_PG] = [new_pg]

        self.assertEqual(len(osd[_PG]), len(dist_osd[_PG]))
        for group, dist_group in zip(osd[_PG], dist_osd[_PG]):
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
        dist_osd[_PG] = old_dist_osd_pg

    def _verify_osd_by_load(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        new_optim: torch.optim.Optimizer,
        dist_osd: Dict[str, Any],
    ) -> None:
        new_dist_osd = _gather_state_dict(dist_osd)
        set_state_dict(
            model,
            optimizers=new_optim,
            model_state_dict={},
            optim_state_dict=new_dist_osd,
        )
        self.assertEqual(optim.state_dict(), new_optim.state_dict())
