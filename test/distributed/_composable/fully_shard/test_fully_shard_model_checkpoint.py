# Owner(s): ["oncall: distributed"]

import copy
import itertools
import sys
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import fully_shard
from torch.distributed._state_dict_utils import _gather_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,
    UnitModule,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
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


class TestModelCheckpointing(FSDPTest):
    """Tests ``fully_shard`` model checkpointing."""

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
        self.run_subtests(
            {"ignore_modules": [False, True], "sharded_state_dict": [False, True]},
            self._test_save_dict_save_load_flow,
        )

    def _test_save_dict_save_load_flow(
        self, ignore_modules: bool, sharded_state_dict: bool
    ):
        local_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )

        # Force model parameters and buffers to be nonzero
        for tensor in itertools.chain(local_model.parameters(), local_model.buffers()):
            if torch.count_nonzero(tensor) == 0:
                with torch.no_grad():
                    tensor.add_(torch.ones_like(tensor))

        save_model = copy.deepcopy(local_model)
        fully_shard(
            save_model,
            policy=ModuleWrapPolicy({TransformerEncoderLayer, TransformerDecoderLayer}),
            ignored_modules=(
                save_model.get_ignored_modules() if ignore_modules else []
            ),
        )

        # TODO: test state_dict_type after https://github.com/pytorch/pytorch/issues/90954 is resolved
        if not sharded_state_dict:
            FSDP.set_state_dict_type(save_model, StateDictType.FULL_STATE_DICT)
        else:
            FSDP.set_state_dict_type(save_model, StateDictType.SHARDED_STATE_DICT)
        state_dict = save_model.state_dict()
        local_state_dict = local_model.state_dict()
        self._check_state_dict_parity(local_state_dict, _gather_state_dict(state_dict))

        load_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
        )
        _zero_model(load_model, zero_buffers=True, summon_full=False)
        fully_shard(
            load_model,
            policy=ModuleWrapPolicy({TransformerDecoderLayer, TransformerEncoderLayer}),
            ignored_modules=(
                load_model.get_ignored_modules() if ignore_modules else []
            ),
        )
        if not sharded_state_dict:
            FSDP.set_state_dict_type(load_model, StateDictType.FULL_STATE_DICT)
        else:
            FSDP.set_state_dict_type(load_model, StateDictType.SHARDED_STATE_DICT)
        load_model.load_state_dict(state_dict)
        self._check_model_parity(load_model, save_model)

    @skip_if_lt_x_gpu(2)
    def test_full_state_dict_save_load_mixed_sharding(self):
        """
        Tests that the full state dict saved from a module with ``fully_shard``
        and ``no_shard`` applied on the module matches that of an equivalent
        local module. Also ensures that this state_dict can be reloaded into
        a composable module and is equivalent to the original composable module.
        """
        local_model = CompositeParamModel(device=torch.device("cuda"))

        def _create_mixed_shard_on_model(mod: nn.Module):
            fully_shard(mod.u1)
            fully_shard(mod, strategy=ShardingStrategy.NO_SHARD)
            return mod

        save_composable = copy.deepcopy(local_model)
        save_composable = _create_mixed_shard_on_model(save_composable)
        local_sd = local_model.state_dict()
        composable_sd = save_composable.state_dict()
        self._check_state_dict_parity(local_sd, composable_sd)

        # Validate load
        load_composable = copy.deepcopy(local_model)
        load_composable = _create_mixed_shard_on_model(load_composable)
        _zero_model(load_composable, summon_full=False)
        for p in load_composable.parameters():
            self.assertEqual(0, p.sum())

        sd = {k: v.clone() for k, v in composable_sd.items()}
        load_composable.load_state_dict(sd)
        self._check_model_parity(load_composable, save_composable)

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
        Checks that ``m1`` and ``m2`` have equal ``named_parameters()``.
        """
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertEqual(p1, p2)


if __name__ == "__main__":
    run_tests()
