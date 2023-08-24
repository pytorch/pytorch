# Owner(s): ["oncall: distributed"]

import contextlib
import sys
from collections import Counter
from enum import auto, Enum
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.distributed_c10d import _rank_not_in_group
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp._init_utils import (
    _init_intra_and_inter_node_groups,
    HYBRID_SHARDING_STRATEGIES,
)

from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
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


@contextlib.contextmanager
def patch_allreduce(new_allreduce):
    """
    Patches dist.all_reduce with a new all_reduce and
    restores upon exiting.
    """
    orig_ar = dist.all_reduce
    dist.all_reduce = new_allreduce
    try:
        yield
    finally:
        dist.all_reduce = orig_ar


@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter):
    """
    Patches dist.reduce_scatter_tensor with a new reduce_scatter_tensor and
    restores upon exiting.
    """
    orig_reduce_scatter = dist.reduce_scatter_tensor
    dist.reduce_scatter_tensor = new_reduce_scatter
    try:
        yield
    finally:
        dist.reduce_scatter_tensor = orig_reduce_scatter


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)
        self.lin3 = nn.Linear(10, 10)

    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(x)))


class ShardingStrategyMode(Enum):
    ALL_HYBRID_SHARD = auto()
    MIXED_HYBRID_FULL_SHARD = auto()


class TestFSDPHybridShard(FSDPTest):
    @property
    def world_size(self):
        return max(torch.cuda.device_count(), 2)

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_raises_manual_wrap_hybrid_shard_when_none_policy(self):
        model = MyModel().cuda()
        err_ctx = self.assertRaisesRegex(
            ValueError, "requires explicit specification of process group"
        )

        with err_ctx:
            model = FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD)

        with err_ctx:
            model = FSDP(model, sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2)

    @skip_if_lt_x_gpu(2)
    def test_hybrid_shard_pg_mismatch_raises(self):
        model = MyModel().cuda()
        intra_pg = self.process_group
        inter_pg = dist.new_group(ranks=[self.rank])
        # Mismatched process groups for intra-node
        model.lin1 = FSDP(
            model.lin1,
            process_group=(intra_pg, inter_pg),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        )
        model = FSDP(
            model,
            process_group=(dist.new_group(), dist.new_group()),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        )
        # Errors during _lazy_init
        inp = torch.randn(4, 10)
        with self.assertRaisesRegex(
            ValueError, "intra-node process groups do not match"
        ):
            model(inp)

        # Mismatched process groups for inter-node
        model = MyModel().cuda()
        model.lin1 = FSDP(
            model.lin1,
            process_group=(intra_pg, inter_pg),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        )
        model = FSDP(
            model,
            process_group=(intra_pg, dist.new_group()),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        )
        with self.assertRaisesRegex(
            ValueError, "inter-node process groups do not match"
        ):
            model(inp)

    @skip_if_lt_x_gpu(4)
    def test_hsdp_save_load_state_dict(self):
        model = MyModel().cuda()
        num_node_devices = torch.cuda.device_count()
        shard_rank_lists = list(range(0, num_node_devices // 2)), list(
            range(num_node_devices // 2, num_node_devices)
        )
        shard_groups = (
            dist.new_group(shard_rank_lists[0]),
            dist.new_group(shard_rank_lists[1]),
        )
        my_shard_group = (
            shard_groups[0] if self.rank in shard_rank_lists[0] else shard_groups[1]
        )
        my_replicate_group = None
        my_rank = self.rank
        # Create groups like (0, 4), (1, 5), (2, 6) etc and assign appropriately
        shard_factor = len(shard_rank_lists[0])
        for i in range(num_node_devices // 2):
            replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
            replicate_group = dist.new_group(replicate_group_ranks)
            if my_rank in replicate_group_ranks:
                my_replicate_group = replicate_group

        fsdp_ctor = partial(
            FSDP,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            process_group=(my_shard_group, my_replicate_group),
        )
        model = fsdp_ctor(model)
        optim = torch.optim.AdamW(model.parameters())
        # Initialize optimizer states
        model(torch.randn(2, 10)).sum().backward()
        optim.step()
        shard_g = model.process_group
        replicate_g = model._inter_node_pg
        assert shard_g == my_shard_group
        assert replicate_g == my_replicate_group
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            msd = model.state_dict()
            osd = FSDP.optim_state_dict(model, optim)

        load_model = fsdp_ctor(MyModel().cuda())
        load_optim = torch.optim.AdamW(load_model.parameters())
        with FSDP.state_dict_type(load_model, StateDictType.SHARDED_STATE_DICT):
            load_model.load_state_dict(msd)
            FSDP.optim_state_dict_to_load(load_model, load_optim, osd)
        load_optim.load_state_dict(osd)

    @skip_if_lt_x_gpu(2)
    def test_invalid_pg_specification_raises(self):
        pol = ModuleWrapPolicy({nn.Linear})
        model = MyModel().cuda()
        with self.assertRaisesRegex(
            ValueError, "Expected process_group to be passed in"
        ):
            model = FSDP(
                model,
                auto_wrap_policy=pol,
                process_group=self.process_group,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )

    # TODO - add test for ZeRO-2 style sharding ensure params are not
    # resharded after forward.

    @skip_if_lt_x_gpu(2)
    def test_fsdp_hybrid_shard_basic_setup(self):
        """
        Tests basic functionality of HYBRID_SHARD and _HYBRID_SHARD_ZERO2:
            1. Inter and intra-node process groups are correctly setup
            2. Process groups are the same across FSDP wrapped instances
            3. reduce_scatter and allreduce called the expected no. of times
        """
        self.run_subtests(
            {
                "hsdp_sharding_strategy": [
                    ShardingStrategy.HYBRID_SHARD,
                    ShardingStrategy._HYBRID_SHARD_ZERO2,
                ],
                "sharding_strategy_mode": [
                    ShardingStrategyMode.ALL_HYBRID_SHARD,
                    ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD,
                ],
                "use_orig_params": [False, True],
            },
            self._test_fsdp_hybrid_shard_basic_setup,
        )

    def _test_fsdp_hybrid_shard_basic_setup(
        self,
        hsdp_sharding_strategy: ShardingStrategy,
        sharding_strategy_mode: ShardingStrategyMode,
        use_orig_params: bool,
    ):
        hsdp_model = self._init_hsdp_model(
            hsdp_sharding_strategy, sharding_strategy_mode, use_orig_params
        )
        # All FSDP modules should have state.process_group as the process group over which to
        # shard (default process group), and state._inter_node_pg (process group containing only
        # this rank)
        intra_node_pgs = set()
        inter_node_pgs = set()
        for fsdp_module in hsdp_model.fsdp_modules(hsdp_model):
            # TODO: This needs to be replaced if we deprecate
            # `FSDP.sharding_strategy` to only use the handle one.
            # https://github.com/pytorch/pytorch/issues/90857
            if fsdp_module.sharding_strategy not in HYBRID_SHARDING_STRATEGIES:
                self.assertEqual(
                    sharding_strategy_mode, ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD
                )
                self.assertEqual(
                    fsdp_module.sharding_strategy, ShardingStrategy.FULL_SHARD
                )
                continue
            # process_group should be across the node, which is just the
            # whole world here.
            self.assertEqual(
                dist.get_world_size(fsdp_module.process_group),
                dist.get_world_size(self.process_group),
            )
            intra_node_pgs.add(fsdp_module.process_group)
            inter_node_pg = fsdp_module._inter_node_pg
            inter_node_pgs.add(inter_node_pg)
            self.assertEqual(1, dist.get_world_size(inter_node_pg))
            self.assertFalse(_rank_not_in_group(inter_node_pg))
            self.assertEqual(hsdp_sharding_strategy, fsdp_module.sharding_strategy)
        # All fsdp modules should share the same process groups
        self.assertEqual(1, len(intra_node_pgs))
        self.assertEqual(1, len(inter_node_pgs))

        orig_ar = dist.all_reduce
        orig_rs = dist.reduce_scatter_tensor

        def patched_collective(orig_collective, counter, *args, **kwargs):
            counter[orig_collective] += 1
            return orig_collective(*args, **kwargs)

        cntr = Counter()
        patched_allreduce = partial(patched_collective, orig_ar, cntr)
        patched_reduce_scatter = partial(patched_collective, orig_rs, cntr)
        with patch_allreduce(patched_allreduce), patch_reduce_scatter(
            patched_reduce_scatter
        ):
            inp = hsdp_model.get_input(device=torch.cuda.current_device())
            out = hsdp_model(inp[0], inp[1])
            loss = hsdp_model.get_loss(inp, out)
            loss.backward()

        if sharding_strategy_mode == ShardingStrategyMode.ALL_HYBRID_SHARD:
            num_flat_params = len(list(traversal_utils._get_fsdp_handles(hsdp_model)))
            self.assertEqual(num_flat_params, cntr[orig_ar])
            self.assertEqual(num_flat_params, cntr[orig_rs])
        elif sharding_strategy_mode == ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD:
            num_hsdp_flat_params = len(
                list(traversal_utils._get_fsdp_handles(hsdp_model.transformer))
            )
            num_flat_params = len(list(traversal_utils._get_fsdp_handles(hsdp_model)))
            self.assertEqual(num_hsdp_flat_params, cntr[orig_ar])
            self.assertEqual(num_flat_params, cntr[orig_rs])

    @skip_if_lt_x_gpu(4)
    def test_fsdp_hybrid_shard_parity(self):
        self.run_subtests(
            {
                "hsdp_sharding_strategy": [
                    ShardingStrategy.HYBRID_SHARD,
                    ShardingStrategy._HYBRID_SHARD_ZERO2,
                ],
                "use_orig_params": [False, True],
            },
            self._test_fsdp_hybrid_shard_parity,
        )

    def _test_fsdp_hybrid_shard_parity(
        self, hsdp_sharding_strategy: ShardingStrategy, use_orig_params: bool
    ):
        fsdp_model = self._init_fsdp_model(use_orig_params)
        global_pg = dist.distributed_c10d._get_default_group()
        hsdp_pgs = _init_intra_and_inter_node_groups(global_pg, 2)
        hsdp_model = self._init_hsdp_model(
            hsdp_sharding_strategy,
            ShardingStrategyMode.ALL_HYBRID_SHARD,
            use_orig_params,
            hsdp_process_groups=hsdp_pgs,
        )
        assert (
            hsdp_model._inter_node_pg.size() > 1
        ), "HSDP model initialized without replication"
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2)
        hsdp_optim = torch.optim.Adam(hsdp_model.parameters(), lr=1e-2)
        torch.manual_seed(global_pg.rank() + 1)
        for _ in range(5):
            inp = fsdp_model.module.get_input(torch.device("cuda"))
            losses: List[torch.Tensor] = []
            for model, optim in ((fsdp_model, fsdp_optim), (hsdp_model, hsdp_optim)):
                optim.zero_grad()
                loss = model(*inp).sum()
                losses.append(loss)
                loss.backward()
                optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(4)
    def test_fsdp_hybrid_shard_accumulation(
        self,
    ):
        fsdp_model = self._init_fsdp_model(use_orig_params=True)
        global_pg = dist.distributed_c10d._get_default_group()
        hsdp_pgs = _init_intra_and_inter_node_groups(global_pg, 2)
        hsdp_model = self._init_hsdp_model(
            ShardingStrategy.HYBRID_SHARD,
            ShardingStrategyMode.ALL_HYBRID_SHARD,
            use_orig_params=True,
            hsdp_process_groups=hsdp_pgs,
        )
        assert (
            hsdp_model._inter_node_pg.size() > 1
        ), "HSDP model initialized without replication"
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2)
        hsdp_optim = torch.optim.Adam(hsdp_model.parameters(), lr=1e-2)
        torch.manual_seed(global_pg.rank() + 1)

        inp = fsdp_model.module.get_input(torch.device("cuda"))

        with fsdp_model.no_sync():
            for _ in range(3):
                loss = fsdp_model(*inp).sum()
                loss.backward()
        loss = fsdp_model(*inp).sum()
        loss.backward()
        fsdp_optim.step()
        fsdp_loss = fsdp_model(*inp).sum()

        with hsdp_model.no_sync(disable_all_reduce_only=True):
            for _ in range(3):
                loss = hsdp_model(*inp).sum()
                loss.backward()
        loss = hsdp_model(*inp).sum()
        loss.backward()
        hsdp_optim.step()
        hsdp_loss = hsdp_model(*inp).sum()

        self.assertEqual(fsdp_loss, hsdp_loss)

    def _init_fsdp_model(self, use_orig_params: bool) -> nn.Module:
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer},
        )
        hsdp_kwargs = {
            "auto_wrap_policy": auto_wrap_policy,
            "device_id": torch.cuda.current_device(),
            "use_orig_params": use_orig_params,
        }
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            hsdp_kwargs,
            deterministic=True,
        )
        return fsdp_model

    def _init_hsdp_model(
        self,
        hsdp_sharding_strategy: ShardingStrategy,
        sharding_strategy_mode: str,
        use_orig_params: bool,
        hsdp_process_groups: Optional[
            Tuple[dist.ProcessGroup, dist.ProcessGroup]
        ] = None,
    ):
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer},
        )
        hsdp_kwargs = {
            "device_id": torch.cuda.current_device(),
            "auto_wrap_policy": auto_wrap_policy,
            "sharding_strategy": hsdp_sharding_strategy,
            "use_orig_params": use_orig_params,
        }
        if sharding_strategy_mode == ShardingStrategyMode.ALL_HYBRID_SHARD:
            hsdp_model = TransformerWithSharedParams.init(
                hsdp_process_groups or self.process_group,
                FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_BEFORE,
                hsdp_kwargs,
                deterministic=True,
            )
        elif sharding_strategy_mode == ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD:
            model = TransformerWithSharedParams.init(
                hsdp_process_groups or self.process_group,
                FSDPInitMode.NO_FSDP,
                CUDAInitMode.CUDA_BEFORE,
                {},
                deterministic=True,
            )
            # Use the HSDP strategy for the transformer module
            model.transformer = FSDP(model.transformer, **hsdp_kwargs)
            # Use `FULL_SHARD` for the embedding and output projection
            hsdp_model = FSDP(
                model,
                device_id=torch.cuda.current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                use_orig_params=use_orig_params,
            )
        return hsdp_model


instantiate_parametrized_tests(TestFSDPHybridShard)

if __name__ == "__main__":
    run_tests()
