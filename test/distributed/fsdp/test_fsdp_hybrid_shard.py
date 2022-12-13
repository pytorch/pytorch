# Owner(s): ["oncall: distributed"]

import contextlib
import sys
from collections import Counter
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.distributed_c10d import _rank_not_in_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
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
    def test_hybrid_shard_strategy_mismatch_raises(self):
        for sharding_strategy in [
            ShardingStrategy._HYBRID_SHARD_ZERO2,
            ShardingStrategy.HYBRID_SHARD,
        ]:
            with self.subTest(sharding_strategy=sharding_strategy):
                model = MyModel().cuda()
                intra_pg = self.process_group
                inter_pg = dist.new_group(ranks=[self.rank])
                model.lin1 = FSDP(
                    model.lin1,
                    process_group=(intra_pg, inter_pg),
                    sharding_strategy=sharding_strategy,
                )
                self.assertEqual(model.lin1.process_group, intra_pg)
                self.assertEqual(model.lin1._inter_node_pg, inter_pg)
                model = FSDP(model, process_group=intra_pg)
                inp = torch.randn(4, 10)
                # Errors during _lazy_init
                with self.assertRaisesRegex(
                    ValueError, "expect sharding strategies to be the same"
                ):
                    model(inp)

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
        for sharding_strategy in [
            ShardingStrategy.HYBRID_SHARD,
            ShardingStrategy._HYBRID_SHARD_ZERO2,
        ]:
            with self.subTest(sharding_strategy=sharding_strategy):
                auto_wrap_policy = ModuleWrapPolicy(
                    {TransformerEncoderLayer, TransformerDecoderLayer},
                )
                fsdp_kwargs = {
                    "auto_wrap_policy": auto_wrap_policy,
                    "device_id": torch.cuda.current_device(),
                    "sharding_strategy": sharding_strategy,
                }
                fsdp_model = TransformerWithSharedParams.init(
                    self.process_group,
                    FSDPInitMode.RECURSIVE,
                    CUDAInitMode.CUDA_BEFORE,
                    fsdp_kwargs,
                )
                # All FSDP modules should have state.process_group as the process group over which to
                # shard (default process group), and state._inter_node_pg (process group containing only
                # this rank)
                intra_node_pgs = set()
                inter_node_pgs = set()
                for mod in fsdp_model.fsdp_modules(fsdp_model):
                    # process_group should be across the node, which is just the
                    # whole world here.
                    self.assertEqual(
                        dist.get_world_size(mod.process_group),
                        dist.get_world_size(self.process_group),
                    )
                    intra_node_pgs.add(mod.process_group)
                    inter_node_pg = mod._inter_node_pg
                    inter_node_pgs.add(inter_node_pg)
                    self.assertEqual(1, dist.get_world_size(inter_node_pg))
                    self.assertFalse(_rank_not_in_group(inter_node_pg))
                    self.assertEqual(sharding_strategy, mod.sharding_strategy)
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
                with patch_allreduce(patched_allreduce), patch_reduce_scatter(patched_reduce_scatter):
                    inp = fsdp_model.get_input(device=torch.cuda.current_device())
                    out = fsdp_model(inp[0], inp[1])
                    loss = fsdp_model.get_loss(inp, out)
                    loss.backward()

                num_flat_params = len(list(FSDP._fsdp_handles(fsdp_model)))
                self.assertEqual(num_flat_params, cntr[orig_ar])
                self.assertEqual(num_flat_params, cntr[orig_rs])
                dist.barrier()


instantiate_parametrized_tests(TestFSDPHybridShard)

if __name__ == "__main__":
    run_tests()
