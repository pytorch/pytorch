# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
import torch.distributed.fsdp  # noqa: F401 -- must init before fsdp submodule
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard
from torch.distributed.tensor.parallel.fsdp import _chunk_tensor, _get_box_for
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class TestGetBoxForUnevenShards(DTensorTestBase):
    """_get_box_for must follow torch.chunk (ceil-division) semantics.

    Per-rank shard offsets and sizes must match DTensor's internal Shard
    placement, which uses torch.chunk. This test suite verifies correctness
    for various global sizes and world sizes.
    """

    @property
    def world_size(self) -> int:
        return 3

    @with_comms
    @parametrize(
        "global_size,expected_offsets,expected_sizes",
        [
            # uneven: 5 / 3 -> [2, 2, 1]
            (5, [0, 2, 4], [2, 2, 1]),
            # ceil vs floor diverge: 7 / 3 -> [3, 3, 1] (ceil), not [3, 2, 2]
            (7, [0, 3, 6], [3, 3, 1]),
            # ceil vs floor diverge: 10 / 3 -> [4, 4, 2] (ceil), not [4, 3, 3]
            (10, [0, 4, 8], [4, 4, 2]),
            # even (regression): 6 / 3 -> [2, 2, 2]
            (6, [0, 2, 4], [2, 2, 2]),
            # global < world: 1 / 3 -> [1, 0, 0]
            (1, [0, 1, 1], [1, 0, 0]),
        ],
    )
    def test_get_box_for_shard_offsets(
        self, global_size, expected_offsets, expected_sizes
    ):
        mesh = init_device_mesh("cpu", (self.world_size,))
        dt = distribute_tensor(
            torch.arange(global_size, dtype=torch.float32), mesh, [Shard(0)]
        )

        for idx in range(self.world_size):
            offsets, sizes = _get_box_for(dt, idx)
            self.assertEqual(offsets[0], expected_offsets[idx])
            self.assertEqual(sizes[0], expected_sizes[idx])

    @with_comms
    def test_get_box_for_replicated_placement(self):
        """Non-shard placement: offsets are zero, sizes are the full global size."""
        mesh = init_device_mesh("cpu", (self.world_size,))
        dt = distribute_tensor(
            torch.arange(5, dtype=torch.float32), mesh, [Replicate()]
        )

        for idx in range(self.world_size):
            offsets, sizes = _get_box_for(dt, idx)
            self.assertEqual(offsets[0], 0)
            self.assertEqual(sizes[0], 5)

    @with_comms
    def test_get_box_for_multidim_non_zero_shard_dim(self):
        """2D tensor sharded on dim=1: dim 0 unchanged, dim 1 chunked."""
        mesh = init_device_mesh("cpu", (self.world_size,))

        # Shape (4, 7), sharded on dim=1 -> torch.chunk(7, 3) -> [3, 3, 1]
        t = torch.randn(4, 7)
        dt = distribute_tensor(t, mesh, [Shard(1)])

        expected_offsets_dim1 = [0, 3, 6]
        expected_sizes_dim1 = [3, 3, 1]

        for idx in range(self.world_size):
            offsets, sizes = _get_box_for(dt, idx)
            # dim 0 is not sharded
            self.assertEqual(offsets[0], 0)
            self.assertEqual(sizes[0], 4)
            # dim 1 is sharded
            self.assertEqual(offsets[1], expected_offsets_dim1[idx])
            self.assertEqual(sizes[1], expected_sizes_dim1[idx])

    @with_comms
    def test_get_box_for_matches_local_tensor_size(self):
        """_get_box_for(dt, rank) shard size must match dt._local_tensor.size().

        This cross-checks the computed metadata against the actual DTensor
        local data, independent of the chunk formula.
        """
        mesh = init_device_mesh("cpu", (self.world_size,))
        dt = distribute_tensor(torch.arange(7, dtype=torch.float32), mesh, [Shard(0)])

        _, sizes = _get_box_for(dt, self.rank)
        self.assertEqual(sizes, dt._local_tensor.size())

    @with_comms
    def test_chunk_tensor_uneven_shard_no_overlap_error(self):
        """_chunk_tensor must not raise ValueError for uneven DTensor shards.

        Before the fix, floor-division offsets produced overlapping
        ShardMetadata, causing validate_non_overlapping_shards_metadata
        to raise ValueError from
        ShardedTensor._init_from_local_shards_and_global_metadata.
        """
        mesh = init_device_mesh("cpu", (self.world_size,))
        dt = distribute_tensor(torch.arange(7, dtype=torch.float32), mesh, [Shard(0)])
        pg = dist.group.WORLD

        # This must not raise ValueError: Shards ... overlap
        st = _chunk_tensor(dt, self.rank, self.world_size, 1, pg)
        self.assertIsInstance(st, ShardedTensor)


instantiate_parametrized_tests(TestGetBoxForUnevenShards)

if __name__ == "__main__":
    run_tests()
