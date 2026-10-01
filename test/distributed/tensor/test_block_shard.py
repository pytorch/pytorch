# Owner(s): ["oncall: distributed"]

import copy
import pickle

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import BlockShard
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorContinuousTestBase,
)


funcol = torch.ops.c10d_functional

# (E, O, I) expert-weight shapes over 4 ranks: aligned, L-shaped (rank rows
# cross expert boundaries), and uneven with an empty last rank.
SHAPES = [(8, 6, 3), (3, 4, 5), (2, 3, 2)]


class TestBlockShardPlacement(TestCase):
    def test_validation(self):
        for numels, repeats in [
            ((), None),
            ((1, -1), None),
            ((1,), (0,)),
            ((0,), None),
            ((1, 2), (1,)),
            ((True,), None),
        ]:
            with self.assertRaises(ValueError):
                BlockShard(numels, repeats)

    def test_canonicalization(self):
        self.assertEqual(BlockShard((4, 4), (1, 1)), BlockShard((4,)))
        self.assertEqual(BlockShard((4,), (2,)), BlockShard((4,)))
        self.assertEqual(hash(BlockShard((4, 4))), hash(BlockShard((4,))))
        self.assertEqual(BlockShard((2, 3, 2, 3)), BlockShard((2, 3)))
        p = BlockShard((0, 5, 0), (3, 1, 4))
        self.assertEqual(p.block_numels, (0, 5, 0))
        self.assertEqual(p.block_repeats, (3, 1, 4))
        self.assertNotEqual(BlockShard((4,)), BlockShard((8,)))

    def test_placement_class(self):
        p = BlockShard((7,))
        self.assertIsInstance(p, Placement)
        self.assertNotIsInstance(p, Shard)
        self.assertFalse(p.is_shard())
        self.assertFalse(p.is_replicate())
        self.assertFalse(p.is_partial())
        self.assertTrue(p.is_block_shard())
        self.assertEqual(pickle.loads(pickle.dumps(p)), p)
        self.assertEqual(copy.deepcopy(p), p)
        self.assertEqual(repr(p), "BlockShard(block_numels=(7,))")

    def test_split_dim(self):
        self.assertEqual(BlockShard((5,))._split_dim((3, 4, 5)), 2)
        self.assertEqual(BlockShard((1,))._split_dim((3, 4, 5)), 3)
        self.assertEqual(BlockShard((5,))._merged_shape((3, 4, 5)), (12, 5))
        self.assertEqual(BlockShard.split_leading((3, 4, 5), 2), BlockShard((5,)))
        # k == 1 is Shard(0), including degenerate shapes whose smallest match is 1.
        for numels, shape in [((20,), (3, 4, 5)), ((5,), (3, 1, 5))]:
            with self.assertRaisesRegex(ValueError, "use Shard"):
                BlockShard(numels)._split_dim(shape)
        with self.assertRaisesRegex(ValueError, "use Shard"):
            BlockShard.split_leading((3, 4, 5), 1)
        for p in [BlockShard((3,)), BlockShard((10,)), BlockShard((5, 4))]:
            with self.assertRaises(NotImplementedError):
                p._split_dim((3, 4, 5))

    def test_local_boxes(self):
        p = BlockShard((5,))
        # 12 rows over 2 ranks: rank 0 holds rows 0-5 (expert 0 and 2 rows of 1).
        self.assertEqual(
            p._local_boxes((3, 4, 5), 2, 0),
            [((0, 0, 0), (1, 4, 5), 0, 4), ((1, 0, 0), (1, 2, 5), 4, 6)],
        )
        self.assertEqual(
            p._local_boxes((3, 4, 5), 2, 1),
            [((1, 2, 0), (1, 2, 5), 0, 2), ((2, 0, 0), (1, 4, 5), 2, 6)],
        )
        # Empty last rank.
        self.assertEqual(BlockShard((2,))._local_boxes((2, 3, 2), 4, 3), [])
        # Boxes cover every rank's rows exactly once.
        for shape in SHAPES:
            p = BlockShard((shape[2],))
            covered = torch.zeros(shape[:2], dtype=torch.int)
            for rank in range(4):
                rows = p._local_shape(shape, 4, rank)[0]
                boxes = p._local_boxes(shape, 4, rank)
                self.assertEqual(sum(stop - start for *_, start, stop in boxes), rows)
                for offset, size, _, _ in boxes:
                    covered[
                        offset[0] : offset[0] + size[0], offset[1] : offset[1] + size[1]
                    ] += 1
            self.assertTrue(torch.all(covered == 1))


class BlockShardDTensorTest(DTensorContinuousTestBase):
    world_size = 4

    def test_distribute_and_full_tensor(self):
        mesh = self.build_device_mesh()
        for shape in SHAPES:
            x = torch.randn(shape, device=self.device_type)
            p = BlockShard((shape[2],))
            d = distribute_tensor(x, mesh, [p])
            rows = p._local_shape(shape, self.world_size, self.rank)[0]
            self.assertEqual(d.to_local().shape, (rows, shape[2]))
            self.assertEqual(d.full_tensor(), x)
            # Rank r owns merged rows [r * q, r * q + rows).
            q = -(-shape[0] * shape[1] // self.world_size)
            start = self.rank * q
            self.assertEqual(d.to_local(), x.view(-1, shape[2])[start : start + rows])

    def test_redistribute(self):
        mesh = self.build_device_mesh()
        for shape in SHAPES:
            x = torch.randn(shape, device=self.device_type)
            p = BlockShard((shape[2],))
            d = distribute_tensor(x, mesh, [p])
            for target in [[Replicate()], [Shard(0)], [Shard(1)], [Shard(2)]]:
                r = d.redistribute(mesh, target)
                self.assertEqual(r.full_tensor(), x)
                back = r.redistribute(mesh, [p])
                self.assertEqual(back.placements, (p,))
                self.assertEqual(back.to_local(), d.to_local())

    def test_partial_to_block_shard_is_reduce_scatter(self):
        mesh = self.build_device_mesh()
        shape = (3, 4, 5)
        x = torch.randn(shape, device=self.device_type)
        partial = DTensor.from_local(x, mesh, [Partial()])
        comm_mode = CommDebugMode()
        with comm_mode:
            out = partial.redistribute(mesh, [BlockShard((5,))])
        self.assertEqual(comm_mode.get_comm_counts()[funcol.reduce_scatter_tensor], 1)
        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertEqual(out.full_tensor(), x * self.world_size)

    def test_redistribute_backward(self):
        mesh = self.build_device_mesh()
        shape = (3, 4, 5)
        x = torch.randn(shape, device=self.device_type, requires_grad=True)
        d = distribute_tensor(x, mesh, [BlockShard((5,))])
        grad = torch.full(shape, 2.0, device=self.device_type)
        d.redistribute(mesh, [Replicate()]).backward(
            distribute_tensor(grad, mesh, [Replicate()])
        )
        self.assertEqual(d.grad.placements, (BlockShard((5,)),))
        self.assertEqual(d.grad.full_tensor(), grad)

    def test_from_local_and_to_local(self):
        mesh = self.build_device_mesh()
        shape = (3, 4, 5)
        x = torch.randn(shape, device=self.device_type)
        p = BlockShard((5,))
        local = distribute_tensor(x, mesh, [p]).to_local()
        with self.assertRaisesRegex(ValueError, "requires shape="):
            DTensor.from_local(local, mesh, [p])
        d = DTensor.from_local(local, mesh, [p], shape=x.shape)
        self.assertEqual(d.full_tensor(), x)

        local = local.clone().requires_grad_(True)
        d = DTensor.from_local(local, mesh, [p], shape=x.shape)
        (d.to_local() * 3).sum().backward()
        self.assertEqual(local.grad, torch.full_like(local, 3))

    def test_hsdp_mesh(self):
        mesh = init_device_mesh(self.device_type, (2, 2))
        shape = (3, 4, 5)
        x = torch.randn(shape, device=self.device_type)
        p = BlockShard((5,))
        d = distribute_tensor(x, mesh, [Replicate(), p])
        self.assertEqual(d.full_tensor(), x)
        partial = DTensor.from_local(d.to_local(), mesh, [Partial(), p], shape=x.shape)
        reduced = partial.redistribute(mesh, [Replicate(), p])
        self.assertEqual(reduced.placements, (Replicate(), p))
        self.assertEqual(reduced.to_local(), d.to_local() * 2)
        with self.assertRaises(NotImplementedError):
            distribute_tensor(x, mesh, [Shard(2), p])

    def test_unsupported_patterns(self):
        mesh = self.build_device_mesh()
        x = torch.randn(3, 4, 5, device=self.device_type)
        with self.assertRaisesRegex(ValueError, "use Shard"):
            distribute_tensor(x, mesh, [BlockShard((20,))])
        with self.assertRaises(NotImplementedError):
            distribute_tensor(x, mesh, [BlockShard((3,))])


if __name__ == "__main__":
    run_tests()
