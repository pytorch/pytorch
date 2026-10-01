# Owner(s): ["oncall: distributed"]

import copy
import pickle

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed._state_dict_utils import _distribute_tensors
from torch.distributed.checkpoint.planner_helpers import (
    _create_default_metadata_only_plan,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.tensor._dtensor_spec import (
    _lower_block_shard_spec,
    DTensorSpec,
    TensorMeta,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import (
    _validate_block_shard_placements,
    BlockShard,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorContinuousTestBase,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


# (E, O, I) expert-weight shapes over 4 ranks: aligned, rows crossing expert
# boundaries, and uneven with an empty last rank.
SHAPES = [(8, 6, 3), (3, 4, 5), (2, 3, 2)]
# Shapes for expert parallelism over 2 ranks: each EP rank's rows cross expert
# boundaries, or are uneven over the 2 FSDP ranks.
EP_SHAPES = [(6, 2, 3), (2, 3, 4), (4, 5, 2)]


def block_shard_tensor(x, mesh, placements):
    """Build a BlockShard DTensor from the global tensor ``x``.

    The local tensor comes from splitting the merged view with the Shard(0)
    placements BlockShard lowers to. ``x`` must be the same on every rank.
    """
    spec = DTensorSpec(
        mesh, tuple(placements), tensor_meta=TensorMeta(x.shape, x.stride(), x.dtype)
    )
    lowered = _lower_block_shard_spec(spec)
    local = distribute_tensor(
        x.reshape(lowered.shape), mesh, lowered.placements, src_data_rank=None
    ).to_local()
    return DTensor.from_local(local, mesh, placements, shape=x.shape, stride=x.stride())


class TestBlockShardPlacement(TestCase):
    def test_validation_and_class(self):
        for numels, repeats in [
            ((), None),
            ((1, -1), None),
            ((True,), None),
            ((1,), (0,)),
            ((1, 2), (1,)),
        ]:
            with self.assertRaises(ValueError):
                BlockShard(numels, repeats)
        p = BlockShard((7,))
        self.assertIsInstance(p, Placement)
        self.assertNotIsInstance(p, Shard)
        self.assertFalse(p.is_shard())
        self.assertEqual(pickle.loads(pickle.dumps(p)), p)
        self.assertEqual(copy.deepcopy(p), p)
        self.assertEqual(hash(p), hash(BlockShard((7,))))
        self.assertEqual(repr(p), "BlockShard(block_numels=(7,))")

    def test_split_dim(self):
        self.assertEqual(BlockShard((5,))._split_dim((3, 4, 5)), 2)
        self.assertEqual(BlockShard((5,))._merged_shape((3, 4, 5)), (12, 5))
        self.assertEqual(BlockShard.split_leading((3, 4, 5), 2), BlockShard((5,)))
        # k == 1 is Shard(0), including degenerate shapes whose smallest match is 1.
        for numels, shape in [((20,), (3, 4, 5)), ((5,), (3, 1, 5))]:
            with self.assertRaisesRegex(ValueError, "use Shard"):
                BlockShard(numels)._split_dim(shape)
        with self.assertRaisesRegex(ValueError, "use Shard"):
            BlockShard.split_leading((3, 4, 5), 1)
        for p in [BlockShard((3,)), BlockShard((5, 5)), BlockShard((5,), (2,))]:
            with self.assertRaises(NotImplementedError):
                p._split_dim((3, 4, 5))

    def test_layout_validation(self):
        p = BlockShard((5,))
        layout = _validate_block_shard_placements([p, Shard(0)], (4, 4, 5), (2, 2))
        self.assertEqual((layout.mesh_dim, layout.shard0_mesh_dim), (0, 1))
        self.assertEqual(layout.block_shape, (2, 4, 5))
        self.assertIsNone(_validate_block_shard_placements([Shard(0)], (4, 5), (2,)))
        for placements, shape in [
            ([p, Shard(0)], (3, 4, 5)),  # 3 experts don't divide over 2 ranks
            ([p, Shard(1)], (4, 4, 5)),
            ([p, p], (4, 4, 5)),
        ]:
            with self.assertRaises(NotImplementedError):
                _validate_block_shard_placements(placements, shape, (2, 2))


class BlockShardDTensorTest(DTensorContinuousTestBase):
    world_size = 4

    def setUp(self):
        super().setUp()
        # block_shard_tensor needs the same global tensors on every rank.
        torch.manual_seed(0)

    def _layouts(self):
        """(mesh, placements for shape) for 1D, HSDP, and EP layouts."""
        mesh_1d = self.build_device_mesh()
        yield mesh_1d, lambda s: (BlockShard((s[2],)),), SHAPES
        mesh_2d = init_device_mesh(self.device_type, (2, 2))
        yield mesh_2d, lambda s: (Replicate(), BlockShard((s[2],))), SHAPES
        for names in [("efsdp", "ep"), ("ep", "efsdp")]:
            mesh = init_device_mesh(self.device_type, (2, 2), mesh_dim_names=names)
            b, e = names.index("efsdp"), names.index("ep")

            def ep_placements(s, b=b, e=e):
                placements: list[Placement] = [Replicate(), Replicate()]
                placements[b], placements[e] = BlockShard((s[2],)), Shard(0)
                return tuple(placements)

            yield mesh, ep_placements, EP_SHAPES

    def test_local_tensor_and_full_tensor(self):
        mesh = self.build_device_mesh()
        for shape in SHAPES:
            x = torch.randn(shape, device=self.device_type)
            d = block_shard_tensor(x, mesh, [BlockShard((shape[2],))])
            # Rank r owns merged rows [r * q, (r + 1) * q).
            q = -(-shape[0] * shape[1] // self.world_size)
            start = self.rank * q
            self.assertEqual(d.to_local(), x.view(-1, shape[2])[start : start + q])
        for mesh, placements_fn, shapes in self._layouts():
            for shape in shapes:
                x = torch.randn(shape, device=self.device_type)
                d = block_shard_tensor(x, mesh, placements_fn(shape))
                self.assertEqual(d.full_tensor(), x)
                for target in [Shard(1), Shard(2)]:
                    out = d.redistribute(
                        mesh, [target] + [Replicate()] * (mesh.ndim - 1)
                    )
                    self.assertEqual(out.full_tensor(), x)
                replicated = d.redistribute(mesh, [Replicate()] * mesh.ndim)
                with self.assertRaises(NotImplementedError):
                    replicated.redistribute(mesh, placements_fn(shape))

    def test_elementwise_ops(self):
        for mesh, placements_fn, shapes in self._layouts():
            for shape in shapes:
                placements = placements_fn(shape)
                x, y = (torch.randn(shape, device=self.device_type) for _ in range(2))
                dx = block_shard_tensor(x, mesh, placements)
                dy = block_shard_tensor(y, mesh, placements)
                comm_mode = CommDebugMode()
                with comm_mode:
                    out = dx * 2 + dy
                    torch._foreach_mul_([out], 0.5)
                    copied = torch.zeros_like(dx).copy_(out).detach().clone()
                    casted = copied.to(torch.float64)
                self.assertEqual(comm_mode.get_total_counts(), 0)
                for t in (out, copied, casted):
                    self.assertEqual(t.placements, placements)
                self.assertEqual(casted.full_tensor(), (x + 0.5 * y).double())
                self.assertEqual(torch._foreach_norm([dx])[0].full_tensor(), x.norm())
                self.assertEqual(torch.linalg.vector_norm(dx).full_tensor(), x.norm())

    def test_unsupported_ops_raise(self):
        mesh = self.build_device_mesh()
        x = torch.randn(3, 4, 5, device=self.device_type)
        d = block_shard_tensor(x, mesh, [BlockShard((5,))])
        rep = distribute_tensor(x, mesh, [Replicate()])
        for fn in [
            lambda: d.sum(dim=0),
            lambda: d.transpose(0, 1),
            lambda: d.view(12, 5),
            lambda: d + rep,
            lambda: torch.linalg.vector_norm(d, dim=2),
        ]:
            with self.assertRaises(NotImplementedError):
                fn()

    def test_optimizer_and_grad_clipping(self):
        mesh = self.build_device_mesh()
        torch.manual_seed(0)
        shapes_and_placements = [
            ((3, 4, 6), [BlockShard((6,))]),
            ((2, 3, 2), [BlockShard((2,))]),
            ((8, 5), [Shard(0)]),
        ]
        ref_params = [
            torch.nn.Parameter(torch.randn(shape, device=self.device_type))
            for shape, _ in shapes_and_placements
        ]
        params = [
            torch.nn.Parameter(block_shard_tensor(p.detach().clone(), mesh, plc))
            for p, (_, plc) in zip(ref_params, shapes_and_placements)
        ]
        ref_opt = torch.optim.AdamW(ref_params, lr=1e-2, foreach=True)
        opt = torch.optim.AdamW(params, lr=1e-2, foreach=True)
        for _ in range(3):
            for ref, param, (_, plc) in zip(ref_params, params, shapes_and_placements):
                grad = torch.randn_like(ref)
                ref.grad = grad
                param.grad = block_shard_tensor(grad, mesh, plc)
            ref_norm = torch.nn.utils.clip_grad_norm_(ref_params, max_norm=1.0)
            norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            self.assertEqual(norm.full_tensor(), ref_norm)
            ref_opt.step()
            opt.step()
            for ref, param, (_, plc) in zip(ref_params, params, shapes_and_placements):
                self.assertEqual(param.placements, tuple(plc))
                self.assertEqual(param.full_tensor(), ref)

    def test_random_init_matches_merged_shard(self):
        # Random init of a BlockShard tensor draws the same values per element
        # as Shard(0) on its merged view, so ranks don't repeat values.
        for mesh, placements_fn, shapes in self._layouts():
            for shape in shapes:
                placements = placements_fn(shape)
                d = block_shard_tensor(
                    torch.zeros(shape, device=self.device_type), mesh, placements
                )
                spec = d._spec
                lowered = _lower_block_shard_spec(spec)
                merged = distribute_tensor(
                    torch.zeros(lowered.shape, device=self.device_type),
                    mesh,
                    lowered.placements,
                    src_data_rank=None,
                )
                torch.manual_seed(5)
                torch.nn.init.trunc_normal_(d)
                torch.manual_seed(5)
                torch.nn.init.trunc_normal_(merged)
                self.assertEqual(d.placements, placements)
                self.assertEqual(d.to_local(), merged.to_local())
                self.assertEqual(
                    d.full_tensor().view(lowered.shape), merged.full_tensor()
                )

    def test_compile(self):
        mesh = self.build_device_mesh()
        x = torch.randn(3, 4, 6, device=self.device_type)
        d = block_shard_tensor(x, mesh, [BlockShard((6,))])

        def fn(t):
            return torch.linalg.vector_norm(t * 2 + 1)

        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(d)
        self.assertEqual(out.full_tensor(), fn(x))
        for copied in (copy.deepcopy(d), pickle.loads(pickle.dumps(d))):
            self.assertEqual(copied.placements, d.placements)
            self.assertEqual(copied.to_local(), d.to_local())

    def test_partial_operand_raises(self):
        mesh = self.build_device_mesh()
        x = torch.randn(3, 4, 5, device=self.device_type)
        d = block_shard_tensor(x, mesh, [BlockShard((5,))])
        partial = DTensor.from_local(x.clone(), mesh, [Partial()])
        with self.assertRaises(NotImplementedError):
            d + partial


class TestBlockShardBoxes(TestCase):
    def test_local_boxes(self):
        p = BlockShard((5,))
        # 12 rows over 2 ranks: rank 0 holds expert 0 and 2 rows of expert 1.
        self.assertEqual(
            p._local_boxes((3, 4, 5), 2, 0),
            [((0, 0, 0), (1, 4, 5), 0, 4), ((1, 0, 0), (1, 2, 5), 4, 6)],
        )
        self.assertEqual(
            p._local_boxes((3, 4, 5), 2, 1),
            [((1, 2, 0), (1, 2, 5), 0, 2), ((2, 0, 0), (1, 4, 5), 2, 6)],
        )
        self.assertEqual(BlockShard((2,))._local_boxes((2, 3, 2), 4, 3), [])
        for shape in SHAPES:
            p = BlockShard((shape[2],))
            covered = torch.zeros(shape[:2], dtype=torch.int)
            for rank in range(4):
                for offset, size, start, stop in p._local_boxes(shape, 4, rank):
                    self.assertEqual(stop - start, size[0] * size[1])
                    covered[
                        offset[0] : offset[0] + size[0], offset[1] : offset[1] + size[1]
                    ] += 1
            self.assertTrue(torch.all(covered == 1))


class BlockShardCheckpointTest(DTensorContinuousTestBase):
    world_size = 4

    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def _ep_mesh(self):
        return init_device_mesh(
            self.device_type, (2, 2), mesh_dim_names=("efsdp", "ep")
        )

    def test_chunk_list_and_write_items(self):
        for mesh, placements, shape in [
            (self.build_device_mesh(), [BlockShard((5,))], (3, 4, 5)),
            (self._ep_mesh(), [BlockShard((3,)), Shard(0)], (6, 2, 3)),
        ]:
            x = torch.randn(shape, device=self.device_type)
            d = block_shard_tensor(x, mesh, placements)
            boxes = d._block_shard_boxes()
            chunks = d.__create_chunk_list__()
            self.assertEqual(
                [(tuple(c.offsets), tuple(c.sizes)) for c in chunks],
                [(offset, size) for offset, size, _, _ in boxes],
            )
            items = _create_default_metadata_only_plan({"w": d}).items
            self.assertEqual(len(items), len(boxes))
            for item, (offset, size, start, stop) in zip(items, boxes):
                self.assertEqual(item.tensor_data.size, torch.Size(shape))
                shard = d.__get_tensor_shard__(item.index)
                self.assertEqual(shard.data_ptr(), d.to_local()[start:stop].data_ptr())
                self.assertEqual(
                    shard, x[tuple(slice(o, o + n) for o, n in zip(offset, size))]
                )

    @with_temp_dir
    def test_save_load_reshard(self):
        mesh = self.build_device_mesh()
        mesh_2d = init_device_mesh(self.device_type, (2, 2))
        ep_mesh = self._ep_mesh()
        full = {
            "w": torch.randn(3, 4, 5, device=self.device_type),
            "e": torch.randn(6, 2, 3, device=self.device_type),
        }
        saved = {
            "w": block_shard_tensor(full["w"], mesh, [BlockShard((5,))]),
            "e": block_shard_tensor(full["e"], ep_mesh, [BlockShard((3,)), Shard(0)]),
        }
        dcp.save(saved, checkpoint_id=self.temp_dir)

        def zeros_like(t, mesh, placements):
            return block_shard_tensor(torch.zeros_like(t), mesh, placements)

        targets = {
            "block_shard": lambda t: zeros_like(t, mesh, [BlockShard((t.shape[2],))]),
            "hsdp_block_shard": lambda t: zeros_like(
                t, mesh_2d, [Replicate(), BlockShard((t.shape[2],))]
            ),
            "ep_block_shard": lambda t: zeros_like(
                t, ep_mesh, [BlockShard((t.shape[2],)), Shard(0)]
            )
            if t.shape[0] % 2 == 0
            else zeros_like(t, mesh, [BlockShard((t.shape[2],))]),
            "replicate": lambda t: distribute_tensor(
                torch.zeros_like(t), mesh, [Replicate()]
            ),
            "shard_1": lambda t: distribute_tensor(
                torch.zeros_like(t), mesh, [Shard(1)]
            ),
        }
        for target_name, make in targets.items():
            loaded = {name: make(t) for name, t in full.items()}
            dcp.load(loaded, checkpoint_id=self.temp_dir)
            for name, t in full.items():
                self.assertEqual(loaded[name].full_tensor(), t, msg=target_name)

    @with_temp_dir
    def test_load_into_block_shard(self):
        mesh = self.build_device_mesh()
        full = torch.randn(3, 4, 5, device=self.device_type)
        dcp.save(
            {"w": distribute_tensor(full, mesh, [Shard(1)])},
            checkpoint_id=self.temp_dir,
        )
        loaded = {
            "w": block_shard_tensor(torch.zeros_like(full), mesh, [BlockShard((5,))])
        }
        dcp.load(loaded, checkpoint_id=self.temp_dir)
        self.assertEqual(loaded["w"].placements, (BlockShard((5,)),))
        self.assertEqual(loaded["w"].full_tensor(), full)

    def test_distribute_tensors(self):
        # set_model_state_dict(..., broadcast_from_rank0=True) fills sharded
        # params from full tensors with _distribute_tensors.
        for mesh, placements, shape in [
            (self.build_device_mesh(), [BlockShard((5,))], (3, 4, 5)),
            (self._ep_mesh(), [BlockShard((3,)), Shard(0)], (6, 2, 3)),
        ]:
            full = torch.randn(shape, device=self.device_type)
            local_state = block_shard_tensor(torch.zeros_like(full), mesh, placements)
            state_dict = {"w": (local_state, full)}
            _distribute_tensors(state_dict, ["w"], torch.device(self.device_type))
            self.assertEqual(state_dict["w"].full_tensor(), full)


if __name__ == "__main__":
    run_tests()
