# Owner(s): ["oncall: distributed"]

"""Tests for custom Python process groups (PassthroughProcessGroup, _pg_bypass)."""

import os

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests


class CustomProcessGroupTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 4

    def test_pg_bypass_delegates_collective(self):
        """@_pg_bypass forwards dist.all_reduce to pg.all_reduce if defined."""

        calls = []

        class _CollectivePG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "collective_bypass"

            def all_reduce(self, tensor, op=None, async_op=False):
                calls.append(
                    {
                        "tensor_shape": list(tensor.shape),
                        "op": op,
                        "async_op": async_op,
                    }
                )
                return None

        dist.Backend.register_backend(
            "collective_bypass",
            lambda *args, **kwargs: _CollectivePG(
                args[0].group_rank, args[0].group_size
            ),
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "collective_bypass", rank=self.rank, world_size=self.world_size
        )

        try:
            tensor = torch.zeros(4)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["tensor_shape"], [4])
            self.assertEqual(calls[0]["op"], dist.ReduceOp.MAX)
            self.assertFalse(calls[0]["async_op"])
        finally:
            dist.destroy_process_group()

    def test_pg_bypass_falls_through_when_not_overridden(self):
        """@_pg_bypass falls through when PG doesn't override the method."""

        class _MinimalPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "minimal_bypass"

        dist.Backend.register_backend(
            "minimal_bypass",
            lambda *args, **kwargs: _MinimalPG(args[0].group_rank, args[0].group_size),
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "minimal_bypass", rank=self.rank, world_size=self.world_size
        )

        try:
            # all_reduce is not overridden on _MinimalPG, so _pg_bypass
            # falls through to the original dist.all_reduce implementation.
            # _MinimalPG has no C++ backend, so this raises RuntimeError.
            tensor = torch.zeros(4)
            with self.assertRaises(RuntimeError):
                dist.all_reduce(tensor)
        finally:
            dist.destroy_process_group()

    def test_pg_bypass_extracts_group_positionally(self):
        """@_pg_bypass handles group passed as a positional argument."""

        calls = []

        class _PositionalPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "positional_bypass"

            def broadcast(self, tensor, src=None, async_op=False, group_src=None):
                calls.append({"src": src, "async_op": async_op})
                return None

        dist.Backend.register_backend(
            "positional_bypass",
            lambda *args, **kwargs: _PositionalPG(
                args[0].group_rank, args[0].group_size
            ),
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "positional_bypass", rank=self.rank, world_size=self.world_size
        )

        try:
            tensor = torch.zeros(4)
            # Pass group as keyword
            dist.broadcast(tensor, src=0, group=dist.group.WORLD)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["src"], 0)
        finally:
            dist.destroy_process_group()

    def test_getattr_forwards_custom_collective(self):
        """Module __getattr__ forwards unknown functions to the PG."""

        calls = []

        class _CustomPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "custom_collective"

            def my_custom_collective(self, tensor, scale=1.0):
                calls.append({"shape": list(tensor.shape), "scale": scale})
                return None

        dist.Backend.register_backend(
            "custom_collective",
            lambda *args, **kwargs: _CustomPG(args[0].group_rank, args[0].group_size),
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "custom_collective", rank=self.rank, world_size=self.world_size
        )

        try:
            tensor = torch.zeros(8)
            dist.my_custom_collective(tensor, scale=2.0, group=dist.group.WORLD)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["shape"], [8])
            self.assertEqual(calls[0]["scale"], 2.0)
        finally:
            dist.destroy_process_group()

    def test_getattr_raises_for_nonexistent(self):
        """Module __getattr__ raises AttributeError for unknown methods."""

        class _EmptyPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "empty_getattr"

        dist.Backend.register_backend(
            "empty_getattr",
            lambda *args, **kwargs: _EmptyPG(args[0].group_rank, args[0].group_size),
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "empty_getattr", rank=self.rank, world_size=self.world_size
        )

        try:
            tensor = torch.zeros(4)
            with self.assertRaises(AttributeError):
                dist.nonexistent_function(tensor, group=dist.group.WORLD)
        finally:
            dist.destroy_process_group()

    def test_passthrough_delegates_to_inner(self):
        """PassthroughProcessGroup delegates unhandled calls to inner PG."""

        inner_calls = []

        class _InnerPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "inner_terminal"

            def all_reduce(self, tensor, op=None, async_op=False):
                inner_calls.append({"op": "all_reduce"})
                tensor.add_(1)
                return None

            def my_custom_op(self, tensor, scale=1.0):
                inner_calls.append({"op": "my_custom_op", "scale": scale})
                tensor.mul_(scale)
                return None

        outer_calls = []

        class _OuterPG(dist.distributed_c10d.PassthroughProcessGroup):
            def broadcast(self, tensor, src=None, async_op=False, **kwargs):
                outer_calls.append({"op": "broadcast"})
                tensor.fill_(42)
                return None

        def create_inner(dist_opts, pg_options=None):
            return _InnerPG(dist_opts.group_rank, dist_opts.group_size)

        def create_outer(dist_opts, pg_options=None):
            pg = _OuterPG(dist_opts.group_rank, dist_opts.group_size)
            dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
            return pg

        dist.Backend.register_backend(
            "inner_terminal",
            create_inner,
            extended_api=True,
        )
        dist.Backend.register_backend(
            "outer_passthrough",
            create_outer,
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "outer_passthrough(inner_terminal)",
            rank=self.rank,
            world_size=self.world_size,
        )

        try:
            # broadcast is overridden on _OuterPG
            tensor = torch.zeros(4)
            dist.broadcast(tensor, src=0)
            self.assertEqual(len(outer_calls), 1)
            self.assertEqual(outer_calls[0]["op"], "broadcast")
            self.assertEqual(tensor, torch.full((4,), 42.0))

            # all_reduce is NOT on _OuterPG; delegates to _InnerPG
            tensor = torch.zeros(4)
            dist.all_reduce(tensor)
            self.assertEqual(len(inner_calls), 1)
            self.assertEqual(inner_calls[0]["op"], "all_reduce")
            self.assertEqual(tensor, torch.ones(4))

            # custom op defined only on _InnerPG, forwarded via
            # __getattr__ on PassthroughProcessGroup
            tensor = torch.full((4,), 3.0)
            dist.my_custom_op(tensor, scale=2.0, group=dist.group.WORLD)
            self.assertEqual(len(inner_calls), 2)
            self.assertEqual(inner_calls[1]["op"], "my_custom_op")
            self.assertEqual(tensor, torch.full((4,), 6.0))
        finally:
            dist.destroy_process_group()

    def test_passthrough_fallback(self):
        """Passthrough PG can fall back to inner PG."""

        calls = []

        class _InnerPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "fb_inner"

            def all_reduce(self, tensor, op=None, async_op=False):
                calls.append("inner")
                tensor.add_(10)
                return None

        class _OuterPG(dist.distributed_c10d.PassthroughProcessGroup):
            def all_reduce(self, tensor, op=None, async_op=False):
                if tensor.shape[0] > 8:
                    calls.append("outer_custom")
                    tensor.fill_(99)
                    return None
                calls.append("outer_fallback")
                return self._inner_pg.all_reduce(
                    tensor,
                    op=op,
                    async_op=async_op,
                )

        def create_inner(dist_opts, pg_options=None):
            return _InnerPG(dist_opts.group_rank, dist_opts.group_size)

        def create_outer(dist_opts, pg_options=None):
            pg = _OuterPG(dist_opts.group_rank, dist_opts.group_size)
            dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
            return pg

        dist.Backend.register_backend(
            "fb_inner",
            create_inner,
            extended_api=True,
        )
        dist.Backend.register_backend(
            "fb_outer",
            create_outer,
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "fb_outer(fb_inner)",
            rank=self.rank,
            world_size=self.world_size,
        )

        try:
            # Small tensor: outer falls back to inner
            tensor = torch.zeros(4)
            dist.all_reduce(tensor)
            self.assertEqual(calls, ["outer_fallback", "inner"])
            self.assertEqual(tensor, torch.full((4,), 10.0))

            # Large tensor: outer handles it directly
            calls.clear()
            tensor = torch.zeros(16)
            dist.all_reduce(tensor)
            self.assertEqual(calls, ["outer_custom"])
            self.assertEqual(tensor, torch.full((16,), 99.0))
        finally:
            dist.destroy_process_group()

    def test_passthrough_kwargs_forwarding(self):
        """Subset PG with **kwargs forwards extra params to inner PG."""

        calls = []

        class _InnerPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "kw_inner"

            def broadcast(self, tensor, src=None, async_op=False, group_src=None):
                calls.append(
                    {
                        "op": "inner_broadcast",
                        "src": src,
                        "group_src": group_src,
                    }
                )
                tensor.fill_(7)
                return None

        class _OuterPG(dist.distributed_c10d.PassthroughProcessGroup):
            def broadcast(self, tensor, src=None, async_op=False, **kwargs):
                calls.append({"op": "outer_broadcast"})
                return self._inner_pg.broadcast(
                    tensor,
                    src=src,
                    async_op=async_op,
                    **kwargs,
                )

        def create_inner(dist_opts, pg_options=None):
            return _InnerPG(dist_opts.group_rank, dist_opts.group_size)

        def create_outer(dist_opts, pg_options=None):
            pg = _OuterPG(dist_opts.group_rank, dist_opts.group_size)
            dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
            return pg

        dist.Backend.register_backend(
            "kw_inner",
            create_inner,
            extended_api=True,
        )
        dist.Backend.register_backend(
            "kw_outer",
            create_outer,
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "kw_outer(kw_inner)",
            rank=self.rank,
            world_size=self.world_size,
        )

        try:
            tensor = torch.zeros(4)
            dist.broadcast(tensor, src=0, group_src=42)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["op"], "outer_broadcast")
            self.assertEqual(calls[1]["op"], "inner_broadcast")
            self.assertEqual(calls[1]["src"], 0)
            self.assertEqual(calls[1]["group_src"], 42)
            self.assertEqual(tensor, torch.full((4,), 7.0))
        finally:
            dist.destroy_process_group()

    def test_passthrough_pg_options_dict(self):
        """pg_options dict dispatches per-layer options to each creator."""

        received_opts = {}

        class _InnerPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "opts_inner"

        class _OuterPG(dist.distributed_c10d.PassthroughProcessGroup):
            pass

        def create_inner(dist_opts, pg_options=None):
            received_opts["inner"] = pg_options
            received_opts["inner_dist"] = dist_opts.pg_options
            return _InnerPG(dist_opts.group_rank, dist_opts.group_size)

        def create_outer(dist_opts, pg_options=None):
            received_opts["outer"] = pg_options
            received_opts["outer_dist"] = dist_opts.pg_options
            pg = _OuterPG(dist_opts.group_rank, dist_opts.group_size)
            dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
            return pg

        dist.Backend.register_backend(
            "opts_inner",
            create_inner,
            extended_api=True,
        )
        dist.Backend.register_backend(
            "opts_outer",
            create_outer,
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"

        outer_opts = {"compression": True}
        inner_opts = {"batch_size": 64}
        dist_pg_opts = "nccl_options_placeholder"

        dist.init_process_group(
            "opts_outer(opts_inner)",
            rank=self.rank,
            world_size=self.world_size,
            pg_options={
                "opts_outer": outer_opts,
                "opts_inner": inner_opts,
                "dist": dist_pg_opts,
            },
        )

        try:
            # Outer creator got its own options and dist options
            self.assertEqual(received_opts["outer"], outer_opts)
            self.assertEqual(received_opts["outer_dist"], dist_pg_opts)

            # Inner creator got its own options and dist options
            self.assertEqual(received_opts["inner"], inner_opts)
            self.assertEqual(received_opts["inner_dist"], dist_pg_opts)
        finally:
            dist.destroy_process_group()

    def test_passthrough_pg_options_non_dict(self):
        """Non-dict pg_options passes through to dist as-is."""

        received_opts = {}

        class _InnerPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "ndopt_inner"

        class _OuterPG(dist.distributed_c10d.PassthroughProcessGroup):
            pass

        def create_inner(dist_opts, pg_options=None):
            received_opts["inner"] = pg_options
            received_opts["inner_dist"] = dist_opts.pg_options
            return _InnerPG(dist_opts.group_rank, dist_opts.group_size)

        def create_outer(dist_opts, pg_options=None):
            received_opts["outer"] = pg_options
            received_opts["outer_dist"] = dist_opts.pg_options
            pg = _OuterPG(dist_opts.group_rank, dist_opts.group_size)
            dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
            return pg

        dist.Backend.register_backend(
            "ndopt_inner",
            create_inner,
            extended_api=True,
        )
        dist.Backend.register_backend(
            "ndopt_outer",
            create_outer,
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"

        raw_opts = "some_nccl_options"
        dist.init_process_group(
            "ndopt_outer(ndopt_inner)",
            rank=self.rank,
            world_size=self.world_size,
            pg_options=raw_opts,
        )

        try:
            # Non-dict: creators get pg_options=None,
            # dist_opts.pg_options carries the raw value
            self.assertIsNone(received_opts["outer"])
            self.assertEqual(received_opts["outer_dist"], raw_opts)
            self.assertIsNone(received_opts["inner"])
            self.assertEqual(received_opts["inner_dist"], raw_opts)
        finally:
            dist.destroy_process_group()

    def test_passthrough_pg_options_invalid_key(self):
        """pg_options dict with unknown keys raises ValueError."""

        class _SimplePG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "valopt_simple"

        dist.Backend.register_backend(
            "valopt_simple",
            lambda *args, **kwargs: _SimplePG(args[0].group_rank, args[0].group_size),
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"

        with self.assertRaises(ValueError):
            dist.init_process_group(
                "valopt_simple",
                rank=self.rank,
                world_size=self.world_size,
                pg_options={"bogus_key": "value"},
            )

    def test_nested_three_layers(self):
        """Three-layer nesting: outer(middle(inner))."""

        calls = []

        class _Layer(dist.distributed_c10d.PassthroughProcessGroup):
            pass

        class _InnerPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "layer_inner"

            def all_reduce(self, tensor, op=None, async_op=False):
                calls.append("inner")
                tensor.add_(1)
                return None

        class _MiddlePG(_Layer):
            def broadcast(self, tensor, src=None, async_op=False, **kwargs):
                calls.append("middle")
                tensor.add_(10)
                return None

        class _OuterPG(_Layer):
            def barrier(self, async_op=False, **kwargs):
                calls.append("outer_barrier")
                return None

        def create_inner(dist_opts, pg_options=None):
            return _InnerPG(dist_opts.group_rank, dist_opts.group_size)

        def create_middle(dist_opts, pg_options=None):
            pg = _MiddlePG(dist_opts.group_rank, dist_opts.group_size)
            dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
            return pg

        def create_outer(dist_opts, pg_options=None):
            pg = _OuterPG(dist_opts.group_rank, dist_opts.group_size)
            dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
            return pg

        dist.Backend.register_backend(
            "layer_inner",
            create_inner,
            extended_api=True,
        )
        dist.Backend.register_backend(
            "layer_middle",
            create_middle,
            extended_api=True,
        )
        dist.Backend.register_backend(
            "layer_outer",
            create_outer,
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "layer_outer(layer_middle(layer_inner))",
            rank=self.rank,
            world_size=self.world_size,
        )

        try:
            # barrier: defined on _OuterPG
            dist.barrier()
            self.assertEqual(calls, ["outer_barrier"])

            # broadcast: not on _OuterPG, delegates to _MiddlePG
            calls.clear()
            tensor = torch.zeros(4)
            dist.broadcast(tensor, src=0)
            self.assertEqual(calls, ["middle"])
            self.assertEqual(tensor, torch.full((4,), 10.0))

            # all_reduce: not on _OuterPG or _MiddlePG,
            # delegates to _InnerPG
            calls.clear()
            tensor = torch.zeros(4)
            dist.all_reduce(tensor)
            self.assertEqual(calls, ["inner"])
            self.assertEqual(tensor, torch.ones(4))
        finally:
            dist.destroy_process_group()

    def test_passthrough_custom_work(self):
        """Passthrough PG can return a custom Work object."""

        calls = []

        class _CustomWork:
            def __init__(self, inner_work, op_name):
                self._inner = inner_work
                self._op_name = op_name

            def wait(self):
                calls.append(f"wait:{self._op_name}")
                if self._inner is not None:
                    return self._inner.wait()
                return True

        class _InnerPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "work_inner"

            def all_reduce(self, tensor, op=None, async_op=False):
                calls.append("inner_all_reduce")
                tensor.add_(5)
                return None

        class _OuterPG(dist.distributed_c10d.PassthroughProcessGroup):
            def all_reduce(self, tensor, op=None, async_op=False, **kwargs):
                inner_work = self._inner_pg.all_reduce(
                    tensor,
                    op=op,
                    async_op=async_op,
                    **kwargs,
                )
                calls.append("outer_wrapped")
                return _CustomWork(inner_work, "all_reduce")

        def create_inner(dist_opts, pg_options=None):
            return _InnerPG(dist_opts.group_rank, dist_opts.group_size)

        def create_outer(dist_opts, pg_options=None):
            pg = _OuterPG(dist_opts.group_rank, dist_opts.group_size)
            dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
            return pg

        dist.Backend.register_backend(
            "work_inner",
            create_inner,
            extended_api=True,
        )
        dist.Backend.register_backend(
            "work_outer",
            create_outer,
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "work_outer(work_inner)",
            rank=self.rank,
            world_size=self.world_size,
        )

        try:
            tensor = torch.zeros(4)
            work = dist.all_reduce(tensor, async_op=True)
            self.assertIsInstance(work, _CustomWork)
            work.wait()
            self.assertEqual(
                calls,
                ["inner_all_reduce", "outer_wrapped", "wait:all_reduce"],
            )
            self.assertEqual(tensor, torch.full((4,), 5.0))
        finally:
            dist.destroy_process_group()

    def test_passthrough_new_group_wraps_subgroup(self):
        """Passthrough PG can override new_group to wrap subgroups."""

        calls = []

        class _InnerPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "ng_inner"

            def all_reduce(self, tensor, op=None, async_op=False):
                calls.append("inner_all_reduce")
                tensor.add_(1)
                return None

        class _OuterPG(dist.distributed_c10d.PassthroughProcessGroup):
            def all_reduce(self, tensor, op=None, async_op=False, **kwargs):
                calls.append("outer_all_reduce")
                return self._inner_pg.all_reduce(
                    tensor,
                    op=op,
                    async_op=async_op,
                    **kwargs,
                )

            def new_group(
                self,
                ranks,
                timeout=None,
                pg_options=None,
                group_name=None,
                group_desc=None,
            ):
                calls.append("outer_new_group")
                my_rank = self.rank()
                if my_rank not in ranks:
                    return None
                inner_sub = _InnerPG(
                    ranks.index(my_rank),
                    len(ranks),
                )
                sub = _OuterPG(
                    ranks.index(my_rank),
                    len(ranks),
                )
                sub._inner_pg = inner_sub
                sub._dist = self._dist
                dist.register_process_group(sub, group_name, ranks)
                return sub

        def create_inner(dist_opts, pg_options=None):
            return _InnerPG(
                dist_opts.group_rank,
                dist_opts.group_size,
            )

        def create_outer(dist_opts, pg_options=None):
            pg = _OuterPG(
                dist_opts.group_rank,
                dist_opts.group_size,
            )
            dist.distributed_c10d.setup_inner_pg(pg, dist_opts)
            return pg

        dist.Backend.register_backend(
            "ng_inner",
            create_inner,
            extended_api=True,
        )
        dist.Backend.register_backend(
            "ng_outer",
            create_outer,
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "ng_outer(ng_inner)",
            rank=self.rank,
            world_size=self.world_size,
        )

        try:
            sub_pg = dist.new_group(ranks=[0, 1])
            self.assertIn("outer_new_group", calls)

            if self.rank in [0, 1]:
                self.assertIsNotNone(sub_pg)
                self.assertNotEqual(
                    sub_pg,
                    dist.distributed_c10d.GroupMember.NON_GROUP_MEMBER,
                )
                self.assertIsInstance(
                    sub_pg,
                    _OuterPG,
                )

                calls.clear()
                tensor = torch.zeros(4)
                dist.all_reduce(tensor, group=sub_pg)
                self.assertIn("outer_all_reduce", calls)
                self.assertIn("inner_all_reduce", calls)
                self.assertEqual(tensor, torch.ones(4))
            else:
                self.assertTrue(
                    sub_pg is None
                    or sub_pg == dist.distributed_c10d.GroupMember.NON_GROUP_MEMBER
                )
        finally:
            dist.destroy_process_group()

    def test_terminal_split_group_with_register(self):
        """Terminal PG can override split_group and register subgroups."""

        calls = []

        class _TerminalPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "sg_terminal"

            def all_reduce(self, tensor, op=None, async_op=False):
                calls.append("all_reduce")
                tensor.fill_(float(self.size()))
                return None

            def split_group(
                self,
                split_ranks=None,
                timeout=None,
                pg_options=None,
                group_desc=None,
                **kwargs,
            ):
                calls.append("split_group")
                my_rank = self.rank()
                my_ranks = None
                for rank_list in split_ranks:
                    if my_rank in rank_list:
                        my_ranks = rank_list
                        break
                if my_ranks is None:
                    return None
                child_name = group_desc or "split"
                sub = _TerminalPG(
                    my_ranks.index(my_rank),
                    len(my_ranks),
                )
                dist.register_process_group(sub, child_name, my_ranks)
                return sub

        dist.Backend.register_backend(
            "sg_terminal",
            lambda *args, **kwargs: _TerminalPG(args[0].group_rank, args[0].group_size),
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "sg_terminal",
            rank=self.rank,
            world_size=self.world_size,
        )

        try:
            sub_pg = dist.split_group(
                split_ranks=[[0, 1], [2, 3]],
                group_desc="test_split",
            )
            self.assertIn("split_group", calls)
            self.assertIsNotNone(sub_pg)
            self.assertEqual(sub_pg.size(), 2)

            calls.clear()
            tensor = torch.zeros(4)
            dist.all_reduce(tensor, group=sub_pg)
            self.assertIn("all_reduce", calls)
            self.assertEqual(tensor, torch.full((4,), 2.0))
        finally:
            dist.destroy_process_group()

    def test_terminal_new_group_with_register(self):
        """Terminal PG can override new_group and register subgroups."""

        calls = []

        class _TerminalPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "ngt_terminal"

            def all_reduce(self, tensor, op=None, async_op=False):
                calls.append("all_reduce")
                tensor.fill_(float(self.size()))
                return None

            def new_group(
                self,
                ranks,
                timeout=None,
                pg_options=None,
                group_name=None,
                group_desc=None,
            ):
                calls.append("new_group")
                my_rank = self.rank()
                if my_rank not in ranks:
                    return None
                sub = _TerminalPG(
                    ranks.index(my_rank),
                    len(ranks),
                )
                dist.register_process_group(sub, group_name, ranks)
                return sub

        dist.Backend.register_backend(
            "ngt_terminal",
            lambda *args, **kwargs: _TerminalPG(args[0].group_rank, args[0].group_size),
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "ngt_terminal",
            rank=self.rank,
            world_size=self.world_size,
        )

        try:
            sub_pg = dist.new_group(ranks=[0, 1])
            self.assertIn("new_group", calls)

            if self.rank in [0, 1]:
                self.assertIsNotNone(sub_pg)
                self.assertEqual(sub_pg.size(), 2)

                calls.clear()
                tensor = torch.zeros(4)
                dist.all_reduce(tensor, group=sub_pg)
                self.assertIn("all_reduce", calls)
                self.assertEqual(tensor, torch.full((4,), 2.0))
            else:
                self.assertTrue(
                    sub_pg is None
                    or sub_pg == dist.distributed_c10d.GroupMember.NON_GROUP_MEMBER
                )
        finally:
            dist.destroy_process_group()

    def test_pg_bypass_isend_irecv(self):
        """@_pg_bypass forwards dist.isend/irecv to pg.isend/irecv."""

        calls = []

        class _P2PPG(dist.ProcessGroup):
            def __init__(self, rank, size):
                super().__init__(rank, size)

            def getBackendName(self):
                return "p2p_bypass"

            def isend(self, tensor, dst=0, tag=0, group_dst=None, **kwargs):
                calls.append(
                    {
                        "op": "isend",
                        "shape": list(tensor.shape),
                        "dst": dst,
                    }
                )

                class _FakeWork:
                    def wait(self):
                        return True

                return _FakeWork()

            def irecv(self, tensor, src=None, tag=0, group_src=None, **kwargs):
                calls.append(
                    {
                        "op": "irecv",
                        "shape": list(tensor.shape),
                        "src": src,
                    }
                )
                tensor.fill_(99.0)

                class _FakeWork:
                    def wait(self):
                        return True

                return _FakeWork()

        dist.Backend.register_backend(
            "p2p_bypass",
            lambda *args, **kwargs: _P2PPG(args[0].group_rank, args[0].group_size),
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "p2p_bypass",
            rank=self.rank,
            world_size=self.world_size,
        )

        try:
            tensor = torch.zeros(4)
            if self.rank == 0:
                work = dist.isend(tensor, dst=1)
                self.assertIsNotNone(work)
                work.wait()
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["op"], "isend")
                self.assertEqual(calls[0]["dst"], 1)
            else:
                work = dist.irecv(tensor, src=0)
                self.assertIsNotNone(work)
                work.wait()
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["op"], "irecv")
                self.assertEqual(calls[0]["src"], 0)
                self.assertEqual(tensor, torch.full((4,), 99.0))
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
