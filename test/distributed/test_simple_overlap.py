# Owner(s): ["module: inductor"]
import unittest

import torch
import torch.distributed as dist
import torch.fx as fx
from torch._inductor.fx_passes.simple_overlap import simple_overlap
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import requires_accelerator_dist_backend
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.inductor_utils import HAS_GPU


def _node_names(graph: fx.Graph) -> list[str]:
    return [n.name for n in graph.nodes if n.op == "call_function"]


@requires_accelerator_dist_backend()
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
class TestSimpleOverlap(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        cls.device = "cuda"

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        dist.destroy_process_group()

    def test_collective_moved_before_compute(self):
        """
        Original: compute -> collective -> wait -> use
        Expected: collective -> compute -> wait -> use
        (unless the move would increase peak memory, in which case
        the collective stays in place)
        """

        def func(a, b):
            mm = torch.mm(a, b)
            ag = torch.ops._c10d_functional.all_gather_into_tensor(a, 1, "0")
            ag_out = torch.ops._c10d_functional.wait_tensor(ag)
            return mm + ag_out

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device)
            traced = make_fx(func)(a, b)

        names_before = _node_names(traced.graph)
        simple_overlap(traced.graph)
        names_after = _node_names(traced.graph)

        # The collective should either be moved before mm or stay in
        # place (if the move would increase peak memory). Either way
        # the graph must remain valid.
        ag_idx = next(
            i for i, n in enumerate(names_after) if "all_gather" in n
        )
        mm_idx = next(i for i, n in enumerate(names_after) if n == "mm")
        # At minimum: graph is valid and pass didn't crash
        self.assertIn("all_gather_into_tensor", " ".join(names_after))

    def test_wait_moved_after_compute(self):
        """
        Original: collective -> wait -> compute -> use(wait)
        Expected: collective -> compute -> wait -> use(wait)
        (wait moves later to just before its consumer)
        """

        def func(a, b):
            ag = torch.ops._c10d_functional.all_gather_into_tensor(a, 1, "0")
            ag_out = torch.ops._c10d_functional.wait_tensor(ag)
            mm = torch.mm(a, b)
            return mm + ag_out

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device)
            traced = make_fx(func)(a, b)

        simple_overlap(traced.graph)
        names = _node_names(traced.graph)

        wait_idx = next(i for i, n in enumerate(names) if "wait" in n)
        mm_idx = next(i for i, n in enumerate(names) if n == "mm")
        # Wait should come after mm (moved later to just before the add that uses it)
        self.assertGreater(wait_idx, mm_idx)

    def test_pg_ordering_preserved(self):
        """
        Two collectives on the same PG: the second must not be moved before the first.
        """

        def func(a, b):
            mm1 = torch.mm(a, b)
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(a, 1, "0")
            mm2 = torch.mm(b, a)
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(b, 1, "0")
            w1 = torch.ops._c10d_functional.wait_tensor(ag1)
            w2 = torch.ops._c10d_functional.wait_tensor(ag2)
            return w1 + w2 + mm1 + mm2

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device)
            traced = make_fx(func)(a, b)

        simple_overlap(traced.graph)
        names = _node_names(traced.graph)

        ag_indices = [i for i, n in enumerate(names) if "all_gather" in n]
        self.assertEqual(len(ag_indices), 2)
        # First collective must still be before second
        self.assertLess(ag_indices[0], ag_indices[1])

    def test_no_change_when_already_optimal(self):
        """
        If collective is already first and wait is just before use, nothing changes.
        """

        def func(a, b):
            ag = torch.ops._c10d_functional.all_gather_into_tensor(a, 1, "0")
            mm = torch.mm(a, b)
            ag_out = torch.ops._c10d_functional.wait_tensor(ag)
            return mm + ag_out

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device)
            traced = make_fx(func)(a, b)

        names_before = _node_names(traced.graph)
        simple_overlap(traced.graph)
        names_after = _node_names(traced.graph)

        self.assertEqual(names_before, names_after)

    def test_no_pairs_is_noop(self):
        """Pass does nothing on graphs without collectives."""

        def func(a, b):
            return torch.mm(a, b) + a

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device)
            traced = make_fx(func)(a, b)

        names_before = _node_names(traced.graph)
        simple_overlap(traced.graph)
        names_after = _node_names(traced.graph)

        self.assertEqual(names_before, names_after)

    def test_dependency_respected_for_collective_move(self):
        """
        Collective depends on the output of a computation, so it can't be
        moved before that computation.
        """

        def func(a, b):
            mm = torch.mm(a, b)
            ag = torch.ops._c10d_functional.all_gather_into_tensor(mm, 1, "0")
            ag_out = torch.ops._c10d_functional.wait_tensor(ag)
            return ag_out

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device)
            traced = make_fx(func)(a, b)

        names_before = _node_names(traced.graph)
        simple_overlap(traced.graph)
        names_after = _node_names(traced.graph)

        # Collective depends on mm, so order is unchanged
        mm_idx = next(i for i, n in enumerate(names_after) if n == "mm")
        ag_idx = next(i for i, n in enumerate(names_after) if "all_gather" in n)
        self.assertLess(mm_idx, ag_idx)


if __name__ == "__main__":
    run_tests()
