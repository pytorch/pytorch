# Owner(s): ["module: inductor"]
import unittest

import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case
import torch.distributed as dist
import torch.fx as fx

# for some reason importing functional collectives after dynamo breaks collectives handling!
from torch._C import FileCheck
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import requires_accelerator_dist_backend
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils._ordered_set import OrderedSet


# flake8: noqa: B950
# Owner(s): ["module: inductor"]


aten = torch.ops.aten

from torch.testing._internal.common_fsdp import get_devtype


device_type = str(get_devtype())


import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case


# for some reason importing functional collectives after dynamo breaks collectives handling!


@requires_accelerator_dist_backend(["nccl", "xccl"])
def build_collective_info(graph, hiding_annotations):
    """
    Build CollectiveInfo dict from manual hiding annotations.

    hiding_annotations: dict mapping collective_start -> hiding_compute_node
    """
    from torch._inductor.fx_passes.overlap_scheduling import CollectiveInfo

    collective_info = {}

    # Find all collective starts and their corresponding waits
    start_to_wait = {}
    for node in graph.nodes:
        if node.op == "call_function" and "wait_tensor" in str(node.target):
            wait_input = node.args[0]
            if isinstance(wait_input, fx.Node):
                start_to_wait[wait_input] = node

    # Build CollectiveInfo for each collective
    for start_node, wait_node in start_to_wait.items():
        hiding_node = hiding_annotations.get(start_node)

        # Estimate size and time
        size_bytes = 16 * 4  # 4x4 tensor of floats
        estimated_time_ms = 1.0  # Dummy time
        exposed_time_ms = 0.0 if hiding_node else 1.0  # Hidden if has hiding_node

        collective_info[start_node] = CollectiveInfo(
            start_node=start_node,
            wait_node=wait_node,
            size_bytes=size_bytes,
            estimated_time_ms=estimated_time_ms,
            exposed_time_ms=exposed_time_ms,
            hiding_node=hiding_node,
        )

    return collective_info


def compute_ancestors(graph):
    """Compute ancestor sets for all nodes in the graph."""
    node_ancestors = {}

    for node in graph.nodes:
        ancestors = OrderedSet()
        stack = list(node.all_input_nodes)
        visited = set()

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            ancestors.add(current)
            stack.extend(current.all_input_nodes)

        node_ancestors[node] = ancestors

    return node_ancestors


@requires_accelerator_dist_backend()
@unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
@instantiate_parametrized_tests
class TestOverlapPreservingBucketing(InductorTestCase):
    """
    Unit tests for overlap-preserving bucketing pass.
    """

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

    def test_can_bucket_independent_collectives(self):
        """
        Test that independent collectives with separate hiding nodes CAN bucket.

        Graph structure:
        ag1_start -> ag2_start -> mm1 (hides ag1) -> mm2 (hides ag2) -> ag1_wait -> ag2_wait
        """

        def func(a, b):
            group_name = "0"
            group_size = 1

            # Start both collectives
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )

            # Independent compute that can hide both
            mm1 = torch.mm(a, a)
            mm2 = torch.mm(b, b)

            # Wait for both
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + ag2_out.sum() + mm1.sum() + mm2.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device) * 2

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes using find_nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        mm1, mm2 = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            ag2: mm2,  # mm2 hides ag2
        }

        # Build collective info and ancestors
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        node_ancestors = compute_ancestors(traced.graph)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            node_ancestors,
            scheduled,
        )
        bucketer.bucket_collectives()

        # Verify: should have 1 bucketed collective (all_gather_into_tensor_out)
        graph_str = str(traced.graph)
        FileCheck().check_count("all_gather_into_tensor_out", 1, exactly=False).run(
            graph_str
        )

    def test_cant_bucket_nested_hiding_intervals(self):
        """
        Test that nested hiding intervals prevent bucketing.

        Graph structure:
        ag1_start -> ag2_start -> mm2 (hides ag2) -> ag2_wait -> mm1 (hides ag1) -> ag1_wait

        ag2's hiding interval is nested inside ag1's hiding interval.
        """

        def func(a, b):
            group_name = "0"
            group_size = 1

            # ag1 starts first
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )

            # ag2 starts (inside ag1's interval)
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )

            # mm2 hides ag2
            mm2 = torch.mm(b[:2, :2], b[:2, :2])

            # ag2 waits (still inside ag1's interval)
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)

            # mm1 uses ag2's result and hides ag1
            mm1 = torch.mm(a + ag2_out[:4, :4], a)

            # ag1 waits last
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)

            return ag1_out.sum() + ag2_out.sum() + mm1.sum() + mm2.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device) * 2

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes using find_nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        mm_nodes = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )
        # mm2 is the first mm, mm1 is the second (based on graph order)
        mm2 = mm_nodes[0]
        mm1 = mm_nodes[1]

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            ag2: mm2,  # mm2 hides ag2
        }

        # Build collective info and ancestors
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        node_ancestors = compute_ancestors(traced.graph)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            node_ancestors,
            scheduled,
        )
        bucketer.bucket_collectives()

        # Verify: nested hiding intervals should prevent bucketing
        # Should have 2 separate all_gathers, not 1 bucketed one
        graph_str = str(traced.graph)
        FileCheck().check_count("all_gather_into_tensor", 2, exactly=False).run(
            graph_str
        )

    @parametrize("final_mm_hidden", (True, False))
    def test_cant_bucket_ag_with_rs_hiding_interval_between(self, final_mm_hidden):
        """
        Test that all_gathers can't bucket when a reduce_scatter's hiding interval is between them.

        Graph structure:
        ag1_start -> mm1 (hides ag1) -> ag1_wait ->
        rs_start -> mm2 (hides rs) -> rs_wait ->

        if final_mm_hidden:
            ag2_start -> mm3 (hides ag2) -> ag2_wait

        if final_mm_hidden:
            Bucketing ag1 and ag2 would require moving one of them, which would break hiding relationships:
            - Moving ag2 earlier would break ag2's hiding by mm3
            - Moving ag1 later would break ag1's hiding by mm1
            - The rs hiding interval creates an obstacle between them

        otherwise, we can bucket
        """

        def func(a, b, c):
            group_name = dist.distributed_c10d._get_default_group().group_name
            group_size = 1

            # First all_gather
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            mm1 = torch.mm(a, a)  # hides ag1
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)

            # Reduce scatter in between
            rs = torch.ops._c10d_functional.reduce_scatter_tensor(
                b, "sum", group_size, group_name
            )
            mm2 = torch.mm(b[:4, :4], b[:4, :4])  # hides rs
            rs_out = torch.ops._c10d_functional.wait_tensor(rs)

            # Second all_gather
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                c, group_size, group_name
            )
            mm3 = torch.mm(c, c)  # hides ag2
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + rs_out.sum() + ag2_out.sum(), mm1, mm2, mm3

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(8, 4, device=self.device)
            c = torch.ones(4, 4, device=self.device)

            # Trace with make_fx
            traced = make_fx(func)(a, b, c)

        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        (rs,) = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
        )
        mm1, mm2, mm3 = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            # rs: mm2,   # mm2 hides rs
            ag2: mm3,
        }
        if final_mm_hidden:
            hiding_annotations[rs] = mm2

        # Build collective info and ancestors
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        node_ancestors = compute_ancestors(traced.graph)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing logic to find buckets (without applying them, which would require process groups)
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            node_ancestors,
            scheduled,
        )

        bucketer.bucket_collectives()

        graph_str = str(traced.graph)

        # check order of mms preserved
        FileCheck().check("%mm").check("%mm_1").check("%mm_2").run(graph_str)

        if final_mm_hidden:
            # Should NOT bucket - 2 separate all_gathers
            # Count all_gather node names (works even when wrapped in control_deps)
            FileCheck().check_count("%all_gather_into_tensor", 2, exactly=False).run(
                graph_str
            )
        else:
            # Should bucket - 1 bucketed all_gather (all_gather_into_tensor_out)
            FileCheck().check_count(
                "%all_gather_into_tensor_out", 1, exactly=False
            ).run(graph_str)

    def test_can_bucket_all_reduce(self):
        """
        Test that all_reduce operations CAN bucket together.

        Graph structure:
        ar1_start -> ar2_start -> mm1 (hides ar1) -> mm2 (hides ar2) -> ar1_wait -> ar2_wait
        """

        def func(a, b):
            group_name = "0"

            # Start both all_reduce operations
            ar1 = torch.ops._c10d_functional.all_reduce(a, "sum", group_name)
            ar2 = torch.ops._c10d_functional.all_reduce(b, "sum", group_name)

            # Independent compute that can hide both
            mm1 = torch.mm(a, a)
            mm2 = torch.mm(b, b)

            # Wait for both
            ar1_out = torch.ops._c10d_functional.wait_tensor(ar1)
            ar2_out = torch.ops._c10d_functional.wait_tensor(ar2)

            return ar1_out.sum() + ar2_out.sum() + mm1.sum() + mm2.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device)
            b = torch.ones(4, 4, device=self.device) * 2

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes
        ar1, ar2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_reduce.default,
        )
        mm1, mm2 = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )

        # For all_reduce, start_node == wait_node (no separate wait)
        hiding_annotations = {
            ar1: mm1,
            ar2: mm2,
        }

        # Build collective info
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        node_ancestors = compute_ancestors(traced.graph)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            node_ancestors,
            scheduled,
        )
        bucketer.bucket_collectives()

        # Verify: should have 1 bucketed all_reduce
        # After bucketing, there should be only one all_reduce node (the bucketed one)
        graph_str = str(traced.graph)
        FileCheck().check_count("%all_reduce", 1, exactly=True).check_count(
            "%mm", 2
        ).run(graph_str)

    def test_can_bucket_multidtype_collectives(self):
        """
        Test that all_gathers with different dtypes CAN bucket together.

        Graph structure:
        ag1_float32 -> mm1 (hides ag1) -> ag1_wait
        ag2_bfloat16 -> mm2 (hides ag2) -> ag2_wait
        """

        def func(a, b):
            group_name = "0"
            group_size = 1

            # Start both collectives with different dtypes
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a,
                group_size,
                group_name,  # float32
            )
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                b,
                group_size,
                group_name,  # bfloat16
            )

            # Independent compute that can hide both
            mm1 = torch.mm(a, a)
            mm2 = torch.mm(b.float(), b.float())

            # Wait for both
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + ag2_out.sum() + mm1.sum() + mm2.sum()

        # Use fake mode to trace without executing
        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device, dtype=torch.float32)
            b = torch.ones(4, 4, device=self.device, dtype=torch.bfloat16)

            # Trace with make_fx
            traced = make_fx(func)(a, b)

        # Find nodes using find_nodes
        ag1, ag2 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        mm_nodes = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )
        mm1 = mm_nodes[0]
        mm2 = mm_nodes[1]

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            ag2: mm2,  # mm2 hides ag2
        }

        # Build collective info and ancestors
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        node_ancestors = compute_ancestors(traced.graph)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing with multidtype mode
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            node_ancestors,
            scheduled,
            bucket_mode="custom_ops_multidtype",
        )
        bucketer.bucket_collectives()

        # Verify: should have 1 bucketed collective (all_gather_into_tensor_out)
        # even though dtypes are different
        graph_str = str(traced.graph)
        FileCheck().check_count("all_gather_into_tensor_out", 1, exactly=False).run(
            graph_str
        )


if __name__ == "__main__":
    run_tests()
