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
from torch._dynamo.utils import same
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

    hiding_annotations: dict mapping collective_start -> hiding_compute_node(s)
                        Can be a single node or a list/OrderedSet of nodes
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
        hiding_annotation = hiding_annotations.get(start_node)

        # Convert to OrderedSet
        hiding_nodes = OrderedSet()
        if hiding_annotation is not None:
            if isinstance(hiding_annotation, list | OrderedSet):
                hiding_nodes = OrderedSet(hiding_annotation)
            else:
                hiding_nodes = OrderedSet([hiding_annotation])

        # Estimate size and time
        size_bytes = 16 * 4  # 4x4 tensor of floats
        estimated_time_ms = 1.0  # Dummy time
        exposed_time_ms = 0.0 if hiding_nodes else 1.0  # Hidden if has hiding_nodes

        collective_info[start_node] = CollectiveInfo(
            start_node=start_node,
            wait_node=wait_node,
            size_bytes=size_bytes,
            estimated_time_ms=estimated_time_ms,
            exposed_time_ms=exposed_time_ms,
            hiding_nodes=hiding_nodes,
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

    def test_can_bucket_with_multiple_hiding_nodes(self):
        """
        Test that collectives with multiple hiding nodes CAN bucket.

        Graph structure:
        ag1_start -> ag2_start -> mm1 -> mm2 -> mm3 -> ag1_wait -> ag2_wait

        Where:
        - ag1 is hidden by mm1 and mm2
        - ag2 is hidden by mm2 and mm3
        - Both collectives share mm2 as a hiding node
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

            # Three compute operations that hide the collectives
            mm1 = torch.mm(a, a)
            mm2 = torch.mm(b, b)
            mm3 = torch.mm(a + b, a + b)

            # Wait for both
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)
            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)

            return ag1_out.sum() + ag2_out.sum() + mm1.sum() + mm2.sum() + mm3.sum()

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
        mm1, mm2, mm3 = traced.graph.find_nodes(
            op="call_function", target=torch.ops.aten.mm.default
        )

        # Manually annotate hiding relationships with multiple hiding nodes
        hiding_annotations = {
            ag1: [mm1, mm2],  # ag1 is hidden by mm1 and mm2
            ag2: [mm2, mm3],  # ag2 is hidden by mm2 and mm3
        }

        # Build collective info and ancestors
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        node_ancestors = compute_ancestors(traced.graph)
        scheduled = OrderedSet(traced.graph.nodes)

        # Verify hiding_nodes are correctly set
        self.assertEqual(len(collective_info[ag1].hiding_nodes), 2)
        self.assertIn(mm1, collective_info[ag1].hiding_nodes)
        self.assertIn(mm2, collective_info[ag1].hiding_nodes)
        self.assertEqual(len(collective_info[ag2].hiding_nodes), 2)
        self.assertIn(mm2, collective_info[ag2].hiding_nodes)
        self.assertIn(mm3, collective_info[ag2].hiding_nodes)

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

        FileCheck().check_count(
            "all_gather_into_tensor_out", 1, exactly=False
        ).check_count("torch.ops.aten.mm.default", 3, exactly=True).run(
            str(traced.graph)
        )

    def test_can_bucket_with_convert_dtype_as_hiding_nodes(self):
        """
        Test that all_gathers can bucket when convert_element_type ops ARE the hiding nodes.

        Graph structure:
        ag1_start -> convert1 (hides ag1) -> ag1_wait -> ag2_start -> convert2 (hides ag2) -> ag2_wait

        The convert_element_type ops ARE hiding nodes - no matmuls.
        This tests that dependencies are transferred correctly when convert nodes are erased.
        """

        def func(a, b, c):
            group_name = "0"
            group_size = 1

            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(
                a, group_size, group_name
            )
            b = torch.ops.prims.convert_element_type.default(b, torch.float16)
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)

            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(
                b, group_size, group_name
            )
            ag3 = torch.ops._c10d_functional.all_gather_into_tensor(
                c, group_size, group_name
            )

            mm = ag1_out @ ag1_out

            ag2_out = torch.ops._c10d_functional.wait_tensor(ag2)
            ag3_out = torch.ops._c10d_functional.wait_tensor(ag3)

            return ag1_out, ag2_out, ag3_out, mm

        with FakeTensorMode():
            a = torch.ones(4, 4, device=self.device, dtype=torch.float32)
            b = torch.ones(4, 4, device=self.device, dtype=torch.float32)
            c = torch.ones(4, 4, device=self.device, dtype=torch.float32)

            traced = make_fx(func)(a, b, c)

        # Find nodes
        ag1, ag2, ag3 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        convert1 = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops.prims.convert_element_type.default,
        )[0]
        mm = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.mm.default,
        )[0]

        hiding_annotations = {
            ag1: convert1,
            ag2: mm,
            ag3: mm,
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

        graph_str = str(traced.graph)

        f = FileCheck()
        f.check_count("%all_gather_into_tensor", 1, exactly=True)
        f.check("pre_bucket_all_gather").check("wait_tensor").check(
            "%all_gather_into_tensor_out"
        )

    def test_split_mm(self):
        def func(a, b):
            a = a * 2
            b = b * 3
            mm = torch.mm(a, b)
            mm = mm * 2
            return mm

        def _inps():
            return torch.randn(16, 8, device=self.device), torch.randn(
                8, 4, device=self.device
            )

        inps = _inps()
        ref_out = func(*inps)

        gm = make_fx(func, tracing_mode="fake")(*inps)

        from torch._inductor.fx_passes.decompose_mm import split_mm

        split_mm(gm, 4, [4])
        graph_str = str(gm.graph)
        FileCheck().check_count(
            "torch.ops.aten.mm",
            4,
            exactly=True,
        ).run(graph_str)
        out = gm(*inps)

        self.assertTrue(same(out, ref_out))

    def test_split_mm_noncont(self):
        # Non contiguous matmuls are not split
        def func(a, b):
            return torch.mm(a, b)

        def _inps():
            return torch.empty_strided((16, 8), (1, 8)), torch.randn(8, 4)

        inps = _inps()

        gm = make_fx(func, tracing_mode="fake")(*inps)
        from torch._inductor.fx_passes.decompose_mm import split_mm

        split_mm(gm, 16, [4])
        graph_str = str(gm.graph)
        FileCheck().check_count(
            "torch.ops.aten.mm",
            1,
            exactly=True,
        ).run(graph_str)

    def test_split_mm_pw_rs(self):
        # permute_89: "bf16[16032, 16384][1, 16032]cuda:0"
        # cat_33: "bf16[16384, 8192][8192, 1]cuda:0"
        # mm_57: "bf16[16032, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_89, cat_33);  permute_89 = cat_33 = None
        # convert_element_type_275: "f32[16032, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        # reduce_scatter_tensor_8: "f32[1002, 8192][8192, 1]cuda:0" = torch.ops._c10d_functional.reduce_scatter_tensor.default(convert_element_type_275, 'sum', 16, '1');  convert_element_type_275 = None
        # wait_tensor_126: "f32[1002, 8192][8192, 1]cuda:0" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_8);  reduce_scatter_tensor_8 = None
        def func(permute_89, cat_33):
            mm_57 = torch.ops.aten.mm.default(permute_89, cat_33)
            convert_element_type_275 = torch.ops.prims.convert_element_type.default(
                mm_57, torch.float32
            )
            reduce_scatter_tensor_8 = (
                torch.ops._c10d_functional.reduce_scatter_tensor.default(
                    convert_element_type_275, "sum", 16, "1"
                )
            )
            wait_tensor_126 = torch.ops._c10d_functional.wait_tensor.default(
                reduce_scatter_tensor_8
            )
            return wait_tensor_126

        def inps():
            return (
                torch.randn(16032, 16384, dtype=torch.bfloat16, device=self.device),
                torch.randn(16384, 8192, dtype=torch.bfloat16, device=self.device),
            )

        fake_tensor_mode = FakeTensorMode()
        with fake_tensor_mode:
            ins = inps()
            gm = make_fx(func)(*ins)

        from torch._inductor.fx_passes.decompose_mm import _split_mm_rs

        num_chunks = 2
        _split_mm_rs(gm, [num_chunks], 1)
        graph_str = str(gm.graph)
        FileCheck().check_count(
            "torch.ops.aten.mm",
            num_chunks,
            exactly=True,
        ).run(graph_str)
        FileCheck().check_count(
            "torch.ops.prims.convert_element_type.default",
            num_chunks,
            exactly=True,
        ).run(graph_str)
        FileCheck().check_count(
            "torch.ops._c10d_functional.reduce_scatter_tensor.default",
            num_chunks,
            exactly=True,
        ).run(graph_str)

    def test_split_mm_cat_rs(self):
        # add_29: "bf16[2, 8192, 1024][8388608, 1024, 1]cuda:0"
        # permute_87: "bf16[3584, 8192][1, 3584]cuda:0"
        # view_198: "bf16[16384, 3584][3584, 1]cuda:0"
        # mm_55: "bf16[16384, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_198, permute_87)
        # view_199: "bf16[2, 8192, 8192][67108864, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [2, 8192, 8192]);  mm_55 = None
        # split_33 = torch.ops.aten.split.Tensor(view_199, 1024, 2);  view_199 = None
        # getitem_336: "bf16[2, 8192, 1024][67108864, 8192, 1]cuda:0" = split_33[0]
        # getitem_337: "bf16[2, 8192, 1024][67108864, 8192, 1]cuda:0" = split_33[1]
        # getitem_338: "bf16[2, 8192, 1024][67108864, 8192, 1]cuda:0" = split_33[2]
        # getitem_339: "bf16[2, 8192, 1024][67108864, 8192, 1]cuda:0" = split_33[3]
        # getitem_340: "bf16[2, 8192, 1024][67108864, 8192, 1]cuda:0" = split_33[4]
        # getitem_341: "bf16[2, 8192, 1024][67108864, 8192, 1]cuda:0" = split_33[5]
        # getitem_342: "bf16[2, 8192, 1024][67108864, 8192, 1]cuda:0" = split_33[6]
        # getitem_343: "bf16[2, 8192, 1024][67108864, 8192, 1]cuda:0" = split_33[7];  split_33 = None
        # cat_32: "bf16[16, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.cat.default([getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342, getitem_343]);  getitem_336 = getitem_337 = getitem_338 = getitem_339 = getitem_340 = getitem_341 = getitem_342 = getitem_343 = None
        # reduce_scatter_tensor_7: "bf16[2, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_32, 'sum', 8, '9');  cat_32 = None
        # wait_tensor_121: "bf16[2, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_7);  reduce_scatter_tensor_7 = None
        # add_31: "bf16[2, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_29, wait_tensor_121);  add_29 = wait_tensor_121 = None

        def func(add_29, permute_87, view_198):
            mm_55 = torch.ops.aten.mm.default(view_198, permute_87)
            view_199 = torch.ops.aten.reshape.default(mm_55, [2, 8192, 8192])
            split_33 = torch.ops.aten.split.Tensor(view_199, 1024, 2)
            getitem_336 = split_33[0]
            getitem_337 = split_33[1]
            getitem_338 = split_33[2]
            getitem_339 = split_33[3]
            getitem_340 = split_33[4]
            getitem_341 = split_33[5]
            getitem_342 = split_33[6]
            getitem_343 = split_33[7]
            cat_32 = torch.ops.aten.cat.default(
                [
                    getitem_336,
                    getitem_337,
                    getitem_338,
                    getitem_339,
                    getitem_340,
                    getitem_341,
                    getitem_342,
                    getitem_343,
                ]
            )
            reduce_scatter_tensor_7 = (
                torch.ops._c10d_functional.reduce_scatter_tensor.default(
                    cat_32, "sum", 8, "9"
                )
            )
            wait_tensor_121 = torch.ops._c10d_functional.wait_tensor.default(
                reduce_scatter_tensor_7
            )
            add_31 = torch.ops.aten.add.Tensor(add_29, wait_tensor_121)
            return add_31

        def inps():
            return (
                torch.randn(2, 8192, 1024, dtype=torch.bfloat16, device=self.device),
                torch.randn(3584, 8192, dtype=torch.bfloat16, device=self.device),
                torch.randn(16384, 3584, dtype=torch.bfloat16, device=self.device),
            )

        fake_tensor_mode = FakeTensorMode()
        with fake_tensor_mode:
            ins = inps()
            gm = make_fx(func)(*ins)

        from torch._inductor.fx_passes.decompose_mm import _split_mm_rs

        num_chunks = 2
        _split_mm_rs(gm, [num_chunks], 8192)
        graph_str = str(gm.graph)
        FileCheck().check_count(
            "torch.ops.aten.mm",
            num_chunks,
            exactly=True,
        ).run(graph_str)

        FileCheck().check_count(
            "torch.ops._c10d_functional.reduce_scatter_tensor_out.default",
            num_chunks,
            exactly=True,
        ).run(graph_str)


if __name__ == "__main__":
    run_tests()
