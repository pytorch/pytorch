# Owner(s): ["module: inductor"]
import unittest

import torch
import torch._dynamo
import torch.fx as fx
from torch._C import FileCheck
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import (
    DynamoDistributedSingleProcTestCase,
    requires_accelerator_dist_backend,
)
from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils._ordered_set import OrderedSet
import torch.distributed as dist
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    IS_X86,
    MACOS_VERSION,
    parametrize,
)
# flake8: noqa: B950
# Owner(s): ["module: inductor"]
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case

# for some reason importing functional collectives after dynamo breaks collectives handling!
import torch.distributed._functional_collectives as _functional_collectives
from torch._C import FileCheck
from torch._dynamo.utils import counters, same
from torch._inductor.utils import run_and_get_code, run_and_get_triton_code
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    at_least_x_gpu,
    DynamoDistributedMultiProcTestCase,
    requires_accelerator_dist_backend,
)


aten = torch.ops.aten
import functools

from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import skipIfRocm
from torch.testing._internal.inductor_utils import HAS_GPU

device_type = str(get_devtype())


import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case
import torch.distributed as c10d

# for some reason importing functional collectives after dynamo breaks collectives handling!
import torch.distributed._functional_collectives as _functional_collectives
from torch._C import FileCheck
from torch._dynamo.testing import CompileCounter
from torch._dynamo.utils import same
from torch._inductor.comms import (
    _reorder_communication_preserving_peak_memory_internal,
    ReorderInfo,
    sink_waits_iterative,
)
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch._inductor.scheduler import (
    _get_mm_like_fn,
    BaseSchedulerNode,
    get_estimate_runtime_cache,
    get_estimate_runtime_cache_key_from_snode,
)
from torch._inductor.utils import fresh_inductor_cache, run_and_get_triton_code
from torch.distributed.distributed_c10d import GroupMember
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    DynamoDistributedMultiProcTestCase,
    DynamoDistributedSingleProcTestCase,
    MultiProcessTestCase,
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
    skipIfXpu,
    TEST_XPU,
    xfailIf,
)
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils._python_dispatch import TorchDispatchMode


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



# @requires_accelerator_dist_backend(["nccl", "xccl"])
# @instantiate_parametrized_tests
# class TestCollectivesMultiProc(DynamoDistributedMultiProcTestCase):
#     """
#     Run correctness checks in multi-proc runner, mark with minimum # GPUs to run under
#     """

#     device = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

#     def get_world_trs(self):
#         return {
#             "tag": "",
#             "ranks": list(range(self.world_size)),
#             "group_size": self.world_size,
#         }

#     @property
#     def world_size(self) -> int:
#         # hack: no matter whether we have 2 or 3 or 4 gpus, just run on 2
#         # works around issue with skipif<2 and workers with unpredictable #s gpu
#         return 2

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
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(a, group_size, group_name)
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(b, group_size, group_name)

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

        # Find nodes by iterating through graph
        ag1, ag2, mm1, mm2 = None, None, None, None
        ag1_wait, ag2_wait = None, None

        for node in traced.graph.nodes:
            if node.op == "call_function":
                if "all_gather_into_tensor" in str(node.target):
                    if ag1 is None:
                        ag1 = node
                    else:
                        ag2 = node
                elif "mm" in str(node.target):
                    if mm1 is None:
                        mm1 = node
                    else:
                        mm2 = node
                elif "wait_tensor" in str(node.target):
                    if ag1_wait is None:
                        ag1_wait = node
                    else:
                        ag2_wait = node

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            ag2: mm2,  # mm2 hides ag2
        }

        # Build collective info and ancestors
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        node_ancestors = compute_ancestors(traced.graph)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing logic to find buckets (without applying them)
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )
        from torch._inductor.fx_passes.bucketing import bucket_key

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            node_ancestors,
            scheduled,
        )

        # Find buckets manually (same logic as bucket_collectives but without applying)
        from collections import defaultdict
        from torch.utils._ordered_set import OrderedSet as OS
        from torch._inductor.fx_passes.overlap_scheduling import get_group_name

        # Group by PG first
        pg_collectives = defaultdict(OS)
        for start in collective_info:
            pg = get_group_name(start)
            pg_collectives[pg].add(start)

        all_buckets = []
        for pg, collectives in pg_collectives.items():
            # Populate node_to_event for this PG
            bucketer._populate_node_to_event(pg)

            # Group by bucket key within this PG
            grouped_collectives = defaultdict(OS)
            for start in collectives:
                key = bucket_key(start)
                if key is not None:
                    grouped_collectives[key].add(start)

            # Find buckets for this PG
            for collective_group in grouped_collectives.values():
                buckets = bucketer._find_buckets(collective_group)
                all_buckets.extend(buckets)

        # Verify: should have 1 bucket with 2 collectives
        self.assertEqual(len(all_buckets), 1, f"Expected 1 bucket, got {len(all_buckets)}")
        self.assertEqual(len(all_buckets[0].collectives), 2, f"Expected 2 collectives in bucket, got {len(all_buckets[0].collectives)}")

        # Verify both collectives are in the bucket
        bucketed_colls = set(all_buckets[0].collectives)
        self.assertIn(ag1, bucketed_colls)
        self.assertIn(ag2, bucketed_colls)

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
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(a, group_size, group_name)

            # ag2 starts (inside ag1's interval)
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(b, group_size, group_name)

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

        # Find nodes
        ag1, ag2, mm1, mm2 = None, None, None, None

        for node in traced.graph.nodes:
            if node.op == "call_function":
                if "all_gather_into_tensor" in str(node.target):
                    if ag1 is None:
                        ag1 = node
                    else:
                        ag2 = node
                elif "mm" in str(node.target):
                    if mm2 is None:
                        mm2 = node
                    else:
                        mm1 = node

        # Manually annotate hiding relationships
        hiding_annotations = {
            ag1: mm1,  # mm1 hides ag1
            ag2: mm2,  # mm2 hides ag2
        }

        # Build collective info and ancestors
        collective_info = build_collective_info(traced.graph, hiding_annotations)
        node_ancestors = compute_ancestors(traced.graph)
        scheduled = OrderedSet(traced.graph.nodes)

        # Run bucketing logic to find buckets (without applying them)
        from torch._inductor.fx_passes.overlap_preserving_bucketer import (
            OverlapPreservingBucketer,
        )
        from torch._inductor.fx_passes.bucketing import bucket_key

        bucketer = OverlapPreservingBucketer(
            traced.graph,
            collective_info,
            node_ancestors,
            scheduled,
        )

        # Find buckets manually
        from collections import defaultdict
        from torch.utils._ordered_set import OrderedSet as OS
        from torch._inductor.fx_passes.overlap_scheduling import get_group_name

        # Group by PG first
        pg_collectives = defaultdict(OS)
        for start in collective_info:
            pg = get_group_name(start)
            pg_collectives[pg].add(start)

        all_buckets = []
        for pg, collectives in pg_collectives.items():
            # Populate node_to_event for this PG
            bucketer._populate_node_to_event(pg)

            # Group by bucket key within this PG
            grouped_collectives = defaultdict(OS)
            for start in collectives:
                key = bucket_key(start)
                if key is not None:
                    grouped_collectives[key].add(start)

            # Find buckets for this PG
            for collective_group in grouped_collectives.values():
                buckets = bucketer._find_buckets(collective_group)
                all_buckets.extend(buckets)

        # Verify: nested hiding intervals should prevent bucketing
        # So we should have either 0 buckets (both stay separate) or 2 buckets with 1 collective each
        # Either way, no bucket should have 2 collectives
        for bucket in all_buckets:
            self.assertLess(len(bucket.collectives), 2, "Nested hiding intervals should prevent bucketing of both collectives")

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
            ag1 = torch.ops._c10d_functional.all_gather_into_tensor(a, group_size, group_name)
            mm1 = torch.mm(a, a)  # hides ag1
            ag1_out = torch.ops._c10d_functional.wait_tensor(ag1)

            # Reduce scatter in between
            rs = torch.ops._c10d_functional.reduce_scatter_tensor(b, "sum", group_size, group_name)
            mm2 = torch.mm(b[:4, :4], b[:4, :4])  # hides rs
            rs_out = torch.ops._c10d_functional.wait_tensor(rs)

            # Second all_gather
            ag2 = torch.ops._c10d_functional.all_gather_into_tensor(c, group_size, group_name)
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

        ag1, ag2 = traced.graph.find_nodes(op="call_function", target=torch.ops._c10d_functional.all_gather_into_tensor.default)
        rs, = traced.graph.find_nodes(op="call_function", target=torch.ops._c10d_functional.reduce_scatter_tensor.default)
        mm1, mm2, mm3 = traced.graph.find_nodes(op="call_function", target=torch.ops.aten.mm.default)

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
            FileCheck().check_count("%all_gather_into_tensor", 2, exactly=False).run(graph_str)
        else:
            # Should bucket - 1 bucketed all_gather (all_gather_into_tensor_out)
            FileCheck().check_count("%all_gather_into_tensor_out", 1, exactly=False).run(graph_str)


if __name__ == "__main__":
    run_tests()
