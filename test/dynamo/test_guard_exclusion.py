# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo
import torch._dynamo.testing
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class GraphTracker:
    """Backend that tracks which compiled graph (by compilation order) handles each call."""

    def __init__(self):
        self.graphs = []
        self.call_log = []

    def __call__(self, gm, example_inputs):
        graph_id = len(self.graphs)
        self.graphs.append(gm)

        def wrapper(*args, **kwargs):
            self.call_log.append(graph_id)
            return gm.forward(*args, **kwargs)

        return wrapper

    @property
    def frame_count(self):
        return len(self.graphs)

    def reset(self):
        self.graphs.clear()
        self.call_log.clear()


@skipIfTorchDynamo("uses custom backend incompatible with PYTORCH_TEST_WITH_DYNAMO")
@torch._dynamo.config.patch(stable_graph_selection_for_automatic_dynamic=True)
class TestGuardExclusion(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_automatic_dynamic_exclusive_guard_basic(self):
        """
        Scenario with 2D tensors:
        1. Call with shape [3, 4] -> compiles Graph 0 (static)
        2. Call with shape [5, 4] -> dim 0 differs, triggers automatic_dynamic,
           compiles Graph 1 (dynamic dim 0)
        3. Call with shape [3, 4] -> same shape as first call.
           Which graph handles this: Graph 0 (static) or Graph 1 (dynamic)?
        """

        def foo(x):
            return x * 2

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        # Call 1: shape [3, 4] -> compiles Graph 0 (static)
        x1 = torch.randn(3, 4)
        result1 = opt(x1)
        self.assertEqual(tracker.frame_count, 1)
        self.assertEqual(tracker.call_log, [0], "Call 1 should use Graph 0")

        # Call 2: shape [5, 4] -> Graph 0 guard fails, compiles Graph 1 (dynamic)
        x2 = torch.randn(5, 4)
        result2 = opt(x2)
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log, [0, 1], "Call 2 should use Graph 1")

        # Call 3: shape [3, 4] -> same as first call.
        # Graph 0's static guard (size(0)==3) matches first, so it should
        # revert to Graph 0, not use Graph 1 (dynamic).
        x3 = torch.randn(3, 4)
        result3 = opt(x3)
        self.assertEqual(tracker.frame_count, 2, "No recompilation expected")
        self.assertEqual(
            tracker.call_log[2],
            0,
            "Call 3 [3,4] should use Graph 0 (static), same as call 1",
        )

        # Verify correctness
        self.assertEqual(result1, x1 * 2)
        self.assertEqual(result2, x2 * 2)
        self.assertEqual(result3, x3 * 2)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_automatic_dynamic_only_one_dim_changes(self):
        """
        Only dim 0 changes; dim 1 stays the same.
        Track which graph handles each call.
        """

        def foo(x):
            return x + 1

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        # Call 1: shape [3, 4] -> Graph 0 (static)
        opt(torch.randn(3, 4))
        # Call 2: shape [5, 4] -> Graph 1 (dynamic dim 0)
        opt(torch.randn(5, 4))
        # Call 3: shape [7, 4] -> should reuse dynamic graph
        opt(torch.randn(7, 4))
        # Call 4: shape [3, 4] -> same as first
        opt(torch.randn(3, 4))

        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[0], 0, "Call 1 [3,4] -> Graph 0 (static)")
        self.assertEqual(tracker.call_log[1], 1, "Call 2 [5,4] -> Graph 1 (dynamic)")
        self.assertEqual(tracker.call_log[2], 1, "Call 3 [7,4] -> Graph 1 (dynamic)")
        self.assertEqual(
            tracker.call_log[3],
            0,
            "Call 4 [3,4] should revert to Graph 0 (static), same shape as call 1",
        )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_automatic_dynamic_both_dims_change(self):
        """
        Both dimensions change across calls.
        """

        def foo(x):
            return x.sum()

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        opt(torch.randn(3, 4))
        opt(torch.randn(5, 6))
        opt(torch.randn(3, 4))
        opt(torch.randn(8, 9))

        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[0], 0, "Call 1 [3,4] -> Graph 0 (static)")
        self.assertEqual(tracker.call_log[1], 1, "Call 2 [5,6] -> Graph 1 (dynamic)")
        self.assertEqual(
            tracker.call_log[2],
            0,
            "Call 3 [3,4] should revert to Graph 0 (static), same shape as call 1",
        )
        self.assertEqual(tracker.call_log[3], 1, "Call 4 [8,9] -> Graph 1 (dynamic)")

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_accumulated_exclusion_does_not_shadow_intermediate_graph(self):
        """
        Scenario exposing the accumulation bug:
        1. func(3, 4)  -> Graph 0: static (3, 4)
        2. func(5, 4)  -> Graph 1: dynamic (s0, 4), excluded_sizes=(3, None)
        3. func(3, 19) -> Graph 2: dynamic (s0, s1), excluded_sizes should be
                          (None, 4) but the current code accumulates to (3, 4).
                          With AND logic the exclusion only rejects the exact
                          combo (3, 4), so input (5, 4) slips past Graph 2's
                          exclusion and is handled by Graph 2 instead of Graph 1.
        4. func(5, 4)  -> should use Graph 1 (s0, 4), NOT Graph 2 (s0, s1).
        """

        def foo(x):
            return x * 2

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        # Call 1: shape [3, 4] -> compiles Graph 0 (static)
        opt(torch.randn(3, 4))
        self.assertEqual(tracker.frame_count, 1)
        self.assertEqual(tracker.call_log[-1], 0)

        # Call 2: shape [5, 4] -> compiles Graph 1 (s0, 4)
        opt(torch.randn(5, 4))
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[-1], 1)

        # Call 3: shape [3, 19] -> Graph 1 exclusion rejects (size(0)==3),
        # Graph 0 rejects (19!=4), recompiles Graph 2 (s0, s1)
        opt(torch.randn(3, 19))
        self.assertEqual(tracker.frame_count, 3)
        self.assertEqual(tracker.call_log[-1], 2)

        # Call 4: shape [5, 4] -> should still use Graph 1 (s0, 4).
        # With the accumulation bug, Graph 2's exclusion is (3 AND 4),
        # so (5, 4) passes (5!=3) and Graph 2 steals the input.
        opt(torch.randn(5, 4))

        self.assertEqual(
            tracker.call_log[-1],
            1,
            "Input [5,4] should use Graph 1 (s0, 4), not Graph 2 (s0, s1). "
            "Graph 2's exclusion must reject size(1)==4 independently, not "
            "require size(0)==3 AND size(1)==4.",
        )

        # Call 5: shape [3, 4] -> should still use Graph 0 (static)
        opt(torch.randn(3, 4))
        self.assertEqual(
            tracker.call_log[-1],
            0,
            "Input [3,4] should use Graph 0 (static)",
        )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_4d_non_adjacent_dims_become_dynamic(self):
        """
        4D tensor where dims 1 and 3 change simultaneously, dims 0 and 2 stay static.
        Graph 0: (2, 3, 4, 5) static
        Graph 1: (2, dyn, 4, dyn) with excluded dim1=3, dim3=5
        Verify original shape routes back to Graph 0.
        """

        def foo(x):
            return x + 1

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        opt(torch.randn(2, 3, 4, 5))
        self.assertEqual(tracker.call_log[-1], 0)

        opt(torch.randn(2, 7, 4, 9))
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[-1], 1)

        # Original shape should route to Graph 0, not Graph 1
        opt(torch.randn(2, 3, 4, 5))
        self.assertEqual(tracker.call_log[-1], 0)

        # New dynamic shape should use Graph 1
        opt(torch.randn(2, 11, 4, 13))
        self.assertEqual(tracker.call_log[-1], 1)

        # Partially matching excluded: dim1=3 matches but dim3=13≠5.
        # AND semantics: Graph 1 accepts because NOT all excluded dims match.
        # This is correct — Graph 0 can't handle (2,3,4,13) anyway (dim3≠5).
        opt(torch.randn(2, 3, 4, 13))
        self.assertEqual(
            tracker.call_log[-1], 1, "(2,3,4,13) -> Graph 1 (AND: not all match)"
        )
        self.assertEqual(tracker.frame_count, 2, "No recompilation needed")

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_4d_progressive_dynamism_cascading(self):
        """
        4D tensor where dims become dynamic one at a time across recompilations.
        Each new graph is more general, and exclusion guards ensure inputs cascade
        to the most specialized graph.

        Graph 0: (2, 3, 4, 5) static
        Graph 1: (dyn, 3, 4, 5) excluded dim0=2
        Graph 2: (dyn, dyn, 4, 5) excluded dim1=3
        Graph 3: (dyn, dyn, dyn, 5) excluded dim2=4
        """

        def foo(x):
            return x.sum()

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        # Graph 0: (2, 3, 4, 5) static
        opt(torch.randn(2, 3, 4, 5))
        self.assertEqual(tracker.frame_count, 1)

        # Graph 1: dim0 changes -> (dyn, 3, 4, 5)
        opt(torch.randn(7, 3, 4, 5))
        self.assertEqual(tracker.frame_count, 2)

        # Graph 2: dim1 also changes -> (dyn, dyn, 4, 5)
        # Input (7, 8, 4, 5): Graph 1 rejects dim1=8≠3, Graph 0 rejects dim0=7≠2
        opt(torch.randn(7, 8, 4, 5))
        self.assertEqual(tracker.frame_count, 3)

        # Graph 3: dim2 also changes -> (dyn, dyn, dyn, 5)
        # Input (7, 8, 9, 5): Graph 2 rejects dim2=9≠4
        opt(torch.randn(7, 8, 9, 5))
        self.assertEqual(tracker.frame_count, 4)

        # Now verify cascading: each input routes to the most specialized graph.
        # (2, 3, 4, 5) -> Graph 0 (static, most specialized)
        opt(torch.randn(2, 3, 4, 5))
        self.assertEqual(tracker.call_log[-1], 0, "(2,3,4,5) -> Graph 0 (static)")

        # (7, 3, 4, 5) -> Graph 1 (dyn, 3, 4, 5)
        opt(torch.randn(7, 3, 4, 5))
        self.assertEqual(tracker.call_log[-1], 1, "(7,3,4,5) -> Graph 1 (dyn,3,4,5)")

        # (7, 8, 4, 5) -> Graph 2 (dyn, dyn, 4, 5)
        opt(torch.randn(7, 8, 4, 5))
        self.assertEqual(tracker.call_log[-1], 2, "(7,8,4,5) -> Graph 2 (dyn,dyn,4,5)")

        # (7, 8, 9, 5) -> Graph 3 (dyn, dyn, dyn, 5)
        opt(torch.randn(7, 8, 9, 5))
        self.assertEqual(
            tracker.call_log[-1], 3, "(7,8,9,5) -> Graph 3 (dyn,dyn,dyn,5)"
        )

        # (20, 30, 40, 5) -> Graph 3 (most general, no exclusion hit)
        opt(torch.randn(20, 30, 40, 5))
        self.assertEqual(
            tracker.call_log[-1], 3, "(20,30,40,5) -> Graph 3 (most general)"
        )

        self.assertEqual(tracker.frame_count, 4, "No additional recompilations")

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_5d_two_rounds_of_dynamism(self):
        """
        5D tensor with two rounds of automatic_dynamic. Verify inputs route
        to the most specialized graph after each round.

        Graph 0: (2, 3, 4, 5, 6) static
        Graph 1: (dyn, 3, 4, 5, 6) excluded dim0=2
        Graph 2: (dyn, 3, dyn, 5, 6) excluded dim2=4
        """

        def foo(x):
            return x * 2

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        # Graph 0: static
        opt(torch.randn(2, 3, 4, 5, 6))
        self.assertEqual(tracker.frame_count, 1)

        # Graph 1: dim0 becomes dynamic
        opt(torch.randn(8, 3, 4, 5, 6))
        self.assertEqual(tracker.frame_count, 2)

        # Graph 2: dim2 also becomes dynamic
        opt(torch.randn(8, 3, 10, 5, 6))
        self.assertEqual(tracker.frame_count, 3)

        # Verify routing:
        # Original static shape -> Graph 0
        opt(torch.randn(2, 3, 4, 5, 6))
        self.assertEqual(tracker.call_log[-1], 0, "Original -> Graph 0")

        # dim0 differs, dim2 matches static -> Graph 1
        opt(torch.randn(9, 3, 4, 5, 6))
        self.assertEqual(tracker.call_log[-1], 1, "dim0 changed -> Graph 1")

        # dim0 differs, dim2 differs -> Graph 2
        opt(torch.randn(9, 3, 11, 5, 6))
        self.assertEqual(tracker.call_log[-1], 2, "dim0+dim2 changed -> Graph 2")

        # dim0 is original excluded value, dim2 differs -> Graph 2 should still
        # accept because dim0's exclusion is None (already dynamic when snapshot taken)
        opt(torch.randn(2, 3, 11, 5, 6))
        self.assertEqual(
            tracker.call_log[-1],
            2,
            "dim0=2 with dim2≠4 -> Graph 2 (dim0 exclusion is None)",
        )

        self.assertEqual(tracker.frame_count, 3, "No additional recompilations")

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_many_entries_wrong_graph_selection(self):
        """
        Convoluted scenario: 4D tensor, three rounds of dynamism creating 4 graphs.
        Without exclusion guards, the most general graph would shadow all others.
        Test that each input gets the best (most specialized) match.

        Graph 0: (2, 3, 4, 5) static
        Graph 1: (dyn, 3, 4, 5) excluded dim0=2
        After Graph 1, (2, 8, 4, 5) triggers dim1 dynamic:
        Graph 2: (dyn, dyn, 4, 5) excluded dim1=3
        After Graph 2, (2, 8, 9, 5) triggers dim2 dynamic:
        Graph 3: (dyn, dyn, dyn, 5) excluded dim2=4

        Key: Graph 3 should NOT steal inputs that belong to Graph 0, 1, or 2.
        """

        def foo(x):
            return x.relu()

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        # Build up 4 graphs progressively
        opt(torch.randn(2, 3, 4, 5))  # Graph 0
        opt(torch.randn(7, 3, 4, 5))  # Graph 1: dim0 dynamic
        opt(torch.randn(7, 8, 4, 5))  # Graph 2: dim1 also dynamic
        opt(torch.randn(7, 8, 9, 5))  # Graph 3: dim2 also dynamic
        self.assertEqual(tracker.frame_count, 4)

        # Now stress-test routing with various inputs:
        test_cases = [
            # (shape, expected_graph, description)
            ((2, 3, 4, 5), 0, "exact original -> static Graph 0"),
            ((7, 3, 4, 5), 1, "dim0 differs -> Graph 1"),
            ((99, 3, 4, 5), 1, "dim0 differs (large) -> Graph 1"),
            ((7, 8, 4, 5), 2, "dim0+dim1 differ -> Graph 2"),
            ((7, 99, 4, 5), 2, "dim0+dim1 differ (large) -> Graph 2"),
            ((7, 8, 9, 5), 3, "dim0+dim1+dim2 differ -> Graph 3"),
            ((99, 99, 99, 5), 3, "all non-static dims differ -> Graph 3"),
        ]

        for shape, expected_graph, desc in test_cases:
            opt(torch.randn(*shape))
            self.assertEqual(tracker.call_log[-1], expected_graph, desc)

        self.assertEqual(tracker.frame_count, 4, "No additional recompilations")

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_multi_dim_dynamic_and_semantics(self):
        """
        When multiple dims become dynamic at once, AND semantics is critical.
        Graph 0: (4, 3, 234, 5) static
        Graph 1: (s0, 3, s2, s3) dynamic on dims 0,2,3. excluded=(4, _, 234, 5)

        OR semantics (wrong): rejects (4, 3, 100, 20) because dim0==4 matches.
        AND semantics (correct): accepts (4, 3, 100, 20) because not ALL excluded
        dims match (dim2=100≠234 and dim3=20≠5).
        """

        def foo(x):
            return x * 2

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        opt(torch.randn(4, 3, 234, 5))  # Graph 0: static
        opt(torch.randn(10, 3, 100, 20))  # Graph 1: dims 0,2,3 dynamic
        self.assertEqual(tracker.frame_count, 2)

        # Only the exact original shape should be excluded from Graph 1
        opt(torch.randn(4, 3, 234, 5))
        self.assertEqual(tracker.call_log[-1], 0, "Exact original -> Graph 0")

        # Partial matches should NOT be excluded (AND semantics)
        opt(torch.randn(4, 3, 100, 20))  # dim0=4 matches, dims 2,3 don't
        self.assertEqual(tracker.call_log[-1], 1, "dim0=4 partial match -> Graph 1")

        opt(torch.randn(10, 3, 234, 20))  # dim2=234 matches, dims 0,3 don't
        self.assertEqual(tracker.call_log[-1], 1, "dim2=234 partial match -> Graph 1")

        opt(torch.randn(10, 3, 234, 5))  # dim2=234, dim3=5 match, dim0 doesn't
        self.assertEqual(tracker.call_log[-1], 1, "dim2+dim3 partial match -> Graph 1")

        opt(torch.randn(4, 3, 234, 20))  # dim0=4, dim2=234 match, dim3 doesn't
        self.assertEqual(tracker.call_log[-1], 1, "dim0+dim2 partial match -> Graph 1")

        # Totally new shape, no exclusion hit
        opt(torch.randn(99, 3, 88, 77))
        self.assertEqual(tracker.call_log[-1], 1, "New shape -> Graph 1")

        self.assertEqual(tracker.frame_count, 2, "No additional recompilations")


if __name__ == "__main__":
    run_tests()
