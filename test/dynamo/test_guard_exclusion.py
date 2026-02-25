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
    def test_integer_input_exclusion_basic(self):
        """
        Integer inputs that become dynamic should also get exclusion guards.
        1. foo(x, 3) -> Graph 0: static n=3
        2. foo(x, 5) -> Graph 1: dynamic n, excluded should reject n==3
        3. foo(x, 3) -> should use Graph 0 (static), not Graph 1
        """

        def foo(x, n):
            return x * n

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        x = torch.randn(4)

        # Call 1: n=3 -> Graph 0 (static)
        opt(x, 3)
        self.assertEqual(tracker.frame_count, 1)
        self.assertEqual(tracker.call_log[-1], 0)

        # Call 2: n=5 -> Graph 1 (dynamic n)
        opt(x, 5)
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[-1], 1)

        # Call 3: n=3 -> should use Graph 0, not Graph 1
        opt(x, 3)
        self.assertEqual(
            tracker.call_log[-1],
            0,
            "Input n=3 should use Graph 0 (static), not Graph 1 (dynamic n). "
            "Integer inputs need exclusion guards too.",
        )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_integer_input_exclusion_accumulation(self):
        """
        Same accumulation scenario as the tensor test but with integer inputs.
        1. foo(x, 3, 4)  -> Graph 0: static (3, 4)
        2. foo(x, 5, 4)  -> Graph 1: dynamic (s0, 4), exclusion rejects n0==3
        3. foo(x, 3, 19) -> Graph 2: dynamic (s0, s1), exclusion should reject
                            n1==4 independently, not require n0==3 AND n1==4
        4. foo(x, 5, 4)  -> should use Graph 1, not Graph 2
        """

        def foo(x, n, m):
            return x * n + m

        tracker = GraphTracker()
        opt = torch.compile(foo, backend=tracker)

        x = torch.randn(4)

        # Call 1: (3, 4) -> Graph 0 (static)
        opt(x, 3, 4)
        self.assertEqual(tracker.frame_count, 1)
        self.assertEqual(tracker.call_log[-1], 0)

        # Call 2: (5, 4) -> Graph 1 (dynamic n)
        opt(x, 5, 4)
        self.assertEqual(tracker.frame_count, 2)
        self.assertEqual(tracker.call_log[-1], 1)

        # Call 3: (3, 19) -> Graph 2 (both dynamic)
        opt(x, 3, 19)
        self.assertEqual(tracker.frame_count, 3)
        self.assertEqual(tracker.call_log[-1], 2)

        # Call 4: (5, 4) -> should use Graph 1, not Graph 2
        opt(x, 5, 4)
        self.assertEqual(
            tracker.call_log[-1],
            1,
            "Input (5, 4) should use Graph 1 (s0, 4), not Graph 2 (s0, s1). "
            "Integer exclusion must reject m==4 independently.",
        )

        # Call 5: (3, 4) -> should use Graph 0 (static)
        opt(x, 3, 4)
        self.assertEqual(
            tracker.call_log[-1],
            0,
            "Input (3, 4) should use Graph 0 (static)",
        )


if __name__ == "__main__":
    run_tests()
