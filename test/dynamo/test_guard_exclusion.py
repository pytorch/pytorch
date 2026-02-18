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


if __name__ == "__main__":
    run_tests()
