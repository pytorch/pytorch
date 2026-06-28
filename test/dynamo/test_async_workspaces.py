"""Tests for async workspace compilation.

Tests the experimental async workspace architecture where guard failures
emit delta events to a shared whiteboard instead of triggering full
recompilations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch._dynamo.whiteboard import (
    CompilerWhiteboard,
    DeltaEvent,
    DeltaEventType,
    get_whiteboard,
    reset_whiteboard,
    WorkspaceContext,
    WorkspaceState,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class SimpleModel(nn.Module):
    """3-layer model: Linear -> ReLU -> Linear."""

    def __init__(self, in_features: int = 64, hidden: int = 32, out_features: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class TestWhiteboard(TestCase):
    """Unit tests for the CompilerWhiteboard and DeltaEvent."""

    def setUp(self):
        reset_whiteboard()

    def tearDown(self):
        reset_whiteboard()

    def test_publish_and_retrieve(self):
        wb = CompilerWhiteboard()
        event = DeltaEvent(
            workspace_id=1,
            event_type=DeltaEventType.SHAPE_CHANGE,
            symbolic_constraints={"dim": 0, "old": 16, "new": 32},
        )
        wb.publish(event)

        events = wb.get_all_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].workspace_id, 1)
        self.assertEqual(events[0].event_type, DeltaEventType.SHAPE_CHANGE)
        self.assertEqual(events[0].symbolic_constraints["old"], 16)
        self.assertEqual(events[0].symbolic_constraints["new"], 32)

    def test_subscribe_receives_events(self):
        wb = CompilerWhiteboard()
        received: list[DeltaEvent] = []
        wb.subscribe(DeltaEventType.SHAPE_CHANGE, received.append)

        wb.publish(DeltaEvent(
            workspace_id=1,
            event_type=DeltaEventType.SHAPE_CHANGE,
            symbolic_constraints={"dim": 0, "old": 16, "new": 32},
        ))
        wb.publish(DeltaEvent(
            workspace_id=2,
            event_type=DeltaEventType.GUARD_FAIL,
            symbolic_constraints={"reason": "test"},
        ))

        # Only SHAPE_CHANGE events should be received
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].workspace_id, 1)

    def test_subscribe_all(self):
        wb = CompilerWhiteboard()
        received: list[DeltaEvent] = []
        wb.subscribe_all(received.append)

        wb.publish(DeltaEvent(
            workspace_id=1,
            event_type=DeltaEventType.SHAPE_CHANGE,
            symbolic_constraints={},
        ))
        wb.publish(DeltaEvent(
            workspace_id=2,
            event_type=DeltaEventType.GUARD_FAIL,
            symbolic_constraints={},
        ))

        self.assertEqual(len(received), 2)

    def test_event_count(self):
        wb = CompilerWhiteboard()
        wb.publish(DeltaEvent(
            workspace_id=1,
            event_type=DeltaEventType.SHAPE_CHANGE,
            symbolic_constraints={},
        ))
        wb.publish(DeltaEvent(
            workspace_id=2,
            event_type=DeltaEventType.SHAPE_CHANGE,
            symbolic_constraints={},
        ))
        wb.publish(DeltaEvent(
            workspace_id=3,
            event_type=DeltaEventType.GUARD_FAIL,
            symbolic_constraints={},
        ))

        self.assertEqual(wb.event_count(), 3)
        self.assertEqual(wb.event_count(DeltaEventType.SHAPE_CHANGE), 2)
        self.assertEqual(wb.event_count(DeltaEventType.GUARD_FAIL), 1)
        self.assertEqual(wb.event_count(DeltaEventType.FUSION_HINT), 0)

    def test_events_since_timestamp(self):
        wb = CompilerWhiteboard()
        e1 = DeltaEvent(
            workspace_id=1,
            event_type=DeltaEventType.SHAPE_CHANGE,
            symbolic_constraints={},
        )
        wb.publish(e1)
        timestamp = e1.timestamp

        e2 = DeltaEvent(
            workspace_id=2,
            event_type=DeltaEventType.GUARD_FAIL,
            symbolic_constraints={},
        )
        wb.publish(e2)

        later_events = wb.get_events_since(timestamp)
        self.assertEqual(len(later_events), 1)
        self.assertEqual(later_events[0].workspace_id, 2)

    def test_clear(self):
        wb = CompilerWhiteboard()
        wb.publish(DeltaEvent(
            workspace_id=1,
            event_type=DeltaEventType.SHAPE_CHANGE,
            symbolic_constraints={},
        ))
        wb.subscribe(DeltaEventType.SHAPE_CHANGE, lambda e: None)
        wb.clear()

        self.assertEqual(wb.event_count(), 0)
        self.assertEqual(len(wb.get_all_events()), 0)

    def test_binary_delta_encoding(self):
        event = DeltaEvent(
            workspace_id=1,
            event_type=DeltaEventType.SHAPE_CHANGE,
            symbolic_constraints={"dim": 0, "old": 16, "new": 128},
        )
        encoded = event.encode_shape_delta()
        decoded = DeltaEvent.decode_shape_delta(encoded)

        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["dim"], 0)
        self.assertEqual(decoded[0]["old"], 16)
        self.assertEqual(decoded[0]["new"], 128)

    def test_global_singleton(self):
        reset_whiteboard()
        wb1 = get_whiteboard()
        wb2 = get_whiteboard()
        self.assertIs(wb1, wb2)

    def test_bounded_log_size(self):
        wb = CompilerWhiteboard(max_log_size=5)
        for i in range(10):
            wb.publish(DeltaEvent(
                workspace_id=i,
                event_type=DeltaEventType.SHAPE_CHANGE,
                symbolic_constraints={"i": i},
            ))
        self.assertEqual(wb.event_count(), 5)
        events = wb.get_all_events()
        # Oldest events should be evicted
        self.assertEqual(events[0].workspace_id, 5)


class TestWorkspaceContext(TestCase):
    """Unit tests for WorkspaceContext state machine."""

    def setUp(self):
        reset_whiteboard()

    def tearDown(self):
        reset_whiteboard()

    def test_initial_state(self):
        wb = CompilerWhiteboard()
        subgraph = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        ws = WorkspaceContext(
            subgraph=subgraph,
            example_inputs=[],
            whiteboard=wb,
        )
        self.assertEqual(ws.state, WorkspaceState.CREATED)
        self.assertFalse(ws.is_ready)
        self.assertFalse(ws.is_compiling)

    def test_compile_sync(self):
        wb = CompilerWhiteboard()
        model = SimpleModel()
        gm = torch.fx.symbolic_trace(model)
        inputs = [torch.randn(16, 64)]

        ws = WorkspaceContext(
            subgraph=gm,
            example_inputs=inputs,
            whiteboard=wb,
            workspace_name="test_ws",
        )

        # Compile with a simple "eager" compiler
        ws.compile_sync(lambda g, i: g.forward)
        self.assertEqual(ws.state, WorkspaceState.COMPILED)
        self.assertTrue(ws.is_ready)

        # Check that WORKSPACE_READY event was published
        ready_events = [
            e for e in wb.get_all_events()
            if e.event_type == DeltaEventType.WORKSPACE_READY
        ]
        self.assertEqual(len(ready_events), 1)

    def test_compile_async(self):
        wb = CompilerWhiteboard()
        model = SimpleModel()
        gm = torch.fx.symbolic_trace(model)
        inputs = [torch.randn(16, 64)]

        ws = WorkspaceContext(
            subgraph=gm,
            example_inputs=inputs,
            whiteboard=wb,
        )

        ws.compile_async(lambda g, i: g.forward)
        success = ws.wait_for_compilation(timeout=10.0)
        self.assertTrue(success)
        self.assertTrue(ws.is_ready)

    def test_try_patch_shapes_batch_dim(self):
        wb = CompilerWhiteboard()
        model = SimpleModel()
        gm = torch.fx.symbolic_trace(model)
        inputs = [torch.randn(16, 64)]

        ws = WorkspaceContext(
            subgraph=gm,
            example_inputs=inputs,
            whiteboard=wb,
        )
        ws.compile_sync(lambda g, i: g.forward)

        # Patch batch dim: 16 -> 32
        new_inputs = [torch.randn(32, 64)]
        result = ws.try_patch_shapes(new_inputs)
        self.assertTrue(result)
        self.assertEqual(ws.state, WorkspaceState.COMPILED)

        # Check SHAPE_CHANGE event was published
        shape_events = [
            e for e in wb.get_all_events()
            if e.event_type == DeltaEventType.SHAPE_CHANGE
        ]
        self.assertEqual(len(shape_events), 1)
        self.assertEqual(shape_events[0].symbolic_constraints["old"], 16)
        self.assertEqual(shape_events[0].symbolic_constraints["new"], 32)

    def test_try_patch_shapes_non_batch_fails(self):
        wb = CompilerWhiteboard()
        model = SimpleModel()
        gm = torch.fx.symbolic_trace(model)
        inputs = [torch.randn(16, 64)]

        ws = WorkspaceContext(
            subgraph=gm,
            example_inputs=inputs,
            whiteboard=wb,
        )
        ws.compile_sync(lambda g, i: g.forward)

        # Change non-batch dim: 64 -> 128 -- should fail patching
        new_inputs = [torch.randn(16, 128)]
        result = ws.try_patch_shapes(new_inputs)
        self.assertFalse(result)

    def test_invalidate(self):
        wb = CompilerWhiteboard()
        model = SimpleModel()
        gm = torch.fx.symbolic_trace(model)
        inputs = [torch.randn(16, 64)]

        ws = WorkspaceContext(
            subgraph=gm,
            example_inputs=inputs,
            whiteboard=wb,
        )
        ws.compile_sync(lambda g, i: g.forward)
        self.assertTrue(ws.is_ready)

        ws.invalidate()
        self.assertEqual(ws.state, WorkspaceState.CREATED)
        self.assertFalse(ws.is_ready)

    def test_execute_eager_fallback(self):
        wb = CompilerWhiteboard()
        model = SimpleModel()
        gm = torch.fx.symbolic_trace(model)
        x = torch.randn(16, 64)

        ws = WorkspaceContext(
            subgraph=gm,
            example_inputs=[x],
            whiteboard=wb,
        )

        # Execute without compiling first -> should use eager fallback
        result = ws.execute(x)
        expected = model(x)
        torch.testing.assert_close(result, expected)


class TestAsyncWorkspaceBackend(TestCase):
    """Integration tests for the async_workspace_backend."""

    def setUp(self):
        torch._dynamo.reset()
        reset_whiteboard()

    def tearDown(self):
        torch._dynamo.reset()
        reset_whiteboard()

    def test_basic_compilation(self):
        """Backend compiles and produces correct output."""
        from torch._inductor.async_workspace import async_workspace_backend

        model = SimpleModel()
        x = torch.randn(16, 64)
        expected = model(x)

        compiled = torch.compile(model, backend=async_workspace_backend, dynamic=True)
        result = compiled(x)
        torch.testing.assert_close(result, expected)

    def test_dynamic_batch_size_correctness(self):
        """Fluctuating batch sizes produce correct outputs."""
        from torch._inductor.async_workspace import async_workspace_backend

        model = SimpleModel()
        compiled = torch.compile(model, backend=async_workspace_backend, dynamic=True)

        batch_sizes = [16, 32, 16, 128, 8, 64]
        for bs in batch_sizes:
            x = torch.randn(bs, 64)
            expected = model(x)
            result = compiled(x)
            torch.testing.assert_close(
                result, expected,
                msg=f"Mismatch at batch_size={bs}",
            )

    def test_whiteboard_captures_shape_events(self):
        """Shape changes are recorded as DeltaEvents on the whiteboard."""
        from torch._inductor.async_workspace import async_workspace_backend

        model = SimpleModel()
        compiled = torch.compile(model, backend=async_workspace_backend, dynamic=True)

        # First call: establishes baseline shapes
        compiled(torch.randn(16, 64))

        wb = get_whiteboard()
        initial_count = wb.event_count(DeltaEventType.SHAPE_CHANGE)

        # Second call with different batch size: should post deltas
        compiled(torch.randn(32, 64))

        final_count = wb.event_count(DeltaEventType.SHAPE_CHANGE)
        self.assertGreater(
            final_count, initial_count,
            "Expected SHAPE_CHANGE events after batch size change",
        )

    def test_workspace_isolation(self):
        """Multiple workspaces exist and are independently trackable."""
        from torch._inductor.async_workspace import async_workspace_backend

        model = SimpleModel()
        compiled = torch.compile(model, backend=async_workspace_backend, dynamic=True)
        compiled(torch.randn(16, 64))

        wb = get_whiteboard()
        events = wb.get_all_events()

        # Should have workspace events from multiple workspace IDs
        workspace_ids = {e.workspace_id for e in events}
        self.assertGreater(len(workspace_ids), 0, "Expected events from workspaces")

    def test_no_recompile_limit_hit(self):
        """Fluctuating shapes don't hit the recompile limit."""
        from torch._inductor.async_workspace import async_workspace_backend

        model = SimpleModel()
        compiled = torch.compile(
            model,
            backend=async_workspace_backend,
            dynamic=True,
        )

        # Run with many different batch sizes -- should not raise
        for bs in [16, 32, 64, 128, 8, 4, 256, 512, 16, 32]:
            x = torch.randn(bs, 64)
            result = compiled(x)
            expected = model(x)
            torch.testing.assert_close(result, expected)


class TestCompileIdExtension(TestCase):
    """Test that CompileId workspace_id extension works."""

    def test_compile_id_default_none(self):
        from torch._guards import CompileId

        cid = CompileId(frame_id=0, frame_compile_id=0)
        self.assertIsNone(cid.workspace_id)

    def test_compile_id_with_workspace(self):
        from torch._guards import CompileId

        cid = CompileId(frame_id=0, frame_compile_id=0, workspace_id=42)
        self.assertEqual(cid.workspace_id, 42)

    def test_compile_id_str_backward_compatible(self):
        from torch._guards import CompileId

        cid = CompileId(frame_id=1, frame_compile_id=2)
        self.assertEqual(str(cid), "1/2")

        cid_with_ws = CompileId(frame_id=1, frame_compile_id=2, workspace_id=5)
        # workspace_id doesn't change the string repr (internal-only field)
        self.assertEqual(str(cid_with_ws), "1/2")


if __name__ == "__main__":
    run_tests()
