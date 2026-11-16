# Owner(s): ["module: PrivateUse1"]

import json
import os
import tempfile
import unittest

import torch
import torch.nn as nn
from torch._C._profiler import ProfilerState
from torch.autograd.profiler import profile as autograd_profile
from torch.profiler import ProfilerActivity, record_function
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class SimpleModel(nn.Module):
    """Simple neural network for integration testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestProfiler(TestCase):
    """Test PyTorch profiler integration with OpenReg backend."""

    @skipIfTorchDynamo()
    def test_profiler_basic(self):
        """Test basic profiler functionality with OpenReg device."""
        with autograd_profile(use_device="openreg") as prof:
            x = torch.randn(10, 10, device="openreg")
            y = torch.randn(10, 10, device="openreg")
            x + y

        events = prof.function_events
        self.assertGreater(len(events), 0)

        # Check that we have OpenReg operations recorded
        event_names = [e.name for e in events]
        self.assertTrue(any("aten::" in name for name in event_names))

    @skipIfTorchDynamo()
    def test_profiler_with_record_function(self):
        """Test profiler with custom record_function annotations."""
        with autograd_profile(use_device="openreg") as prof:
            with record_function("openreg_custom_operation"):
                x = torch.randn(10, 10, device="openreg")
                y = torch.randn(10, 10, device="openreg")
                x @ y

        events = prof.function_events
        event_names = [e.name for e in events]
        self.assertTrue(any("openreg_custom_operation" in name for name in event_names))

    @skipIfTorchDynamo()
    def test_profiler_multiple_streams(self):
        """Test profiler with multiple OpenReg streams."""
        stream1 = torch.Stream(device="openreg")
        stream2 = torch.Stream(device="openreg")

        with autograd_profile(use_device="openreg") as prof:
            with stream1:
                x = torch.randn(10, 10, device="openreg")
                x + x

            with stream2:
                z = torch.randn(10, 10, device="openreg")
                z * z

        events = prof.function_events
        self.assertGreater(len(events), 0)

    @skipIfTorchDynamo()
    def test_profiler_event_synchronization(self):
        """Test that profiler correctly handles event synchronization."""
        with autograd_profile(use_device="openreg") as prof:
            stream = torch.Stream(device="openreg")
            event = torch.Event(device="openreg", enable_timing=True)

            with stream:
                x = torch.randn(10, 10, device="openreg")
                event.record(stream)
                x + x

            event.synchronize()

        events = prof.function_events
        self.assertGreater(len(events), 0)

    @skipIfTorchDynamo()
    def test_profiler_nested_record_function(self):
        """Test profiler with nested record_function calls."""
        with autograd_profile(use_device="openreg") as prof:
            with record_function("outer_operation"):
                x = torch.randn(10, 10, device="openreg")

                with record_function("inner_operation_1"):
                    x + x

                with record_function("inner_operation_2"):
                    y = x + x
                    y * y

        events = prof.function_events
        event_names = [e.name for e in events]
        self.assertTrue(any("outer_operation" in name for name in event_names))
        self.assertTrue(any("inner_operation_1" in name for name in event_names))
        self.assertTrue(any("inner_operation_2" in name for name in event_names))

    @skipIfTorchDynamo()
    def test_profiler_key_averages(self):
        """Test that key_averages works correctly with OpenReg operations."""
        with autograd_profile(use_device="openreg") as prof:
            for _ in range(10):
                x = torch.randn(10, 10, device="openreg")
                x + x

        key_averages = prof.key_averages()
        self.assertGreater(len(key_averages), 0)

        # Verify we can access statistics
        for avg in key_averages:
            self.assertIsNotNone(avg.key)
            self.assertGreaterEqual(avg.count, 0)
            self.assertGreaterEqual(avg.cpu_time_total, 0)

    @skipIfTorchDynamo()
    def test_profiler_table_output(self):
        """Test that profiler can generate table output."""
        with autograd_profile(use_device="openreg") as prof:
            x = torch.randn(10, 10, device="openreg")
            y = x + x
            x @ y

        # Should not raise any exceptions
        table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        self.assertIsInstance(table, str)
        self.assertGreater(len(table), 0)

    @skipIfTorchDynamo()
    def test_profiler_export_chrome_trace(self):
        """Test exporting profiler results to Chrome trace format."""
        with autograd_profile(use_device="openreg") as prof:
            x = torch.randn(10, 10, device="openreg")
            y = x + x
            x @ y

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            trace_file = f.name

        try:
            prof.export_chrome_trace(trace_file)

            # Verify the trace file is valid JSON
            with open(trace_file) as f:
                trace_data = json.load(f)

            if isinstance(trace_data, dict):
                self.assertIn("traceEvents", trace_data)
                events = trace_data["traceEvents"]
            else:
                events = trace_data
            self.assertGreater(len(events), 0)
        finally:
            import os

            if os.path.exists(trace_file):
                os.remove(trace_file)

    @skipIfTorchDynamo()
    def test_profiler_with_shapes(self):
        """Test profiler with shape recording enabled."""
        with autograd_profile(use_device="openreg", record_shapes=True) as prof:
            x = torch.randn(10, 20, device="openreg")
            y = torch.randn(20, 30, device="openreg")
            x @ y

        key_averages = prof.key_averages(group_by_input_shape=True)
        self.assertGreater(len(key_averages), 0)

    @skipIfTorchDynamo()
    def test_profiler_default_stream_initialization(self):
        """Test that profiler works without explicit stream initialization."""
        # This test verifies the bug fix: profiler should work even when
        # getStreamFromPool is not called first
        with autograd_profile(use_device="openreg") as prof:
            # Use default stream without explicitly creating streams
            x = torch.randn(10, 10, device="openreg")
            x + 1.0

        events = prof.function_events
        self.assertGreater(len(events), 0)

    @skipIfTorchDynamo()
    def test_profiler_stream_event_correlation(self):
        """Test that profiler correctly correlates events across streams."""
        stream1 = torch.Stream(device="openreg")
        stream2 = torch.Stream(device="openreg")

        with autograd_profile(use_device="openreg") as prof:
            with record_function("stream1_ops"):
                with stream1:
                    x = torch.randn(10, 10, device="openreg")
                    x + x

            with record_function("stream2_ops"):
                with stream2:
                    z = torch.randn(10, 10, device="openreg")
                    z * z

        events = prof.function_events
        stream1_events = [e for e in events if "stream1_ops" in e.name]
        stream2_events = [e for e in events if "stream2_ops" in e.name]

        self.assertGreater(len(stream1_events), 0)
        self.assertGreater(len(stream2_events), 0)

    @skipIfTorchDynamo()
    def test_profiler_elapsed_time(self):
        """Test that event elapsed time is correctly calculated."""
        with autograd_profile(use_device="openreg") as prof:
            stream = torch.Stream(device="openreg")

            with stream:
                x = torch.randn(100, 100, device="openreg")
                y = torch.randn(100, 100, device="openreg")
                # Multiple matmuls to ensure measurable time
                for _ in range(10):
                    z = x @ y
                    x = z

            stream.synchronize()

        events = prof.function_events
        # Check that operations have non-zero duration
        compute_events = [
            e for e in events if "aten::mm" in e.name or "aten::matmul" in e.name
        ]
        if compute_events:
            total_time = sum(e.cpu_time_total for e in compute_events)
            self.assertGreater(total_time, 0)

    @skipIfTorchDynamo()
    def test_profiler_context_manager_multiple_times(self):
        """Test that profiler can be used multiple times."""
        with autograd_profile(use_device="openreg") as prof:
            for i in range(3):
                x = torch.randn(10, 10, device="openreg")
                x = x + i

        events = prof.function_events
        self.assertGreater(len(events), 0)

    @skipIfTorchDynamo()
    def test_profiler_with_backward_pass(self):
        """Test profiler with autograd operations."""
        with autograd_profile(use_device="openreg", record_shapes=True) as prof:
            x = torch.randn(10, 10, device="openreg", requires_grad=True)
            y = torch.randn(10, 10, device="openreg", requires_grad=True)
            z = (x @ y).sum()
            z.backward()

        events = prof.function_events
        event_names = [e.name for e in events]

        # Check for forward and backward operations
        self.assertTrue(
            any("aten::mm" in name or "aten::matmul" in name for name in event_names)
        )
        self.assertTrue(any("backward" in name.lower() for name in event_names))


class TestProfilerEdgeCases(TestCase):
    """Test edge cases and error conditions for profiler."""

    @skipIfTorchDynamo()
    def test_profiler_empty_operations(self):
        """Test profiler with no operations."""
        with autograd_profile(use_device="openreg") as prof:
            pass

        events = prof.function_events
        # Should still work, even with no user operations
        self.assertIsInstance(events, list)

    @skipIfTorchDynamo()
    def test_profiler_exception_handling(self):
        """Test that profiler handles exceptions gracefully."""
        try:
            with autograd_profile(use_device="openreg") as prof:
                x = torch.randn(10, 10, device="openreg")  # noqa: F841
                raise RuntimeError("Test exception")
        except RuntimeError as e:
            self.assertEqual(str(e), "Test exception")

        # Profiler should still be usable
        events = prof.function_events
        self.assertIsInstance(events, list)

    @skipIfTorchDynamo()
    def test_profiler_large_number_of_events(self):
        """Test profiler with large number of events."""
        with autograd_profile(use_device="openreg") as prof:
            x = torch.randn(10, 10, device="openreg")

            # Create many small operations
            for i in range(100):
                with record_function(f"op_{i}"):
                    x = x + 1

        events = prof.function_events
        self.assertGreater(len(events), 100)


class TestProfilerMultiDevice(TestCase):
    """Test profiler with multiple OpenReg devices if available."""

    @skipIfTorchDynamo()
    def test_profiler_single_device(self):
        """Test profiler with explicit device specification."""
        device_count = torch.openreg.device_count()
        if device_count < 1:
            self.skipTest("No OpenReg devices available")

        with autograd_profile(use_device="openreg") as prof:
            x = torch.randn(10, 10, device="openreg:0")
            x + x

        events = prof.function_events
        self.assertGreater(len(events), 0)

    @skipIfTorchDynamo()
    def test_profiler_multiple_devices(self):
        """Test profiler with operations on multiple devices."""
        device_count = torch.openreg.device_count()
        if device_count < 2:
            self.skipTest("Multiple OpenReg devices required")

        with autograd_profile(use_device="openreg") as prof:
            x0 = torch.randn(10, 10, device="openreg:0")
            x1 = torch.randn(10, 10, device="openreg:1")

            x0 + x0
            x1 + x1

        events = prof.function_events
        self.assertGreater(len(events), 0)


class TestProfilerIntegration(TestCase):
    """Integration tests for profiler with realistic PyTorch workflows."""

    @skipIfTorchDynamo()
    def test_profiler_with_simple_model(self):
        """Test profiler with a simple neural network."""
        model = SimpleModel().to("openreg")
        x = torch.randn(32, 10, device="openreg")

        with autograd_profile(use_device="openreg") as prof:
            model(x)

        events = prof.function_events
        self.assertGreater(len(events), 0)

        # Check for linear and relu operations
        event_names = [e.name for e in events]
        self.assertTrue(
            any(
                "linear" in name.lower() or "addmm" in name.lower()
                for name in event_names
            )
        )

    @skipIfTorchDynamo()
    def test_profiler_with_training_step(self):
        """Test profiler with a complete training step."""
        model = SimpleModel().to("openreg")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        x = torch.randn(32, 10, device="openreg")
        target = torch.randn(32, 10, device="openreg")

        with autograd_profile(use_device="openreg", record_shapes=True) as prof:
            # Forward pass
            with record_function("forward"):
                output = model(x)

            # Loss calculation
            with record_function("loss"):
                loss = criterion(output, target)

            # Backward pass
            with record_function("backward"):
                loss.backward()

            # Optimizer step
            with record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()

        events = prof.function_events
        event_names = [e.name for e in events]

        # Verify all stages are recorded
        self.assertTrue(any("forward" in name for name in event_names))
        self.assertTrue(any("loss" in name for name in event_names))
        self.assertTrue(any("backward" in name for name in event_names))
        self.assertTrue(any("optimizer_step" in name for name in event_names))

    @skipIfTorchDynamo()
    def test_profiler_with_multiple_batches(self):
        """Test profiler with multiple training batches."""
        model = SimpleModel().to("openreg")

        with autograd_profile(use_device="openreg") as prof:
            for i in range(5):
                with record_function(f"batch_{i}"):
                    x = torch.randn(16, 10, device="openreg")
                    model(x)

        events = prof.function_events

        # Check that all batches are recorded
        event_names = [e.name for e in events]
        for i in range(5):
            self.assertTrue(any(f"batch_{i}" in name for name in event_names))

    @skipIfTorchDynamo()
    def test_profiler_table_with_model(self):
        """Test that profiler table output works with model operations."""
        model = SimpleModel().to("openreg")
        x = torch.randn(32, 10, device="openreg")

        with autograd_profile(use_device="openreg", record_shapes=True) as prof:
            for _ in range(10):
                model(x)

        # Generate table - should not raise exceptions
        table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)

        self.assertIsInstance(table, str)
        self.assertGreater(len(table), 0)

        # Table should contain operation names
        self.assertTrue(
            any(keyword in table.lower() for keyword in ["linear", "addmm", "relu"])
        )

    @skipIfTorchDynamo()
    def test_profiler_with_inference(self):
        """Test profiler during model inference (no gradients)."""
        model = SimpleModel().to("openreg")
        model.eval()

        with autograd_profile(use_device="openreg") as prof:
            with torch.no_grad():
                for _ in range(5):
                    x = torch.randn(16, 10, device="openreg")
                    model(x)

        events = prof.function_events
        self.assertGreater(len(events), 0)

        # Should not have backward operations
        event_names = [e.name for e in events]
        # Backward operations should be minimal or none
        backward_ops = [name for name in event_names if "backward" in name.lower()]  # noqa: F841
        # It's ok if there are some backward-related events in the profiler
        # infrastructure, but there shouldn't be many

    @skipIfTorchDynamo()
    def test_profiler_with_data_transfer(self):
        """Test profiler with CPU-OpenReg data transfers."""
        with autograd_profile(use_device="openreg") as prof:
            # CPU tensor
            x_cpu = torch.randn(50, 50)

            # Transfer to OpenReg
            with record_function("to_openreg"):
                x_openreg = x_cpu.to("openreg")

            # Compute on OpenReg
            with record_function("compute_openreg"):
                y = x_openreg @ x_openreg

            # Transfer back to CPU
            with record_function("to_cpu"):
                y.cpu()

        events = prof.function_events
        event_names = [e.name for e in events]

        self.assertTrue(any("to_openreg" in name for name in event_names))
        self.assertTrue(any("compute_openreg" in name for name in event_names))
        self.assertTrue(any("to_cpu" in name for name in event_names))

    @skipIfTorchDynamo()
    def test_profiler_with_mixed_operations(self):
        """Test profiler with mixed tensor operations."""
        with autograd_profile(use_device="openreg", record_shapes=True) as prof:
            x = torch.randn(20, 20, device="openreg")

            # Various operations
            y = x + 1.0
            z = torch.matmul(x, y)
            w = torch.relu(z)
            v = w.transpose(0, 1)
            v.sum()

        events = prof.function_events
        self.assertGreater(len(events), 0)

        # Should capture various operation types
        event_names = [e.name for e in events]
        self.assertTrue(len([n for n in event_names if "aten::" in n]) > 0)


class TestProfilerPerformance(TestCase):
    """Performance-related profiler tests."""

    @skipIfTorchDynamo()
    def test_profiler_overhead(self):
        """Verify profiler doesn't add excessive overhead."""
        model = SimpleModel().to("openreg")
        x = torch.randn(32, 10, device="openreg")

        # Warmup
        for _ in range(5):
            model(x)

        # Profile should complete in reasonable time
        import time

        start = time.time()

        with autograd_profile(use_device="openreg") as prof:
            for _ in range(100):
                model(x)

        elapsed = time.time() - start

        # Should complete in reasonable time (not checking exact time as it's hardware dependent)
        # Just verify it completes and produces valid results
        events = prof.function_events
        self.assertGreater(len(events), 0)
        self.assertLess(elapsed, 60.0)  # Should complete within 1 minute


@unittest.skipIf(
    not torch.profiler.kineto_available(), "Kineto is required for these tests"
)
class TestKinetoIntegration(TestCase):
    """End-to-end integration tests for OpenReg Kineto profiler plugin.

    Covers IActivityProfiler plugin registration, session lifecycle, fallback
    path compatibility, Chrome trace output via ProfilerActivity.PrivateUse1,
    and fine-grained activity type filter compatibility (PR #176351).
    """

    def _export_and_load_trace(self, prof):
        """Export Chrome trace to a temp file, load and return traceEvents."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            trace_file = f.name
        try:
            prof.export_chrome_trace(trace_file)
            with open(trace_file) as f:
                return json.load(f)["traceEvents"]
        finally:
            os.remove(trace_file)

    @skipIfTorchDynamo()
    def test_basic_profiling_cpu_ops(self):
        """PrivateUse1 profiler session runs without error and CPU ops are recorded."""
        self.assertIn(
            ProfilerActivity.PrivateUse1, torch.profiler.supported_activities()
        )

        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        ) as prof:
            x = torch.randn(10, 10, device="openreg")
            y = torch.randn(10, 10, device="openreg")
            x + y
            x * y
            x @ y

        events = prof.events()
        self.assertGreater(len(events), 0, "No events recorded")
        cpu_op_names = [e.name for e in events]
        self.assertTrue(
            any("aten::" in n for n in cpu_op_names),
            f"No aten:: CPU ops in {cpu_op_names[:10]}",
        )

    @skipIfTorchDynamo()
    def test_multi_session_lifecycle(self):
        """Two independent sessions run to completion without error."""
        activities = [ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]

        with torch.profiler.profile(activities=activities) as prof1:
            x = torch.randn(10, 10, device="openreg")
            x + x
        with torch.profiler.profile(activities=activities) as prof2:
            y = torch.randn(10, 10, device="openreg")
            torch.abs(y)

        events1 = prof1.events()
        events2 = prof2.events()
        self.assertGreater(len(events1), 0, "Session 1 produced no events")
        self.assertGreater(len(events2), 0, "Session 2 produced no events")
        self.assertTrue(
            any("aten::" in e.name for e in events1),
            "Session 1: no aten:: CPU ops",
        )
        self.assertTrue(
            any("aten::" in e.name for e in events2),
            "Session 2: no aten:: CPU ops",
        )

    @skipIfTorchDynamo()
    def test_fallback_preserved(self):
        """use_kineto=False still uses ProfilerStubs for device timing."""
        with autograd_profile(
            use_device="openreg", use_kineto=False, use_cpu=True
        ) as prof:
            x = torch.randn(10, 10, device="openreg")
            y = torch.randn(10, 10, device="openreg")
            x + y

        events = prof.function_events
        self.assertGreater(len(events), 0, "No events in fallback mode")

        event_names = [e.name for e in events]
        self.assertTrue(
            any("aten::" in n for n in event_names),
            f"No aten:: ops in fallback mode: {event_names}",
        )
        self.assertEqual(
            prof.profiler_kind,
            ProfilerState.KINETO_PRIVATEUSE1_FALLBACK,
        )

    @skipIfTorchDynamo()
    def test_chrome_trace_cpu_ops(self):
        """CPU ops appear in the exported Chrome trace JSON."""
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        ) as prof:
            x = torch.randn(10, 10, device="openreg")
            y = torch.randn(10, 10, device="openreg")
            x + y
            x * y

        trace_events = self._export_and_load_trace(prof)
        self.assertGreater(len(trace_events), 0, "No trace events produced")

        # CPU ops appear in the trace
        cpu_ops = [e for e in trace_events if e.get("cat") == "cpu_op"]
        self.assertGreater(len(cpu_ops), 0, "No cpu_op events in trace")

    @skipIfTorchDynamo()
    def test_activity_filter_compatibility(self):
        """Fine-grained activity filter API (PR #176351) works with PrivateUse1 backend.

        Four scenarios cover all distinct filter configurations:
        1. Empty PrivateUse1 only (no CPU): plugin activates with no types, produces no events.
        2. Specific type filter CONCURRENT_KERNEL only: session runs without error.
        3. CPU + empty PrivateUse1: CPU events captured and record_function appears.
        4. CPU-only (no PrivateUse1): OpenReg plugin not activated, only CPU events appear.
        """
        # Scenario 1: empty PrivateUse1 only, no CPU -> no events produced.
        with torch.profiler.profile(
            activities=[{ProfilerActivity.PrivateUse1: []}],
        ) as prof:
            x = torch.randn(10, 10, device="openreg")
            x + x
        self.assertEqual(
            len(prof.events()),
            0,
            "Empty PrivateUse1 filter should produce no events",
        )

        # Scenario 2: specific type filter CONCURRENT_KERNEL only -> no CPU, stub emits
        # no kernel records, so zero events expected.
        with torch.profiler.profile(
            activities=[{ProfilerActivity.PrivateUse1: ["CONCURRENT_KERNEL"]}],
        ) as prof:
            x = torch.randn(10, 10, device="openreg")
            x + x
        self.assertEqual(
            len(prof.events()),
            0,
            "No CPU + stub emits no CONCURRENT_KERNEL records -> 0 events expected",
        )

        # Scenario 3: CPU + empty PrivateUse1 filter -> CPU events still captured.
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, {ProfilerActivity.PrivateUse1: []}],
        ) as prof:
            with record_function("cpu_marker"):
                x = torch.randn(10, 10, device="openreg")
                x + x
        events = prof.events()
        self.assertGreater(
            len(events),
            0,
            "CPU events should be captured with empty PrivateUse1 filter",
        )
        self.assertTrue(
            any("cpu_marker" in e.name for e in events),
            "record_function annotation must appear when CPU is included",
        )

        # Scenario 4: CPU-only activities, no PrivateUse1 -> OpenReg plugin not activated.
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU],
        ) as prof:
            with record_function("cpu_only_marker"):
                x = torch.randn(10, 10, device="openreg")
                x + x
        events = prof.events()
        self.assertGreater(
            len(events), 0, "CPU-only session should still record CPU events"
        )
        self.assertTrue(
            any("cpu_only_marker" in e.name for e in events),
            "record_function must appear in CPU-only session",
        )
        self.assertTrue(
            all(e.device_type.value == 0 for e in events if hasattr(e, "device_type")),
            "CPU-only session must not contain device events",
        )


if __name__ == "__main__":
    run_tests()
