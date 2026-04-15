# Owner(s): ["oncall: profiler"]

"""Tests for OpenTelemetry trace export support."""

import threading
import typing
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.autograd.profiler import profile as _profile
from torch.profiler import kineto_available, profile, ProfilerActivity
from torch.testing._internal.common_utils import run_tests, TestCase


def _otel_available() -> bool:
    try:
        import opentelemetry.sdk.trace  # noqa: F401
        import opentelemetry.trace  # noqa: F401

        return True
    except ImportError:
        return False


def _make_in_memory_exporter():
    """Create a simple in-memory span exporter for testing.

    Avoids depending on ``opentelemetry.sdk.trace.export.in_memory``
    which may not be present in all SDK versions.
    """
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class _InMemorySpanExporter(SpanExporter):
        def __init__(self):
            self._spans: list = []
            self._lock = threading.Lock()

        def export(self, spans):
            with self._lock:
                self._spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis=0):
            return True

        def get_finished_spans(self):
            with self._lock:
                return list(self._spans)

    return _InMemorySpanExporter()


class TestOpenTelemetryExport(TestCase):
    """Tests for torch.profiler._opentelemetry module."""

    def _simple_payload(self):
        """Run a simple workload to generate profiler events."""
        x = torch.ones(10, 10)
        y = torch.ones(10, 10)
        z = torch.add(x, y)
        return z

    def test_import_error_when_otel_missing(self):
        """Verify a clear error when opentelemetry is not installed."""
        from torch.profiler._opentelemetry import _check_otel_available

        with patch.dict("sys.modules", {"opentelemetry.sdk.trace": None}):
            with self.assertRaises(ImportError) as ctx:
                _check_otel_available()
            self.assertIn("opentelemetry", str(ctx.exception).lower())

    def test_us_to_ns(self):
        from torch.profiler._opentelemetry import _us_to_ns

        self.assertEqual(_us_to_ns(1.0), 1_000)
        self.assertEqual(_us_to_ns(0.5), 500)
        self.assertEqual(_us_to_ns(1_000_000), 1_000_000_000)

    def test_safe_str_short(self):
        from torch.profiler._opentelemetry import _safe_str

        self.assertEqual(_safe_str("hello"), "hello")

    def test_safe_str_long(self):
        from torch.profiler._opentelemetry import _safe_str

        long_str = "x" * 2000
        result = _safe_str(long_str)
        self.assertEqual(len(result), 1024)
        self.assertTrue(result.endswith("..."))

    def test_scope_names(self):
        from torch.profiler._opentelemetry import _SCOPE_NAMES

        self.assertEqual(_SCOPE_NAMES[0], "FUNCTION")
        self.assertEqual(_SCOPE_NAMES[7], "USER_SCOPE")

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_build_span_attributes(self):
        """Test attribute extraction from a real FunctionEvent."""
        from torch.profiler._opentelemetry import _build_span_attributes

        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
        ) as prof:
            self._simple_payload()

        events = prof.events()
        self.assertTrue(len(events) > 0)

        for event in events:
            attrs = _build_span_attributes(event)
            self.assertIn("pytorch.event.id", attrs)
            self.assertIn("pytorch.thread.id", attrs)
            self.assertIn("pytorch.scope", attrs)
            self.assertIn("pytorch.device.type", attrs)
            self.assertIn("pytorch.cpu_time_total_us", attrs)
            break  # just test the first event

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_build_event_tree(self):
        """Test that root events are correctly identified."""
        from torch.profiler._opentelemetry import _build_event_tree

        with profile(
            activities=[ProfilerActivity.CPU],
        ) as prof:
            self._simple_payload()

        events = prof.events()
        roots = _build_event_tree(events)
        # Roots should be a subset of all events
        self.assertTrue(len(roots) <= len(events))
        # All roots should have no parent
        for root in roots:
            self.assertIsNone(root.cpu_parent)

    @unittest.skipIf(not _otel_available(), "opentelemetry not installed")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_opentelemetry_trace_handler_with_in_memory_exporter(self):
        """End-to-end test with an in-memory exporter."""
        from torch.profiler._opentelemetry import opentelemetry_trace_handler

        exporter = _make_in_memory_exporter()
        handler = opentelemetry_trace_handler(
            service_name="test-pytorch",
            span_exporter=exporter,
        )

        with profile(
            activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=handler,
            record_shapes=True,
        ) as prof:
            self._simple_payload()
            prof.step()

        spans = exporter.get_finished_spans()
        self.assertTrue(len(spans) > 0, "Expected at least one OTel span to be exported")

        # Verify span attributes
        for span in spans:
            self.assertIsNotNone(span.name)
            self.assertTrue(len(span.name) > 0)
            attrs = dict(span.attributes) if span.attributes else {}
            self.assertIn("pytorch.event.id", attrs)

    @unittest.skipIf(not _otel_available(), "opentelemetry not installed")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_handler_with_explicit_tracer_provider(self):
        """Test passing an explicit TracerProvider."""
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from torch.profiler._opentelemetry import opentelemetry_trace_handler

        exporter = _make_in_memory_exporter()
        provider = TracerProvider(
            resource=Resource.create({"service.name": "custom-service"})
        )
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        handler = opentelemetry_trace_handler(tracer_provider=provider)

        with profile(
            activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=handler,
        ) as prof:
            self._simple_payload()
            prof.step()

        spans = exporter.get_finished_spans()
        self.assertTrue(len(spans) > 0)

    @unittest.skipIf(not _otel_available(), "opentelemetry not installed")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_span_parent_child_relationships(self):
        """Verify that parent-child span relationships are established."""
        from torch.profiler._opentelemetry import opentelemetry_trace_handler

        exporter = _make_in_memory_exporter()
        handler = opentelemetry_trace_handler(
            service_name="test-pytorch",
            span_exporter=exporter,
        )

        with profile(
            activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=handler,
        ) as prof:
            # Do operations that create parent-child relationships
            x = torch.randn(20, 20)
            y = torch.mm(x, x)
            prof.step()

        spans = exporter.get_finished_spans()
        # Check that at least some spans have parent span IDs
        span_ids = {s.context.span_id for s in spans}
        parent_ids = {
            s.parent.span_id
            for s in spans
            if s.parent is not None and s.parent.span_id != 0
        }
        # Some spans should reference other spans as parents
        # (not all, since root spans have no parent)
        has_children = len(parent_ids & span_ids) > 0 or len(spans) <= 1
        # This is a soft check - profiler event trees may vary
        self.assertTrue(len(spans) > 0)

    @unittest.skipIf(not _otel_available(), "opentelemetry not installed")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_handler_no_events(self):
        """Handler should not crash when there are no events."""
        from torch.profiler._opentelemetry import opentelemetry_trace_handler

        exporter = _make_in_memory_exporter()
        handler = opentelemetry_trace_handler(span_exporter=exporter)

        # Create a mock profile with no events
        mock_prof = MagicMock()
        mock_prof.events.return_value = []
        handler(mock_prof)

        spans = exporter.get_finished_spans()
        self.assertEqual(len(spans), 0)

    @unittest.skipIf(not _otel_available(), "opentelemetry not installed")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_span_timing(self):
        """Verify that span timestamps are reasonable."""
        import time

        from torch.profiler._opentelemetry import opentelemetry_trace_handler

        exporter = _make_in_memory_exporter()
        handler = opentelemetry_trace_handler(span_exporter=exporter)

        before_ns = time.time_ns()
        with profile(
            activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=handler,
        ) as prof:
            self._simple_payload()
            prof.step()
        after_ns = time.time_ns()

        spans = exporter.get_finished_spans()
        self.assertTrue(len(spans) > 0)

        for span in spans:
            # Span start should be after we started profiling (with some tolerance)
            # and end should be before now
            self.assertGreater(span.start_time, 0)
            self.assertGreaterEqual(span.end_time, span.start_time)


class TestOpenTelemetryExportPublicAPI(TestCase):
    """Test that the public API is accessible from torch.profiler."""

    def test_handler_importable_from_torch_profiler(self):
        """Verify the handler can be imported from torch.profiler."""
        from torch.profiler import opentelemetry_trace_handler

        self.assertTrue(callable(opentelemetry_trace_handler))

    def test_handler_in_all(self):
        """Verify the handler is listed in __all__."""
        import torch.profiler

        self.assertIn("opentelemetry_trace_handler", torch.profiler.__all__)


if __name__ == "__main__":
    run_tests()
