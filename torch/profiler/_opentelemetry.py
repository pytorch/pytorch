"""OpenTelemetry trace export support for PyTorch Profiler.

This module provides an ``on_trace_ready`` handler that converts PyTorch
profiler events into OpenTelemetry spans, allowing seamless integration
with any OTel-compatible backend (Jaeger, Zipkin, OTLP, etc.).

The ``opentelemetry-api`` and ``opentelemetry-sdk`` packages are **optional**
dependencies.  A clear error is raised at runtime if they are missing.

Usage::

    from torch.profiler import profile, schedule
    from torch.profiler._opentelemetry import opentelemetry_trace_handler

    with profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=opentelemetry_trace_handler(service_name="my-model"),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for step in range(8):
            train_step()
            prof.step()
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SpanExporter

    from torch.autograd.profiler_util import FunctionEvent
    from torch.profiler.profiler import _KinetoProfile


logger = logging.getLogger(__name__)


def _check_otel_available() -> None:
    """Raise an informative ``ImportError`` if OpenTelemetry is not installed."""
    try:
        import opentelemetry.sdk.trace  # noqa: F401
        import opentelemetry.trace  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "OpenTelemetry packages are required for opentelemetry_trace_handler. "
            "Install them with: pip install opentelemetry-api opentelemetry-sdk"
        ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCOPE_NAMES = {
    0: "FUNCTION",
    1: "BACKWARD_FUNCTION",
    2: "TORCHSCRIPT_FUNCTION",
    3: "KERNEL_FUNCTION_DTYPE",
    4: "CUSTOM_CLASS",
    5: "BUILD_FEATURE",
    6: "LITE_INTERPRETER",
    7: "USER_SCOPE",
    8: "STATIC_RUNTIME_OP",
    9: "STATIC_RUNTIME_MODEL",
}


def _us_to_ns(us: float) -> int:
    """Convert microseconds (profiler native unit) to nanoseconds (OTel unit)."""
    return int(us * 1_000)


def _safe_str(value: Any) -> str:
    """Safely convert a value to a bounded string representation."""
    s = str(value)
    if len(s) > 1024:
        return s[:1021] + "..."
    return s


def _build_span_attributes(event: FunctionEvent) -> dict[str, Any]:
    """Extract span attributes from a ``FunctionEvent``."""
    attrs: dict[str, Any] = {}

    # Core identity
    attrs["pytorch.event.id"] = event.id
    attrs["pytorch.thread.id"] = event.thread
    attrs["pytorch.scope"] = _SCOPE_NAMES.get(event.scope, str(event.scope))

    # Device information
    attrs["pytorch.device.type"] = str(event.device_type).rsplit(".", 1)[-1]
    attrs["pytorch.device.index"] = event.device_index
    if event.use_device:
        attrs["pytorch.device.name"] = event.use_device

    # Timing (in microseconds for readability)
    attrs["pytorch.cpu_time_total_us"] = event.cpu_time_total
    attrs["pytorch.self_cpu_time_total_us"] = event.self_cpu_time_total
    if event.device_time_total > 0:
        attrs["pytorch.device_time_total_us"] = event.device_time_total
        attrs["pytorch.self_device_time_total_us"] = event.self_device_time_total

    # Memory
    if event.cpu_memory_usage != 0:
        attrs["pytorch.cpu_memory_usage_bytes"] = event.cpu_memory_usage
    if event.device_memory_usage != 0:
        attrs["pytorch.device_memory_usage_bytes"] = event.device_memory_usage

    # FLOPs
    if event.flops is not None and event.flops > 0:
        attrs["pytorch.flops"] = event.flops

    # Input shapes
    if event.input_shapes:
        attrs["pytorch.input_shapes"] = _safe_str(event.input_shapes)

    # Input dtypes
    input_dtypes = getattr(event, "input_dtypes", None)
    if input_dtypes:
        attrs["pytorch.input_dtypes"] = _safe_str(input_dtypes)

    # Stack trace
    if event.stack:
        attrs["pytorch.stack"] = "; ".join(event.stack[:10])

    # Flags
    if event.is_async:
        attrs["pytorch.is_async"] = True
    if event.is_user_annotation:
        attrs["pytorch.is_user_annotation"] = True

    # Kernel info
    if event.kernels:
        kernel_names = [k.name for k in event.kernels]
        attrs["pytorch.kernel_names"] = _safe_str(kernel_names)

    return attrs


# ---------------------------------------------------------------------------
# Event tree reconstruction
# ---------------------------------------------------------------------------

def _build_event_tree(
    events: Sequence[FunctionEvent],
) -> list[FunctionEvent]:
    """Return the root events (those without a CPU parent).

    The profiler already establishes ``cpu_parent`` / ``cpu_children``
    relationships, so we simply filter for root nodes.
    """
    return [e for e in events if e.cpu_parent is None]


# ---------------------------------------------------------------------------
# Span creation
# ---------------------------------------------------------------------------

def _export_event_as_span(
    event: FunctionEvent,
    tracer: Any,
    trace_start_ns: int,
    parent_context: Any | None = None,
) -> None:
    """Recursively create OTel spans for *event* and its children.

    Parameters
    ----------
    event:
        The profiler ``FunctionEvent``.
    tracer:
        An ``opentelemetry.trace.Tracer`` instance.
    trace_start_ns:
        An absolute epoch-nanosecond offset so that profiler-relative
        timestamps become wall-clock timestamps.
    parent_context:
        Optional OTel ``Context`` to set as the parent.
    """
    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace

    attrs = _build_span_attributes(event)

    start_ns = trace_start_ns + _us_to_ns(event.time_range.start)
    end_ns = trace_start_ns + _us_to_ns(event.time_range.end)

    # Ensure end >= start (defensive)
    if end_ns < start_ns:
        end_ns = start_ns

    span = tracer.start_span(
        name=event.name,
        attributes=attrs,
        start_time=start_ns,
        context=parent_context,
    )
    child_context = otel_trace.set_span_in_context(span, parent_context)

    # Recurse into CPU children
    for child in event.cpu_children:
        _export_event_as_span(child, tracer, trace_start_ns, child_context)

    span.end(end_time=end_ns)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def opentelemetry_trace_handler(
    service_name: str = "pytorch",
    tracer_provider: TracerProvider | None = None,
    span_exporter: SpanExporter | None = None,
) -> Callable[..., None]:
    """Return an ``on_trace_ready`` callback that exports spans via OpenTelemetry.

    This follows the same pattern as
    :func:`torch.profiler.tensorboard_trace_handler` — it returns a callable
    that accepts a :class:`torch.profiler.profile` instance.

    Parameters
    ----------
    service_name:
        The OTel service name.  Defaults to ``"pytorch"``.
    tracer_provider:
        An explicit ``TracerProvider``.  When *None* a new
        ``TracerProvider`` is created with a ``SimpleSpanProcessor`` wrapping
        *span_exporter* (or a ``ConsoleSpanExporter`` if that is also *None*).
    span_exporter:
        A ``SpanExporter`` used only when *tracer_provider* is *None*.
        Ignored when a provider is supplied explicitly.

    Returns
    -------
    Callable
        A callback suitable for the ``on_trace_ready`` argument of
        :class:`torch.profiler.profile`.

    Example
    -------
    ::

        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        handler = opentelemetry_trace_handler(
            service_name="my-training-job",
            span_exporter=ConsoleSpanExporter(),
        )

        with torch.profiler.profile(
            on_trace_ready=handler,
            record_shapes=True,
        ) as prof:
            model(inputs)
    """
    _check_otel_available()

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    if tracer_provider is None:
        resource = Resource.create({"service.name": service_name})
        tracer_provider = _TracerProvider(resource=resource)
        exporter = span_exporter or ConsoleSpanExporter()
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = tracer_provider.get_tracer("torch.profiler")

    def handler(prof: _KinetoProfile) -> None:
        import time

        events = prof.events()
        if not events:
            logger.debug("No profiler events to export as OTel spans.")
            return

        # Use current wall-clock time as the anchor for the trace.
        # Profiler timestamps are relative (microseconds since profiling start),
        # so we offset them from "now minus the latest event end".
        now_ns = time.time_ns()
        max_end_us = max(e.time_range.end for e in events)
        trace_start_ns = now_ns - _us_to_ns(max_end_us)

        roots = _build_event_tree(events)
        for root_event in roots:
            _export_event_as_span(root_event, tracer, trace_start_ns)

        logger.debug(
            "Exported %d profiler events (%d root spans) as OTel spans.",
            len(events),
            len(roots),
        )

    return handler
