# Owner(s): ["module: functorch"]

"""
Shared utilities for codegen wrapper tests.

Provides a mixin class with a context manager for capturing trace_structured
codegen artifacts.
"""

import logging
from contextlib import contextmanager


trace_log = logging.getLogger("torch.__trace")


class CodegenArtifactMixin:
    """Mixin providing _capture_codegen_source for TestCase subclasses."""

    @contextmanager
    def _capture_codegen_source(self, artifact_name):  # type: ignore[no-untyped-def]
        """Capture codegen artifacts from the structured trace log."""
        captured: list[str] = []

        class _ArtifactHandler(logging.Handler):
            def emit(self, record):  # type: ignore[no-untyped-def]
                metadata = getattr(record, "metadata", {})
                if (
                    "artifact" in metadata
                    and metadata["artifact"].get("name") == artifact_name
                ):
                    payload = getattr(record, "payload", None)
                    if payload is not None:
                        captured.append(payload)

        handler = _ArtifactHandler()
        handler.setLevel(logging.DEBUG)
        old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)
        trace_log.addHandler(handler)
        try:
            yield captured
        finally:
            trace_log.removeHandler(handler)
            trace_log.setLevel(old_level)
