from __future__ import annotations

import functools
import inspect
import traceback
from typing import Any, Callable, Mapping, Sequence

from torch.onnx._internal.diagnostics.infra import _infra, formatter


def python_frame(frame: traceback.FrameSummary) -> _infra.StackFrame:
    """Returns a StackFrame for the given traceback.FrameSummary."""
    snippet = frame.line

    return _infra.StackFrame(
        location=_infra.Location(
            uri=frame.filename,
            line=frame.lineno,
            snippet=snippet,
            function=frame.name,
            message=snippet,
        )
    )


def python_call_stack(frames_to_skip: int = 0, frames_to_log: int = 16) -> _infra.Stack:
    """Returns the current Python call stack."""
    if frames_to_skip < 0:
        raise ValueError("frames_to_skip must be non-negative")
    if frames_to_log < 0:
        raise ValueError("frames_to_log must be non-negative")
    frames_to_skip += 1  # Skip this function.
    stack = _infra.Stack()
    # Frames are returned in order of oldest to newest.
    frames = traceback.extract_stack(limit=frames_to_skip + frames_to_log)
    frames.reverse()
    stack.frames = [python_frame(frame) for frame in frames[frames_to_skip:]]
    stack.message = "Python call stack"
    return stack


@functools.lru_cache
def _function_source_info(fn: Callable) -> tuple[Sequence[str], int, str | None]:
    """Returns the source lines, line number, and source file path for the given function.

    Essentially, inspect.getsourcelines() and inspect.getsourcefile() combined.
    Caching is applied to reduce the performance impact of this function.
    """
    source_lines, lineno = inspect.getsourcelines(fn)
    return source_lines, lineno, inspect.getsourcefile(fn)


def function_location(fn: Callable) -> _infra.Location:
    """Returns a Location for the given function."""
    source_lines, lineno, uri = _function_source_info(fn)
    snippet = source_lines[0].strip() if len(source_lines) > 0 else "<unknown>"
    return _infra.Location(
        uri=uri,
        line=lineno,
        snippet=snippet,
        message=formatter.display_name(fn),
    )


def function_state(
    fn: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Mapping[str, Any]:
    bind = inspect.signature(fn).bind(*args, **kwargs)
    return bind.arguments
