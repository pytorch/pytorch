import inspect
from typing import Any, Callable, Dict, Mapping, Tuple

from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics.infra import _infra, formatter


@_beartype.beartype
def python_frame(frame: inspect.FrameInfo) -> _infra.StackFrame:
    """Returns a StackFrame for the given inspect.FrameInfo."""
    snippet = (
        frame.code_context[frame.index].strip()
        if frame.code_context is not None and frame.index is not None
        else None
    )

    return _infra.StackFrame(
        location=_infra.Location(
            uri=frame.filename,
            line=frame.lineno,
            snippet=snippet,
            function=frame.function,
            message=snippet,
        )
    )


@_beartype.beartype
def python_call_stack(frames_to_skip: int = 0, frames_to_log: int = 16) -> _infra.Stack:
    """Returns the current Python call stack."""
    if frames_to_skip < 0:
        raise ValueError("frames_to_skip must be non-negative")
    if frames_to_log < 0:
        raise ValueError("frames_to_log must be non-negative")
    frames_to_skip += 2  # Skip this function and beartype.
    stack = _infra.Stack()
    stack.frames = [
        python_frame(frame)
        # TODO(bowbao): Rewrite with 'traceback' to speedup performance.
        # Reference code: `torch/fx/proxy.py`.
        # `inspect.stack(0)` will speedup the call greatly, but loses line snippet.
        for frame in inspect.stack()[frames_to_skip : frames_to_skip + frames_to_log]
    ]
    stack.message = "Python call stack"
    return stack


@_beartype.beartype
def function_location(fn: Callable) -> _infra.Location:
    """Returns a Location for the given function."""
    source_lines, lineno = inspect.getsourcelines(fn)
    snippet = source_lines[0].strip() if len(source_lines) > 0 else "<unknown>"
    return _infra.Location(
        uri=inspect.getsourcefile(fn),
        line=lineno,
        snippet=snippet,
        message=formatter.display_name(fn),
    )


@_beartype.beartype
def function_state(
    fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Mapping[str, Any]:
    bind = inspect.signature(fn).bind(*args, **kwargs)
    return bind.arguments
