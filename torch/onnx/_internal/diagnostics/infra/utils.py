import inspect

from torch.onnx._internal.diagnostics.infra import _infra


def python_frame(frame: inspect.FrameInfo) -> _infra.StackFrame:
    """Returns a StackFrame for the given inspect.FrameInfo."""
    snippet = (
        frame.code_context[frame.index]
        if frame.code_context is not None and frame.index is not None
        else None
    )

    return _infra.StackFrame(
        location=_infra.Location(
            uri=frame.filename,
            line=frame.lineno,
            snippet=snippet,
        )
    )


def python_call_stack(frames_to_skip: int = 0, frames_to_log: int = 32) -> _infra.Stack:
    """Returns the current Python call stack."""
    if frames_to_skip < 0:
        raise ValueError("frames_to_skip must be non-negative")
    if frames_to_log < 0:
        raise ValueError("frames_to_log must be non-negative")
    frames_to_skip += 1  # Skip this function.
    stack = _infra.Stack()
    stack.frames = [
        python_frame(frame)
        for frame in inspect.stack()[frames_to_skip : frames_to_skip + frames_to_log]
    ]
    return stack
