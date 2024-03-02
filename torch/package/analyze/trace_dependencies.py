import sys
from typing import Any, Callable, Iterable, List, Tuple

__all__ = ["trace_dependencies"]


def trace_dependencies(
    callable: Callable[[Any], Any], inputs: Iterable[Tuple[Any, ...]]
) -> List[str]:
    """Trace the execution of a callable in order to determine which modules it uses.

    Args:
        callable: The callable to execute and trace.
        inputs: The input to use during tracing. The modules used by 'callable' when invoked by each set of inputs
            are union-ed to determine all modules used by the callable for the purpooses of packaging.

    Returns: A list of the names of all modules used during callable execution.
    """
    modules_used = set()

    def record_used_modules(frame, event, arg):
        # If the event being profiled is not a Python function
        # call, there is nothing to do.
        if event != "call":
            return

        # This is the name of the function that was called.
        name = frame.f_code.co_name
        module = None

        # Try to determine the name of the module that the function
        # is in:
        #   1) Check the global namespace of the frame.
        #   2) Check the local namespace of the frame.
        #   3) To handle class instance method calls, check
        #       the attribute named 'name' of the object
        #       in the local namespace corresponding to "self".
        if name in frame.f_globals:
            module = frame.f_globals[name].__module__
        elif name in frame.f_locals:
            module = frame.f_locals[name].__module__
        elif "self" in frame.f_locals:
            method = getattr(frame.f_locals["self"], name, None)
            module = method.__module__ if method else None

        # If a module was found, add it to the set of used modules.
        if module:
            modules_used.add(module)

    try:
        # Attach record_used_modules as the profiler function.
        sys.setprofile(record_used_modules)

        # Execute the callable with all inputs.
        for inp in inputs:
            callable(*inp)

    finally:
        # Detach the profiler function.
        sys.setprofile(None)

    return list(modules_used)
