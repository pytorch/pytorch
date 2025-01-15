"""
This coverage plug-in attempts to cover JIT'd functions and methods that were previously missed in code coverage. Any
function and method that was passed through/decorated with torch.jit.script or torch.jit.script_method should now be
marked covered when coverage is run with this plug-in.

DISCLAIMER: note that this will mark the entire JIT'd function/method as covered without seeking proof that the
compiled code has been executed. This means that even if the code chunk is merely compiled and not run, it will get
marked as covered.
"""

from inspect import (
    getsourcefile,
    getsourcelines,
    isclass,
    iscode,
    isfunction,
    ismethod,
    ismodule,
)
from time import time
from typing import Any

from coverage import CoverageData, CoveragePlugin  # type: ignore[import]


# All coverage stats resulting from this plug-in will be in a separate .coverage file that should be merged later with
# `coverage combine`. The convention seems to be .coverage.dotted.suffix based on the following link:
# https://coverage.readthedocs.io/en/coverage-5.5/cmd.html#combining-data-files-coverage-combine
cov_data = CoverageData(basename=f".coverage.jit.{time()}")


def is_not_builtin_class(obj: Any) -> bool:
    return isclass(obj) and not type(obj).__module__ == "builtins"


class JitPlugin(CoveragePlugin):  # type: ignore[misc, no-any-unimported]
    """
    dynamic_context is an overridden function that gives us access to every frame run during the coverage process. We
    look for when the function being run is `should_drop`, as all functions that get passed into `should_drop` will be
    compiled and thus should be marked as covered.
    """

    def dynamic_context(self, frame: Any) -> None:
        if frame.f_code.co_name == "should_drop":
            obj = frame.f_locals["fn"]
            # The many conditions in the if statement below are based on the accepted arguments to getsourcefile. Based
            # on its documentation (https://docs.python.org/3/library/inspect.html#inspect.getsourcefile), the argument
            # must be a module, class, method, function, traceback, frame, or code object AND it cannot be a built-in
            # module, class, or function.
            # Currently, we DO NOT include tracebacks or frames as they should not be JIT'd, and we have not checked for
            # built-in modules or functions as those do not seem to be JIT'd either.
            if (
                is_not_builtin_class(obj)
                or ismodule(obj)
                or ismethod(obj)
                or isfunction(obj)
                or iscode(obj)
            ):
                filename = getsourcefile(obj)
                # We don't want to report for filename = None
                if filename:
                    # TODO: Because torch.jit._IgnoreContextManager relies on Python's `exec` method
                    # which doesn't generate source codelines, getsourcelines(obj) fails. For now,
                    # we just ignore the exception until we figure out a better way to
                    # implement torch.jit._IgnoreContextManager.
                    try:
                        sourcelines, starting_lineno = getsourcelines(obj)
                    except OSError:
                        pass
                    else:
                        line_data = {
                            filename: range(
                                starting_lineno, starting_lineno + len(sourcelines)
                            )
                        }
                        cov_data.add_lines(line_data)
        super().dynamic_context(frame)


def coverage_init(reg: Any, options: Any) -> None:
    reg.add_dynamic_context(JitPlugin())
