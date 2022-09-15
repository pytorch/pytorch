import contextlib

from torch.onnx import diagnostic
from torch.onnx._internal import _beartype
from torch.onnx.diagnostic import options


@contextlib.contextmanager
@_beartype.beartype
def enable_diagnostic(options: options.DiagnosticOptions):
    diagnostic.engine.initialize(options)
    try:
        yield diagnostic.engine
    finally:
        diagnostic.engine.clear()


def set_diagnostic_options(options: options.DiagnosticOptions):
    diagnostic.engine._set_options(options)
    try:
        yield
    finally:
        diagnostic.engine._clear_options()
