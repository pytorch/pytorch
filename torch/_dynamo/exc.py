import os
import textwrap

from .utils import counters


class TorchDynamoException(RuntimeError):
    pass


class InternalTorchDynamoError(TorchDynamoException):
    pass


class RestartAnalysis(TorchDynamoException):
    pass


class SkipFrame(TorchDynamoException):
    pass


class TorchRuntimeError(TorchDynamoException):
    pass


class ResetRequired(TorchDynamoException):
    def __init__(self):
        super(ResetRequired, self).__init__(
            textwrap.dedent(
                """
                Must call `torchdynamo.reset()` before changing backends.  Detected two calls to
                `torchdynamo.optimize(...)` with a different backend compiler arguments.
                """
            )
        )


class BackendCompilerFailed(TorchDynamoException):
    def __init__(self, backend_fn, inner_exception):
        self.backend_name = getattr(backend_fn, "__name__", "?")
        self.inner_exception = inner_exception
        super().__init__(
            f"{self.backend_name} raised {type(inner_exception).__name__}: {inner_exception}"
            "\n\n"
            "You can suppress this exception and fall back to eager by setting:\n"
            "    torchdynamo.config.raise_on_backend_error = False"
        )


class Unsupported(TorchDynamoException):
    def __init__(self, msg):
        super(Unsupported, self).__init__(msg)
        self.real_stack = []
        self.msg = msg
        self.category = None
        self.add_to_stats()

    def remove_from_stats(self):
        counters[self.category][self.msg] -= 1
        if counters[self.category][self.msg] <= 0:
            del counters[self.category][self.msg]

    def add_to_stats(self, category="unimplemented"):
        self.category = category
        counters[category][self.msg] += 1


def unimplemented(msg: str):
    assert msg != os.environ.get("BREAK", False)
    raise Unsupported(msg)


def warning(msg: str):
    counters["warnings"][msg] += 1
    assert msg != os.environ.get("BREAK", False)
