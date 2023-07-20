import os
import textwrap
from enum import auto, Enum
from traceback import extract_stack, format_exc, format_list, FrameSummary
from typing import cast, List

from . import config
from .config import is_fbcode

from .utils import counters, format_bytecode

if is_fbcode():
    from torch.fb.exportdb.logging import exportdb_error_message
else:

    def exportdb_error_message(case_name):
        return ""


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


class InvalidBackend(TorchDynamoException):
    def __init__(self, name):
        super().__init__(
            f"Invalid backend: {name!r}, see `torch._dynamo.list_backends()` for available backends."
        )


class ResetRequired(TorchDynamoException):
    def __init__(self):
        super().__init__(
            textwrap.dedent(
                """
                Must call `torch._dynamo.reset()` before changing backends.  Detected two calls to
                `torch.compile()` with a different backend compiler arguments.
                """
            )
        )


class BackendCompilerFailed(TorchDynamoException):
    def __init__(self, backend_fn, inner_exception):
        self.backend_name = getattr(backend_fn, "__name__", "?")
        self.inner_exception = inner_exception
        msg = f"backend={self.backend_name!r} raised:\n{type(inner_exception).__name__}: {inner_exception}"
        super().__init__(msg)


class Unsupported(TorchDynamoException):
    def __init__(self, msg):
        super().__init__(msg)
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


class RecompileError(TorchDynamoException):
    pass


class ArgsMismatchError(Unsupported):
    def __init__(self, msg):
        super().__init__(msg)


class AttributeMutationError(Unsupported):
    def __init__(self, msg):
        super().__init__(msg)


class CondOpArgsMismatchError(ArgsMismatchError):
    """
    Internal error from cond() due to arguments mismatch.
    """

    def __init__(self, msg):
        super().__init__(msg)


class UserErrorType(Enum):
    DYNAMIC_CONTROL_FLOW = auto()
    ANTI_PATTERN = auto()
    STANDARD_LIBRARY = auto()
    CONSTRAIN_VIOLATION = auto()
    DYNAMIC_DIM = auto()
    INVALID_INPUT = auto()


class UserError(Unsupported):
    def __init__(self, error_type: UserErrorType, msg, case_name=None):
        """
        Type of errors that would be valid in Eager, but not supported in TorchDynamo.
        The error message should tell user about next actions.

        error_type: Type of user error
        msg: Actionable error message
        case_name: (Optional) Unique name (snake case) for the usage example in exportdb.
        """
        if case_name is not None:
            assert isinstance(case_name, str)
            msg += exportdb_error_message(case_name)
        super().__init__(msg)
        self.error_type = error_type
        self.message = msg


class IncorrectUsage(Exception):
    pass


def unimplemented(msg: str):
    assert msg != os.environ.get("BREAK", False)
    raise Unsupported(msg)


def warning(msg: str):
    counters["warnings"][msg] += 1
    assert msg != os.environ.get("BREAK", False)


# KeyError has special handling for its args
# see https://github.com/python/cpython/blob/3.11/Objects/exceptions.c#L2534 for details
class KeyErrorMsg:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self) -> str:
        return self.__str__()


def augment_exc_message(exc, msg="\n"):
    import traceback

    if (
        hasattr(exc, "real_stack")
        and len(exc.real_stack) > 0
        and not (config.verbose and config.suppress_errors)
    ):
        msg += f"\nfrom user code:\n {''.join(traceback.format_list(list(reversed(get_real_stack(exc)[0:2]))))}"

    if config.replay_record_enabled and hasattr(exc, "record_filename"):
        msg += f"\nLast frame execution written to {exc.record_filename}. To run only this frame while debugging, run\
 torch._dynamo.replay('{exc.record_filename}').\n"

    if not config.verbose and hasattr(exc, "real_stack"):
        msg += '\nSet TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information\n'

    if hasattr(exc, "inner_exception") and hasattr(
        exc.inner_exception, "minifier_path"
    ):
        if hasattr(exc.inner_exception, "buck_command"):
            msg += (
                f"\nMinifier script written to {exc.inner_exception.minifier_path}. Run "
                f"this buck command to find the smallest traced graph "
                f"which reproduces this error: {exc.inner_exception.buck_command}\n"
            )
        else:
            msg += (
                f"\nMinifier script written to {exc.inner_exception.minifier_path}. Run "
                "this script to find the smallest traced graph which reproduces this error.\n"
            )

    if not config.suppress_errors:
        msg += (
            "\n\n"
            "You can suppress this exception and fall back to eager by setting:\n"
            "    import torch._dynamo\n"
            "    torch._dynamo.config.suppress_errors = True\n"
        )

    old_msg = "" if len(exc.args) == 0 else str(exc.args[0])

    if isinstance(exc, KeyError):
        exc.args = (KeyErrorMsg(old_msg + msg),) + exc.args[1:]
    else:
        new_msg = old_msg + msg
        exc.args = (new_msg,) + exc.args[1:]


def get_real_stack(exc) -> List[FrameSummary]:
    assert hasattr(exc, "real_stack")
    return cast(List[FrameSummary], exc.real_stack)


# filter out all frames after entering dynamo
def filter_stack(stack):
    user_stack = []
    for frame in stack:
        if "convert_frame" in frame.filename:
            break
        if "eval_frame" in frame.filename or "torch._dynamo.optimize(" in frame.line:
            continue
        user_stack.append(frame)

    return user_stack


def format_error_msg(exc, code, record_filename=None, frame=None):
    msg = os.linesep * 2

    if config.verbose:
        msg = str(
            format_bytecode(
                "WON'T CONVERT",
                code.co_name,
                code.co_filename,
                code.co_firstlineno,
                code,
            )
        )
        msg += "=" * 10 + " TorchDynamo Stack Trace " + "=" * 10 + "\n"
        msg += format_exc()
        if hasattr(exc, "real_stack"):
            msg += (
                "\n"
                + "=" * 10
                + " The above exception occurred while processing the following code "
                + "=" * 10
                + "\n\n"
            )
            stack_above_dynamo = []
            if frame is not None:
                stack_above_dynamo = filter_stack(extract_stack(frame))

            msg += "".join(
                format_list(stack_above_dynamo + list(reversed(get_real_stack(exc))))
            )
            msg += "\n"
            msg += "=" * 10

    else:
        msg = f"WON'T CONVERT {code.co_name} {code.co_filename}\
 line {code.co_firstlineno} \ndue to: \n{format_exc(limit=-1)}"

    return msg
