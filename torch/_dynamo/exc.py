# mypy: allow-untyped-defs
import os
import textwrap
from enum import auto, Enum
from traceback import extract_stack, format_exc, format_list, StackSummary
from typing import Any, cast, List, NoReturn, Optional, Tuple, TYPE_CHECKING

import torch._guards

from . import config
from .graph_break_reason import GraphBreakReason
from .utils import counters


if TYPE_CHECKING:
    from torch._guards import CompileId


def exportdb_error_message(case_name):
    return (
        "For more information about this error, see: "
        + "https://pytorch.org/docs/main/generated/exportdb/index.html#"
        + case_name.replace("_", "-")
    )


import logging


log = logging.getLogger(__name__)
graph_breaks_log = torch._logging.getArtifactLogger(__name__, "graph_breaks")


class TorchDynamoException(RuntimeError):
    pass


class InternalTorchDynamoError(TorchDynamoException):
    pass


class RestartAnalysis(TorchDynamoException):
    restart_reason: str

    def __init__(self, *args, restart_reason=None) -> None:
        self.restart_reason = restart_reason
        super().__init__(*args)


class SpeculationRestartAnalysis(RestartAnalysis):
    pass


class UnspecializeRestartAnalysis(RestartAnalysis):
    pass


class CompileCollectiveRestartAnalysis(RestartAnalysis):
    pass


class SkipFrame(TorchDynamoException):
    pass


class TorchRuntimeError(TorchDynamoException):
    pass


class InvalidBackend(TorchDynamoException):
    def __init__(self, name) -> None:
        super().__init__(
            f"Invalid backend: {name!r}, see `torch._dynamo.list_backends()` for available backends."
        )


class ResetRequired(TorchDynamoException):
    def __init__(self) -> None:
        super().__init__(
            textwrap.dedent(
                """
                Must call `torch._dynamo.reset()` before changing backends.  Detected two calls to
                `torch.compile()` with a different backend compiler arguments.
                """
            )
        )


class BackendCompilerFailed(TorchDynamoException):
    def __init__(self, backend_fn, inner_exception) -> None:
        self.backend_name = getattr(backend_fn, "__name__", "?")
        self.inner_exception = inner_exception
        msg = f"backend={self.backend_name!r} raised:\n{type(inner_exception).__name__}: {inner_exception}"
        super().__init__(msg)


class Unsupported(TorchDynamoException):
    def __init__(self, msg, *, case_name=None) -> None:
        super().__init__(msg)
        self.real_stack = torch._guards.TracingContext.extract_stack()
        self.msg = msg
        self.category: Optional[str] = None
        self.add_to_stats()
        self.case_name: Optional[str] = case_name

    def remove_from_stats(self):
        assert self.category is not None
        counters[self.category][self.msg] -= 1
        if counters[self.category][self.msg] <= 0:
            del counters[self.category][self.msg]

    def add_to_stats(self, category="unimplemented"):
        self.category = category
        counters[category][self.msg] += 1


class RecompileError(TorchDynamoException):
    pass


class ArgsMismatchError(Unsupported):
    def __init__(self, msg) -> None:
        super().__init__(msg)


class AttributeMutationError(Unsupported):
    def __init__(self, msg) -> None:
        super().__init__(msg)


class CondOpArgsMismatchError(ArgsMismatchError):
    """
    Internal error from cond() due to arguments mismatch.
    """

    def __init__(self, msg) -> None:
        super().__init__(msg)


class UserErrorType(Enum):
    DYNAMIC_CONTROL_FLOW = auto()
    ANTI_PATTERN = auto()
    STANDARD_LIBRARY = auto()
    CONSTRAINT_VIOLATION = auto()
    DYNAMIC_DIM = auto()
    INVALID_INPUT = auto()
    INVALID_OUTPUT = auto()
    UNSUPPORTED_ALIASED_MUTATED_DYNAMIC_INPUTS = auto()


class UserError(Unsupported):
    def __init__(self, error_type: UserErrorType, msg, case_name=None) -> None:
        """
        Type of errors that would be valid in Eager, but not supported in TorchDynamo.
        The error message should tell user about next actions.

        error_type: Type of user error
        msg: Actionable error message
        case_name: (Optional) Unique name (snake case) for the usage example in exportdb.
        """
        if case_name is not None:
            assert isinstance(case_name, str)
            if msg.endswith("."):
                msg += " "
            else:
                msg += "\n"
            msg += exportdb_error_message(case_name)
        super().__init__(msg)
        self.error_type = error_type
        self.message = msg


class SkipCodeRecursiveException(TorchDynamoException):
    pass


class CacheLimitExceeded(Unsupported):
    pass


class UnsafeScriptObjectError(TorchDynamoException):
    pass


class UncapturedHigherOrderOpError(TorchDynamoException):
    pass


class IncorrectUsage(Exception):
    pass


# TODO: I'm a little uncertain about what error classification we should have
# for this.  This is potentially a user error, but regressions in
# specialization in PyTorch proper could also trigger this problem
class FailOnCacheLimitHit(Exception):
    pass


class ObservedException(TorchDynamoException):
    # An exception observed during the tracing. This exception is used by Dynamo to handle exceptions.
    pass


class ObservedUserStopIteration(ObservedException):
    # An UserStopIteraion exception observed during the Dynamo tracing (e.g Dynamo tracing __next__)
    value: Optional[Any]

    # Reference `StopIteration_init` in CPython
    # https://github.com/python/cpython/blob/3.11/Objects/exceptions.c#L568-L584
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("unhandled `raise StopIteration`")
        if len(args) > 0:
            self.value = args[0]
        else:
            self.value = None


class ObservedKeyError(ObservedException):
    # A KeyError exception to be raised from inside Dynamo tracing. This can happen on dict __getitem__
    pass


class ObservedAttributeError(ObservedException):
    # An AttributeError exception to be raised from inside Dynamo tracing. This can happen on user defined object __getattr__
    pass


observed_exception_map = {
    StopIteration: ObservedUserStopIteration,
    KeyError: ObservedKeyError,
    AttributeError: ObservedAttributeError,
}


def raise_observed_exception(e, tx):
    from .variables import BuiltinVariable

    # CPython here raises an exception. Since there is no python code, we have to manually setup the exception
    # stack and raise the exception.
    exception_vt = BuiltinVariable(e).call_function(tx, [], {})
    tx.exn_vt_stack.append(exception_vt)
    raise observed_exception_map[e]


def handle_observed_exception(tx):
    # This is essentially exception handling code, equivalent of this pseudo code
    #
    # try:
    #     ... somebody raising StopIteration
    # except StopIteration
    #     pass
    #
    # If this was going through the python code, we would have called exception_handler method, but FOR_ITER
    # handles the exception completely in CPython. For example for 3.11, the resulting bytecode is
    #
    #
    #   6          46 LOAD_GLOBAL              2 (StopIteration)
    #              58 RAISE_VARARGS            1
    #         >>   60 PUSH_EXC_INFO

    #   7          62 LOAD_GLOBAL              2 (StopIteration)
    #              74 CHECK_EXC_MATCH
    #              76 POP_JUMP_FORWARD_IF_FALSE     3 (to 84)
    #              78 POP_TOP

    #   8          80 POP_EXCEPT
    #

    # Fortunately this translates to a simple pop from the exn_vt_stack
    tx.exn_vt_stack.pop()


# These exceptions are ok to fallback to eager/graph_break.
exceptions_allowed_to_be_fallback = (
    torch._subclasses.fake_tensor.DataDependentOutputException,
    torch._subclasses.fake_tensor.DynamicOutputShapeException,
    torch._subclasses.fake_tensor.UnsupportedOperatorException,
    torch._subclasses.fake_tensor.UnsupportedFakeTensorException,
)


def unimplemented_with_warning(e: Exception, code, msg: str) -> NoReturn:
    # This function calls unimplemented internally and eventually graph breaks
    # or falls to eager. unimplemented itself does not print any user warnings,
    # i.e., its very silent. This helper function is intended when an error is
    # encountered in the torch.compile stack which is worth showing as warning
    # to the user. For example, if AOT Autograd backend fails with a fake tensor
    # exception, its ok to fallback to eager but not silently. Here, we can use
    # this function to log the message and the stack trace.
    graph_break_msg = format_error_msg_verbose(e, code)
    torch._logging.trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "dynamo_graph_break_reason",
            "encoding": "string",
        },
        payload_fn=lambda: graph_break_msg,
    )
    graph_breaks_log.debug("%s", graph_break_msg)
    log.warning(msg)
    unimplemented(msg, from_exc=e)


_NOTHING = object()


def unimplemented(
    reason: str,
    *,
    descr: str = "",
    workaround: str = "",
    tags: Optional[List[GraphBreakReason]] = None,
    from_exc: Any = _NOTHING,
    case_name: Optional[str] = None,
) -> NoReturn:
    if not tags:
        tags = []
    tags = [GraphBreakReason(reason, descr, workaround)] + tags
    if from_exc is not _NOTHING:
        raise Unsupported(reason, case_name=case_name) from from_exc
    raise Unsupported(reason, case_name=case_name)


def warning(msg: str) -> None:
    counters["warnings"][msg] += 1


# KeyError has special handling for its args
# see https://github.com/python/cpython/blob/3.11/Objects/exceptions.c#L2534 for details
class KeyErrorMsg:
    def __init__(self, value) -> None:
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return self.__str__()


def augment_exc_message(exc: Exception, msg: str = "\n", export: bool = False) -> None:
    import traceback

    exc.innermost_user_frame_summary = None  # type: ignore[attr-defined]

    real_stack = get_real_stack(exc)
    if real_stack is not None and len(real_stack) > 0:
        exc.innermost_user_frame_summary = real_stack[-1]  # type: ignore[attr-defined]
        msg += f"\nfrom user code:\n {''.join(traceback.format_list(real_stack))}"

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

    if not config.suppress_errors and not export:
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


def get_exc_message(
    e: Exception, compile_id: "CompileId"
) -> Tuple[Optional[str], Optional[int]]:
    filename = None
    lineno = None
    if e.innermost_user_frame_summary is not None:  # type: ignore[attr-defined]
        filename = e.innermost_user_frame_summary.filename  # type: ignore[attr-defined]
        lineno = e.innermost_user_frame_summary.lineno  # type: ignore[attr-defined]
    e.compile_id = compile_id  # type: ignore[attr-defined]
    return filename, lineno


def get_real_stack(exc: Exception, frame=None) -> Optional[StackSummary]:
    real_stack = getattr(exc, "real_stack", None)
    if real_stack is None:
        return None

    # NB: it's possible for real_stack to be []; we still attempt to
    # report a stack anyway because the stack_above_dynamo may still
    # be useful for debugging

    stack_above_dynamo = []
    if frame is not None:
        # NB: frame is PyInterpreterFrame on Python 3.11 and later,
        # not a TRUE frame object.  You can't actually feed it
        # to traceback because it doesn't have enough information.
        # To solve this problem, we technically should just materialize
        # the frame, the same way _PyFrame_GetFrameObject would do
        # (but we cannot actually do this, because this populates
        # frame_obj field, which default eval frame doesn't like).
        #
        # Fortunately, in this case, we can hack it: there's no need
        # to actually use the truly top frame, we can just extract
        # from where we are right now and rely on filter_stack to
        # get rid of all the dynamo frames.  For ease of testing
        # we apply this behavior to ALL Python versions
        stack_above_dynamo = filter_stack(extract_stack())

    return cast(StackSummary, stack_above_dynamo + real_stack)


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


def format_error_msg_verbose(
    exc: Exception, code, record_filename=None, frame=None
) -> str:
    msg = (
        f"WON'T CONVERT {code.co_name} {code.co_filename} line {code.co_firstlineno}\n"
    )
    msg += "=" * 10 + " TorchDynamo Stack Trace " + "=" * 10 + "\n"
    msg += format_exc()
    real_stack = get_real_stack(exc, frame)
    if real_stack is not None:
        msg += (
            "\n"
            + "=" * 10
            + " The above exception occurred while processing the following code "
            + "=" * 10
            + "\n\n"
        )
        msg += "".join(format_list(real_stack))
        msg += "\n"
        msg += "=" * 10

    return msg


def format_error_msg(exc: Exception, code, record_filename=None, frame=None) -> str:
    msg = os.linesep * 2

    if config.verbose:
        msg = format_error_msg_verbose(exc, code, record_filename, frame)
    else:
        msg = f"WON'T CONVERT {code.co_name} {code.co_filename}\
 line {code.co_firstlineno} \ndue to: \n{format_exc()}"

    return msg
