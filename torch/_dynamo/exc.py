from __future__ import annotations


"""Exception handling and error reporting for TorchDynamo.

This module provides a comprehensive set of exception classes and utilities for error
handling in TorchDynamo. It includes:

Base Exceptions:
    - TorchDynamoException: Base class for all TorchDynamo-specific exceptions
    - Various specialized subclasses for different error scenarios

User Error Handling:
    - UserError: Exceptions for user-facing errors in TorchDynamo usage
    - UserErrorType: Enumeration of different categories of user errors
    - Formatted error messages with debugging information

Observed Exceptions:
    - Classes for handling exceptions observed during tracing
    - Special handling for StopIteration, LookupError, etc.
    - Exception state management during compilation

Error Formatting:
    - Stack trace filtering and formatting
    - Error message augmentation
    - Debugging utilities for error reporting
"""

import json
import logging
import os
import re
import textwrap
import typing
from enum import auto, Enum
from functools import lru_cache
from pathlib import Path
from traceback import extract_stack, format_exc, format_list, StackSummary
from typing import Any, NoReturn, Optional, TYPE_CHECKING

import torch._guards

from . import config
from .utils import counters


if TYPE_CHECKING:
    import types

    from torch._guards import CompileId

    from .symbolic_convert import InstructionTranslatorBase
    from .types import DynamoFrameType


def exportdb_error_message(case_name: str) -> str:
    return (
        "For more information about this error, see: "
        + "https://pytorch.org/docs/main/generated/exportdb/index.html#"
        + case_name.replace("_", "-")
    )


log = logging.getLogger(__name__)
graph_breaks_log = torch._logging.getArtifactLogger(__name__, "graph_breaks")


class TorchDynamoException(RuntimeError):
    pass


class InternalTorchDynamoError(TorchDynamoException):
    pass


class ResumePrologueTracingError(TorchDynamoException):
    pass


class RestartAnalysis(TorchDynamoException):
    restart_reason: Optional[str]

    def __init__(self, *args: Any, restart_reason: Optional[str] = None) -> None:
        self.restart_reason = restart_reason
        super().__init__(*args)


class SpeculationRestartAnalysis(RestartAnalysis):
    pass


class UnspecializeRestartAnalysis(RestartAnalysis):
    pass


class CompileCollectiveRestartAnalysis(RestartAnalysis):
    pass


class TensorifyScalarRestartAnalysis(RestartAnalysis):
    pass


class SkipFrame(TorchDynamoException):
    pass


class TorchRuntimeError(TorchDynamoException):
    pass


class InvalidBackend(TorchDynamoException):
    def __init__(self, name: str) -> None:
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


class ShortenTraceback(TorchDynamoException):
    def __init__(
        self, *args: Any, first_useful_frame: Optional[types.FrameType], **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.first_useful_frame = first_useful_frame

    def remove_dynamo_frames(self) -> typing.Self:
        tb = self.__traceback__
        if self.first_useful_frame is None or tb is None or config.verbose:
            return self
        while tb.tb_frame is not self.first_useful_frame:
            tb = tb.tb_next
            assert tb is not None, "internal error, please report a bug"
        return self.with_traceback(tb)


class BackendCompilerFailed(ShortenTraceback):
    def __init__(
        self,
        backend_fn: Any,
        inner_exception: Exception,
        first_useful_frame: Optional[types.FrameType],
    ) -> None:
        self.backend_name = getattr(backend_fn, "__name__", "?")
        self.inner_exception = inner_exception
        msg = f"backend={self.backend_name!r} raised:\n{type(inner_exception).__name__}: {inner_exception}"
        super().__init__(msg, first_useful_frame=first_useful_frame)


class Unsupported(TorchDynamoException):
    def __init__(self, msg: str, *, case_name: Optional[str] = None) -> None:
        super().__init__(msg)
        self.real_stack = torch._guards.TracingContext.extract_stack()
        self.msg = msg
        self.category: Optional[str] = None
        self.add_to_stats()
        self.case_name: Optional[str] = case_name

    def remove_from_stats(self) -> None:
        assert self.category is not None
        counters[self.category][self.msg] -= 1
        if counters[self.category][self.msg] <= 0:
            del counters[self.category][self.msg]

    def add_to_stats(self, category: str = "unimplemented") -> None:
        self.category = category
        counters[category][self.msg] += 1


class UnknownPropertiesDuringBackwardTrace(Unsupported):
    pass


class RecompileError(TorchDynamoException):
    pass


class ArgsMismatchError(Unsupported):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class AttributeMutationError(Unsupported):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class InfiniteGeneratorError(Unsupported):
    # Raised when the number of yielded values is greater than MAX_ITERATOR_LIMIT
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class SideEffectsError(Unsupported):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class CondOpArgsMismatchError(ArgsMismatchError):
    """
    Internal error from cond() due to arguments mismatch.
    """

    def __init__(self, msg: str) -> None:
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
    def __init__(
        self, error_type: UserErrorType, msg: str, case_name: Optional[str] = None
    ) -> None:
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


class RecompileLimitExceeded(Unsupported):
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
class FailOnRecompileLimitHit(Exception):
    pass


class PackageError(TorchDynamoException):
    pass


class ObservedException(TorchDynamoException):
    # An exception observed during the tracing. This exception is used by Dynamo to handle exceptions.
    pass


class ObservedUserStopIteration(ObservedException):
    # An UserStopIteration exception observed during the Dynamo tracing (e.g Dynamo tracing __next__)
    value: Optional[Any]

    # Reference `StopIteration_init` in CPython
    # https://github.com/python/cpython/blob/3.11/Objects/exceptions.c#L568-L584
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("unhandled `raise StopIteration`")
        if len(args) > 0:
            self.value = args[0]
        else:
            self.value = None


class ObservedLookupError(ObservedException):
    # A LookupError exception to be raised from inside Dynamo tracing. This can happen on __getitem__
    pass


class ObservedIndexError(ObservedLookupError):
    # An IndexError exception to be raised from inside Dynamo tracing. This can happen on list __getitem__
    pass


class ObservedKeyError(ObservedLookupError):
    # A KeyError exception to be raised from inside Dynamo tracing. This can happen on dict __getitem__
    pass


class ObservedGeneratorExit(ObservedException):
    pass


class ObservedAttributeError(ObservedException):
    # An AttributeError exception to be raised from inside Dynamo tracing. This can happen on user defined object __getattr__
    pass


class ObservedRuntimeError(ObservedException):
    # A RuntimeError exception to be raised from inside Dynamo tracing. This can happen on generator.throw(..) method
    pass


class ObservedNotImplementedError(ObservedException):
    pass


class ObservedTypeError(ObservedException):
    # A TypeError exception to be raised from inside Dynamo tracing. This can happen on generator.send(..) method
    pass


observed_exception_map = {
    StopIteration: ObservedUserStopIteration,
    LookupError: ObservedLookupError,
    IndexError: ObservedIndexError,
    GeneratorExit: ObservedGeneratorExit,
    KeyError: ObservedKeyError,
    AttributeError: ObservedAttributeError,
    RuntimeError: ObservedRuntimeError,
    NotImplementedError: ObservedNotImplementedError,
    TypeError: ObservedTypeError,
}


def get_dynamo_observed_exception(exc_type: type[Exception]) -> type[ObservedException]:
    if exc_type not in observed_exception_map:
        name = getattr(exc_type, "__name__", str(exc_type))
        observed_exception_map[exc_type] = type(
            f"Observed{name}Error", (ObservedException,), {}
        )
    return observed_exception_map[exc_type]


def raise_observed_exception(
    exc_type: type[Exception],
    tx: InstructionTranslatorBase,
    *,
    args: Optional[list[Any]] = None,
    kwargs: Optional[dict[str, Any]] = None,
) -> NoReturn:
    from .variables import BuiltinVariable

    # CPython here raises an exception. Since there is no python code, we have to manually setup the exception
    # stack and raise the exception.
    exception_vt = BuiltinVariable(exc_type).call_function(tx, args or [], kwargs or {})  # type: ignore[arg-type]
    tx.exn_vt_stack.set_current_exception(exception_vt)
    raise get_dynamo_observed_exception(exc_type)


def handle_observed_exception(tx: Any) -> None:
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
    tx.exn_vt_stack.clear_current_exception()


# These exceptions are ok to fallback to eager/graph_break.
exceptions_allowed_to_be_fallback = (
    torch._subclasses.fake_tensor.DataDependentOutputException,
    torch._subclasses.fake_tensor.DynamicOutputShapeException,
    torch._subclasses.fake_tensor.UnsupportedOperatorException,
    torch._subclasses.fake_tensor.UnsupportedFakeTensorException,
    torch._subclasses.fake_tensor.UnsupportedMutationAliasingException,
)


def unimplemented_with_warning(
    e: Exception, code: types.CodeType, msg: str
) -> NoReturn:
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
    msg: str, *, from_exc: Any = _NOTHING, case_name: Optional[str] = None
) -> NoReturn:
    assert msg != os.environ.get("BREAK", False)
    if from_exc is not _NOTHING:
        raise Unsupported(msg, case_name=case_name) from from_exc
    raise Unsupported(msg, case_name=case_name)


def unimplemented_v2_with_warning(
    e: Exception,
    code: types.CodeType,
    gb_type: str,
    context: str,
    explanation: str,
    hints: list[str],
) -> NoReturn:
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
    unimplemented_v2(gb_type, context, explanation, hints, from_exc=e, log_warning=True)


def format_graph_break_message(
    gb_type: str,
    context: str,
    explanation: str,
    hints: list[str],
) -> str:
    explanation = textwrap.indent(explanation, "    ").lstrip()
    hints_str = "\n".join(
        "  Hint: " + textwrap.indent(hint, "    ").lstrip() for hint in hints
    )
    context = textwrap.indent(context, "    ").lstrip()

    msg = f"""\
{gb_type}
  Explanation: {explanation}
{hints_str}

  Developer debug context: {context}
"""
    return msg


@lru_cache(maxsize=1)
def _load_graph_break_registry() -> dict[str, Any]:
    """
    Loads the graph break registry from JSON file with caching.
    """
    try:
        script_dir = Path(__file__).resolve().parent
        registry_path = script_dir / "graph_break_registry.json"
        with registry_path.open() as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.error("Error accessing the registry file: %s", e)
        return {}


def get_gbid_documentation_link(gb_type: str) -> Optional[str]:
    """
    Retrieves the GBID documentation link for a given graph break type.

    Args:
        gb_type: The graph break type to look up.

    Returns:
        A string containing the documentation URL if found, otherwise None.
    """
    GRAPH_BREAK_SITE_URL = (
        "https://pytorch-labs.github.io/compile-graph-break-site/gb/"  # @lint-ignore
    )

    registry = _load_graph_break_registry()

    for k, v in registry.items():
        if v and v[0].get("Gb_type") == gb_type:
            return f"{GRAPH_BREAK_SITE_URL}gb{k.lstrip('GB')}.html"

    return None


# TODO replace old unimplemented later
def unimplemented_v2(
    gb_type: str,
    context: str,
    explanation: str,
    hints: list[str],
    *,
    from_exc: Any = _NOTHING,
    log_warning: bool = False,
) -> NoReturn:
    """
    Called within dynamo to cause a graph break.
    Args:
        gb_type: Context-free graph break type. It should be a short string without any
                 information specific to the tracing context (i.e. no dynamically-generated strings)
        context: Developer context for the graph break. It can contain tracing context/dynamic strings.
        explanation: User-facing context-dependent explanation for the graph break. Can be dynamic.
        hints: List of user-facing hints for the graph break.
    """

    msg = format_graph_break_message(gb_type, context, explanation, hints)

    documentation_link = get_gbid_documentation_link(gb_type)

    if documentation_link:
        msg += f"\n For more details about this graph break, please visit: {documentation_link}"

    if log_warning:
        log.warning(msg)
    if from_exc is not _NOTHING:
        raise Unsupported(msg) from from_exc
    raise Unsupported(msg)


# KeyError has special handling for its args
# see https://github.com/python/cpython/blob/3.11/Objects/exceptions.c#L2534 for details
class KeyErrorMsg:
    def __init__(self, value: Any) -> None:
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
        msg += (
            f"\nLast frame execution written to {exc.record_filename}. To run only this frame while debugging, run\
 torch._dynamo.replay('{exc.record_filename}').\n"
        )

    if not config.verbose and hasattr(exc, "real_stack"):
        msg += (
            "\nSet TORCHDYNAMO_VERBOSE=1 for the internal stack trace "
            "(please do this especially if you're reporting a bug to PyTorch). "
            'For even more developer context, set TORCH_LOGS="+dynamo"\n'
        )

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

    old_msg = "" if len(exc.args) == 0 else str(exc.args[0])

    if isinstance(exc, KeyError):
        exc.args = (KeyErrorMsg(old_msg + msg),) + exc.args[1:]
    else:
        new_msg = old_msg + msg
        exc.args = (new_msg,) + exc.args[1:]


def get_exc_message(
    e: Exception, compile_id: CompileId
) -> tuple[Optional[str], Optional[int]]:
    filename = None
    lineno = None
    if e.innermost_user_frame_summary is not None:  # type: ignore[attr-defined]
        filename = e.innermost_user_frame_summary.filename  # type: ignore[attr-defined]
        lineno = e.innermost_user_frame_summary.lineno  # type: ignore[attr-defined]
    e.compile_id = compile_id  # type: ignore[attr-defined]
    return filename, lineno


def get_stack_above_dynamo() -> StackSummary:
    return filter_stack(extract_stack())


def get_real_stack(
    exc: Exception, frame: Optional[DynamoFrameType] = None
) -> Optional[StackSummary]:
    real_stack = getattr(exc, "real_stack", None)
    if real_stack is None:
        return None

    # NB: it's possible for real_stack to be []; we still attempt to
    # report a stack anyway because the stack_above_dynamo may still
    # be useful for debugging

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
        stack_above_dynamo = get_stack_above_dynamo()
    else:
        stack_above_dynamo = StackSummary()

    return StackSummary.from_list(stack_above_dynamo + real_stack)


# filter out all frames after entering dynamo
def filter_stack(stack: StackSummary) -> StackSummary:
    user_stack = StackSummary()
    for frame in stack:
        if frame.filename is None:
            continue
        if "convert_frame" in frame.filename:
            break
        if "eval_frame" in frame.filename or (
            frame.line and "torch._dynamo.optimize(" in frame.line
        ):
            continue
        user_stack.append(frame)

    return user_stack


def remove_resume_prefix(name: str) -> Optional[str]:
    from .resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX

    match = re.match(f"{TORCH_DYNAMO_RESUME_IN_PREFIX}_(\\w+)_at_\\d+", name)
    if match:
        return match.group(1)
    return None


def collapse_resume_frames(stack: StackSummary) -> StackSummary:
    """
    When we graph break, we create a resume function and make a regular Python call
    to it, which gets intercepted by Dynamo. This behavior is normally shown in the
    traceback, which can be confusing to a user. So we can filter out resume frames
    for better traceback clarity.

    Example:
    File "..." line 3, in f
        <line 3>
    File "..." line 5, in torch_dynamo_resume_in_f_at_80
        <line 5>
    File "..." line 10, in torch_dynamo_resume_in_f_at_120
        <line 10>

    becomes
    File "..." line 10, in f
        <line 10>
    """

    new_stack = StackSummary()
    for frame in stack:
        if frame.filename is None:
            continue
        name = remove_resume_prefix(frame.name)
        if new_stack and name and new_stack[-1].name == name:
            new_stack[-1] = frame
            frame.name = name
        else:
            new_stack.append(frame)

    return new_stack


def format_error_msg_verbose(
    exc: Exception,
    code: types.CodeType,
    record_filename: Optional[str] = None,
    frame: Optional[DynamoFrameType] = None,
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


def format_error_msg(
    exc: Exception,
    code: types.CodeType,
    record_filename: Optional[str] = None,
    frame: Optional[DynamoFrameType] = None,
) -> str:
    if config.verbose:
        return format_error_msg_verbose(exc, code, record_filename, frame)
    return f"WON'T CONVERT {code.co_name} {code.co_filename}\
 line {code.co_firstlineno} \ndue to: \n{format_exc()}"
