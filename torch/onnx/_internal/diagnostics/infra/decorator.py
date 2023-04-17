import functools
import traceback
from typing import Any, Callable, Dict, Optional, Tuple, Type

from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, utils


MessageFormatterType = Callable[[Callable, Tuple[Any, ...], Dict[str, Any]], str]


@_beartype.beartype
def format_message_in_text(
    fn: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> str:
    return f"{formatter.display_name(fn)}"


@_beartype.beartype
def format_exception_in_markdown(exception: Exception) -> str:
    msg_list = ["### Exception log", "```"]
    msg_list.extend(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )
    msg_list.append("```")
    return "\n".join(msg_list)


@_beartype.beartype
def format_function_signature_in_markdown(
    fn: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    format_argument: Callable[[Any], str] = formatter.format_argument,
) -> str:
    msg_list = [f"### Function Signature {formatter.display_name(fn)}"]

    state = utils.function_state(fn, args, kwargs)

    for k, v in state.items():
        msg_list.append(f"- {k}: {format_argument(v)}")

    return "\n".join(msg_list)


@_beartype.beartype
def format_return_values_in_markdown(
    return_values: Any,
    format_argument: Callable[[Any], str] = formatter.format_argument,
) -> str:
    return f"- Return value: {format_argument(return_values)}"


ModifierCallableType = Callable[
    [infra.Diagnostic, Callable, Tuple[Any, ...], Dict[str, Any], Any], None
]


@_beartype.beartype
def modify_diagnostic(
    diag: infra.Diagnostic,
    fn: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    return_values: Any,
) -> None:
    return


@_beartype.beartype
def diagnose_call(
    get_context: Callable[[], Optional[infra.DiagnosticContext]],
    rule: infra.Rule,
    level: infra.Level = infra.Level.NONE,
    exception_report_level: infra.Level = infra.Level.WARNING,
    diagnostic_type: Type[infra.Diagnostic] = infra.Diagnostic,
    format_argument: Callable[[Any], str] = formatter.format_argument,
    diagnostic_message_formatter: MessageFormatterType = format_message_in_text,
    diagnostic_modifier: ModifierCallableType = modify_diagnostic,
    report_criterion: Callable[
        [Callable, Tuple[Any, ...], Dict[str, Any], Any], bool
    ] = lambda _1, _2, _3, _4: True,
) -> Callable:
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # TODO(bowbao): add switch to disable diagnostics.
            ctx = get_context()
            if ctx is None:
                return fn(*args, **kwargs)

            diag = diagnostic_type(
                rule,
                level,
                diagnostic_message_formatter(fn, args, kwargs),
            )

            # pop the decorator frame
            # TODO(bowbao): by default diagnostic doesn't have stack.
            # So need to check before doing this. Make the code cleaner.
            stack: Optional[infra.Stack] = None
            if len(diag.stacks) > 0:
                stack = diag.stacks[0]
                stack.frames.pop(0)

            # set function location
            fn_location = utils.function_location(fn)
            diag.locations.insert(0, fn_location)
            # Add function location to the top of the stack.
            if stack is not None:
                stack.frames.insert(0, infra.StackFrame(location=fn_location))

            additional_messages = [
                format_function_signature_in_markdown(
                    fn, args, kwargs, format_argument
                ),
            ]

            return_values: Any = None
            report_diagnostic: bool = True
            with ctx.add_inflight_diagnostic(diag) as diag:
                try:
                    return_values = fn(*args, **kwargs)
                    additional_messages.append(
                        format_return_values_in_markdown(return_values, format_argument)
                    )
                    report_diagnostic = report_criterion(
                        fn, args, kwargs, return_values
                    )
                    return return_values
                except Exception as e:
                    # Record exception.
                    report_diagnostic = True
                    diag.level = exception_report_level
                    additional_messages.append(format_exception_in_markdown(e))
                    raise
                finally:
                    if report_diagnostic:
                        diag.with_additional_message(
                            "\n".join(additional_messages).strip()
                        )
                        diagnostic_modifier(diag, fn, args, kwargs, return_values)
                        ctx.add_diagnostic(diag)

        return wrapper

    return decorator


@_beartype.beartype
def diagnose_step(
    get_context: Callable[[], Optional[infra.DiagnosticContext]],
    rule: Optional[infra.Rule] = None,
    message_formatter: MessageFormatterType = format_message_in_text,
    format_argument: Callable[[Any], str] = formatter.format_argument,
) -> Callable:
    """Decorator to log a step in the inflight diagnostic.

    Args:
        get_context: A function that returns the diagnostic context where inflight
            diagnostic is retrieved and modified by the decorator.
        rule: The decorator logs this step to the top inflight diagnostic that matches
            the rule. If None, the top inflight diagnostic in the stack will be picked,
            regardless of its rule.

    Returns:
        A decorator that logs a step in the inflight diagnostic.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ctx = get_context()
            if ctx is None:
                return fn(*args, **kwargs)

            try:
                diag = ctx.inflight_diagnostic(rule=rule)
            except infra.engine.DiagnosticError:
                # TODO(bowbao): this should trigger a built-in diagnostic.
                traceback.print_exc()
                return fn(*args, **kwargs)

            state = utils.function_state(fn, args, kwargs)
            state = {k: format_argument(v) for k, v in state.items()}
            diag.record_python_call(
                fn,
                state,
                message=message_formatter(fn, args, kwargs),
                frames_to_skip=1,
            )

            return_values = fn(*args, **kwargs)
            state["return_values"] = format_argument(return_values)
            return return_values

        return wrapper

    return decorator


# TODO(bowbao): decorator to report only when failed.
