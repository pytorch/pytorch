from __future__ import annotations

import functools
import traceback
from typing import Any, Callable, Dict, Optional, Tuple, Type

from torch.onnx._internal import _beartype
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, utils


MessageFormatterType = Callable[..., str]


@_beartype.beartype
def format_message_in_text(fn: Callable, *args: Any, **kwargs: Any) -> str:
    return f"{formatter.display_name(fn)}. "


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
def diagnose_call(
    rule: infra.Rule,
    *,
    level: infra.Level = infra.Level.NONE,
    diagnostic_type: Type[infra.Diagnostic] = infra.Diagnostic,
    format_argument: Callable[[Any], str] = formatter.format_argument,
    diagnostic_message_formatter: MessageFormatterType = format_message_in_text,
) -> Callable:
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            common_error_message = "diagnose_call can only be applied to callables"
            if not callable(fn):
                raise AssertionError(
                    f"{common_error_message}. Got {type(fn)} instead of callable."
                )
            arg0 = args[0] if len(args) > 0 else None
            if isinstance(ctx := arg0, infra.DiagnosticContext):
                pass
            elif isinstance(
                ctx := getattr(arg0, "diagnostic_context", None),
                infra.DiagnosticContext,
            ):
                pass
            else:
                # NOTE: At decorate time, it can't tell if a callable is function or method.
                # Technically both are regarded as function at that time.
                raise AssertionError(
                    f"{common_error_message}. For {fn}, "
                    f"If it is a function, a DiagnosticContext instance must be present as "
                    f"the first argument. "
                    f"If it is a method, a DiagnosticContext instance must be present as "
                    f"the attribute 'diagnostic_context' of the 'self' argument."
                )

            diag = diagnostic_type(
                rule,
                level,
                diagnostic_message_formatter(fn, *args, **kwargs),
            )

            # pop the decorator frame
            # TODO(bowbao): by default diagnostic doesn't have stack.
            # So need to check before doing this. Make the code cleaner.
            # Option: do not capture stack by default in diagnostic initialization.
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
            with ctx.add_inflight_diagnostic(diag) as diag:
                try:
                    return_values = fn(*args, **kwargs)
                    additional_messages.append(
                        format_return_values_in_markdown(return_values, format_argument)
                    )
                    return return_values
                except Exception as e:
                    # Record exception.
                    diag.level = infra.levels.ERROR
                    # TODO(bowbao): Message emitting api.
                    diag.message = diag.message or ""
                    diag.message += f"Raised from:\n    {type(e).__name__}: {e}"
                    diag.with_source_exception(e)
                    additional_messages.append(format_exception_in_markdown(e))
                finally:
                    diag.with_additional_message("\n".join(additional_messages).strip())
                    ctx.log_and_raise_if_error(diag)

        return wrapper

    return decorator


# TODO(bowbao): decorator to report only when failed.
