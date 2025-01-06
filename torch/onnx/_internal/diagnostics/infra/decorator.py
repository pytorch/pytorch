# mypy: allow-untyped-defs
from __future__ import annotations

import functools
import logging
import traceback
from typing import Any, Callable

from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, utils


MessageFormatterType = Callable[..., str]


def format_message_in_text(fn: Callable, *args: Any, **kwargs: Any) -> str:
    return f"{formatter.display_name(fn)}. "


def format_exception_in_markdown(exception: Exception) -> str:
    msg_list = ["### Exception log", "```"]
    msg_list.extend(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )
    msg_list.append("```")
    return "\n".join(msg_list)


def format_function_signature_in_markdown(
    fn: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    format_argument: Callable[[Any], str] = formatter.format_argument,
) -> str:
    msg_list = [f"### Function Signature {formatter.display_name(fn)}"]

    state = utils.function_state(fn, args, kwargs)

    for k, v in state.items():
        msg_list.append(f"- {k}: {format_argument(v)}")

    return "\n".join(msg_list)


def format_return_values_in_markdown(
    return_values: Any,
    format_argument: Callable[[Any], str] = formatter.format_argument,
) -> str:
    return f"{format_argument(return_values)}"


ModifierCallableType = Callable[
    [infra.Diagnostic, Callable, tuple[Any, ...], dict[str, Any], Any], None
]


def diagnose_call(
    rule: infra.Rule,
    *,
    level: infra.Level = infra.Level.NONE,
    diagnostic_type: type[infra.Diagnostic] = infra.Diagnostic,
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
            stack: infra.Stack | None = None
            if len(diag.stacks) > 0:
                stack = diag.stacks[0]
                stack.frames.pop(0)

            # set function location
            fn_location = utils.function_location(fn)
            diag.locations.insert(0, fn_location)
            # Add function location to the top of the stack.
            if stack is not None:
                stack.frames.insert(0, infra.StackFrame(location=fn_location))

            with diag.log_section(logging.INFO, "Function Signature"):
                diag.log(
                    logging.INFO,
                    "%s",
                    formatter.LazyString(
                        format_function_signature_in_markdown,
                        fn,
                        args,
                        kwargs,
                        format_argument,
                    ),
                )

            return_values: Any = None
            with ctx.add_inflight_diagnostic(diag) as diag:
                try:
                    return_values = fn(*args, **kwargs)
                    with diag.log_section(logging.INFO, "Return values"):
                        diag.log(
                            logging.INFO,
                            "%s",
                            formatter.LazyString(
                                format_return_values_in_markdown,
                                return_values,
                                format_argument,
                            ),
                        )
                    return return_values
                except Exception as e:
                    diag.log_source_exception(logging.ERROR, e)
                    diag.level = infra.Level.ERROR
                finally:
                    ctx.log_and_raise_if_error(diag)

        return wrapper

    return decorator


# TODO(bowbao): decorator to report only when failed.
