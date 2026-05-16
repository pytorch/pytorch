import inspect
from typing import Any


def _signature_metadata(
    sig: inspect.Signature,
) -> tuple[tuple[inspect.Parameter, ...], bool, int]:
    """
    Returns tuple(sig.parameters.values()), if any has VAR_POSITIONAL or VAR_KEYWORD, and the max_positional
    """
    params = tuple(sig.parameters.values())
    has_var_args = False
    max_positional = 0

    for p in params:
        kind = p.kind
        if kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            has_var_args = True
        if kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            max_positional += 1

    return params, has_var_args, max_positional


def _fast_bind(
    sig: inspect.Signature, *args: Any, **kwargs: Any
) -> inspect.BoundArguments:
    """
    Fast path for inspect.Signature.bind() for signatures without
    VAR_POSITIONAL or VAR_KEYWORD parameters. Falls back to sig.bind()
    for signatures that contain *args or **kwargs.
    """
    params, has_var_args, max_positional = _signature_metadata(sig)

    # fallback for complex signatures
    if has_var_args:
        return sig.bind(*args, **kwargs)

    len_args = len(args)

    if len_args > max_positional:
        raise TypeError(
            f"Too many positional arguments: expected max {max_positional}, got {len_args}"
        )

    arguments: dict[str, Any] = {}
    arg_i = 0

    for p in params:
        name = p.name
        kind = p.kind

        if kind is inspect.Parameter.POSITIONAL_ONLY:
            if name in kwargs:
                raise TypeError(
                    f"Got some positional-only arguments passed as keyword arguments: '{name}'"
                )
            if arg_i < len_args:
                arguments[name] = args[arg_i]
                arg_i += 1
            elif p.default is inspect.Parameter.empty:
                raise TypeError(f"Missing required argument '{name}'")

        elif kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if arg_i < len_args:
                if name in kwargs:
                    raise TypeError(f"Multiple values for argument '{name}'")
                arguments[name] = args[arg_i]
                arg_i += 1
            elif name in kwargs:
                arguments[name] = kwargs[name]
            elif p.default is inspect.Parameter.empty:
                raise TypeError(f"Missing required argument '{name}'")

        elif kind is inspect.Parameter.KEYWORD_ONLY:
            if name in kwargs:
                arguments[name] = kwargs[name]
            elif p.default is inspect.Parameter.empty:
                raise TypeError(f"Missing required argument '{name}'")

    # disallow extra keyword arguments not in the signature
    # cause kwargs have been processed by sig.bind at the beginning
    for name in kwargs:
        if name not in sig.parameters:
            raise TypeError(f"Got an unexpected keyword argument '{name}'")

    return inspect.BoundArguments(sig, arguments)  # type: ignore[arg-type]
