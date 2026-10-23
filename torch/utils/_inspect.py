import inspect
import types
import weakref
from collections.abc import Callable, Mapping
from typing import Any


_Parameters = Mapping[str, inspect.Parameter]
# Weakly keyed, so callers can pass short-lived functions such as closures
_function_parameters: weakref.WeakKeyDictionary[types.FunctionType, _Parameters] = (
    weakref.WeakKeyDictionary()
)
_method_parameters: weakref.WeakKeyDictionary[types.FunctionType, _Parameters] = (
    weakref.WeakKeyDictionary()
)


def _cached_parameters(fn: Callable[..., Any]) -> _Parameters:
    """
    Returns inspect.signature(fn).parameters, cached for Python functions and
    methods. A bound method's signature depends only on its underlying function,
    so methods are cached by that function, separately from the function itself,
    whose signature also includes self.
    """
    is_method = isinstance(fn, types.MethodType)
    func = fn.__func__ if is_method else fn
    if not isinstance(func, types.FunctionType):
        return inspect.signature(fn).parameters
    cache = _method_parameters if is_method else _function_parameters
    params = cache.get(func)
    if params is None:
        params = cache[func] = inspect.signature(fn).parameters
    return params


_SignatureMetadata = tuple[tuple[inspect.Parameter, ...], bool, int]

# Keyed by id, since signatures can't be weakly referenced. Each entry keeps its
# signature alive, so the id can't be reused while the entry exists.
_metadata_cache: dict[int, tuple[inspect.Signature, _SignatureMetadata]] = {}
_METADATA_CACHE_SIZE = 1024


def _signature_metadata(sig: inspect.Signature) -> _SignatureMetadata:
    """
    Returns tuple(sig.parameters.values()), if any has VAR_POSITIONAL or VAR_KEYWORD, and the max_positional
    """
    entry = _metadata_cache.get(id(sig))
    if entry is not None:
        return entry[1]
    metadata = _compute_signature_metadata(sig)
    # Some callers build a new signature for each call, so bound the cache
    if len(_metadata_cache) >= _METADATA_CACHE_SIZE:
        _metadata_cache.clear()
    _metadata_cache[id(sig)] = (sig, metadata)
    return metadata


def _compute_signature_metadata(sig: inspect.Signature) -> _SignatureMetadata:
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
    sig: inspect.Signature, *args: object, **kwargs: object
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

    arguments: dict[str, object] = {}
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
