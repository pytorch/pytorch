"""Support for torch.library.pallas_op, a structured way to use Pallas/JAX kernels with PyTorch.

This module provides ``pallas_op``, a decorator analogous to ``triton_op`` that
registers a JAX/Pallas function as a custom PyTorch operator.  It wraps
``custom_op`` and additionally registers a fake kernel and a
``FunctionalTensorMode`` decomposition so that ``torch.compile`` and
``torch.export`` work correctly.

The decorated function must use **JAX type annotations** (``jax.Array`` for
tensors, plain Python types for compile-time constants).  ``pallas_op``
automatically infers the PyTorch schema and identifies static arguments from
these annotations.
"""

import inspect
import types
import typing
from collections.abc import Callable, Sequence
from typing import Any, Union

import torch
from torch._library.custom_ops import CustomOpDef
from torch.utils._exposed_in import exposed_in

from .custom_ops import custom_op
from .infer_schema import infer_schema


# ---------------------------------------------------------------------------
# JAX ↔ Torch dtype maps
# ---------------------------------------------------------------------------
# Populated lazily on first use so that importing this module does not
# require JAX to be installed.

_TORCH_TO_JAX_DTYPE_MAP: dict[torch.dtype, Any] | None = None
_JAX_TO_TORCH_DTYPE_MAP: dict[Any, torch.dtype] | None = None


def _ensure_dtype_maps() -> None:
    global _TORCH_TO_JAX_DTYPE_MAP, _JAX_TO_TORCH_DTYPE_MAP
    if _TORCH_TO_JAX_DTYPE_MAP is not None:
        return
    try:
        import jax.numpy as jnp
    except ImportError as e:
        raise ImportError("JAX is required for pallas_op. Please install JAX.") from e

    _TORCH_TO_JAX_DTYPE_MAP = {
        torch.float32: jnp.float32.dtype,
        torch.float64: jnp.float64.dtype,
        torch.float16: jnp.float16.dtype,
        torch.bfloat16: jnp.bfloat16.dtype,
        torch.float8_e4m3fn: jnp.float8_e4m3fn.dtype,
        torch.float8_e5m2: jnp.float8_e5m2.dtype,
        torch.uint8: jnp.uint8.dtype,
        torch.uint16: jnp.uint16.dtype,
        torch.uint32: jnp.uint32.dtype,
        torch.uint64: jnp.uint64.dtype,
        torch.int8: jnp.int8.dtype,
        torch.int16: jnp.int16.dtype,
        torch.int32: jnp.int32.dtype,
        torch.int64: jnp.int64.dtype,
        torch.complex64: jnp.complex64.dtype,
        torch.complex128: jnp.complex128.dtype,
        torch.bool: jnp.bool_.dtype,
    }
    _JAX_TO_TORCH_DTYPE_MAP = {v: k for k, v in _TORCH_TO_JAX_DTYPE_MAP.items()}


def _torch_to_jax_dtype(dtype: torch.dtype) -> Any:
    _ensure_dtype_maps()
    if _TORCH_TO_JAX_DTYPE_MAP is None:
        raise RuntimeError("dtype maps not initialized")
    result = _TORCH_TO_JAX_DTYPE_MAP.get(dtype)
    if result is None:
        raise NotImplementedError(f"Unsupported dtype for pallas_op: {dtype}")
    return result


def _jax_to_torch_dtype(dtype: Any) -> torch.dtype:
    _ensure_dtype_maps()
    if _JAX_TO_TORCH_DTYPE_MAP is None:
        raise RuntimeError("dtype maps not initialized")
    result = _JAX_TO_TORCH_DTYPE_MAP.get(dtype)
    if result is None:
        raise NotImplementedError(f"Unsupported JAX dtype for pallas_op: {dtype}")
    return result


# ---------------------------------------------------------------------------
# JAX ↔ Torch placeholder conversion
# ---------------------------------------------------------------------------


def jax_placeholder(tensor: torch.Tensor) -> Any:
    """Convert a torch.Tensor to a ``jax.ShapeDtypeStruct`` for tracing."""
    import jax

    if not isinstance(tensor, torch.Tensor):
        return tensor
    return jax.ShapeDtypeStruct(tensor.shape, _torch_to_jax_dtype(tensor.dtype))


def torch_placeholder(aval: Any) -> torch.Tensor | None:
    """Convert a JAX abstract value to a torch.Tensor for fake kernel output."""
    import jax

    if not isinstance(aval, jax.core.ShapedArray):
        return aval
    return torch.empty(aval.shape, dtype=_jax_to_torch_dtype(aval.dtype), device="meta")


# ---------------------------------------------------------------------------
# Signature introspection
# ---------------------------------------------------------------------------


def _is_jax_array(annotation: type[Any]) -> bool:
    try:
        import jax

        return annotation is jax.Array
    except ImportError:
        return False


def _is_valid_base_type(annotation: type[Any]) -> bool:
    return _is_jax_array(annotation) or annotation in (int, float, bool, str)


def _get_underlying_type_from_optional(typ: type[Any]) -> type[Any] | None:
    """Return the underlying type of an Optional, or None if not Optional."""
    if typing.get_origin(typ) not in (Union, types.UnionType):
        return None
    union_types = typing.get_args(typ)
    concrete_args = [t for t in union_types if t is not types.NoneType]
    if len(concrete_args) != 1:
        return None
    return concrete_args[0]


def _is_valid_argument_type(annotation: type[Any]) -> bool:
    if (underlying := _get_underlying_type_from_optional(annotation)) is not None:
        return _is_valid_base_type(underlying)
    return _is_valid_base_type(annotation)


def _is_valid_argument(param: inspect.Parameter) -> bool:
    if (
        param.kind is inspect.Parameter.KEYWORD_ONLY
        and param.default is not inspect.Parameter.empty
    ):
        return True
    return _is_valid_argument_type(param.annotation)


def _verify_signature(signature: inspect.Signature) -> None:
    """Verify that the signature only contains supported types.

    Supported types are ``jax.Array``, ``int``, ``float``, ``bool``, ``str``
    (and Optional variants thereof).
    """
    for param in signature.parameters.values():
        if param.annotation is inspect.Parameter.empty:
            raise ValueError(
                f"Missing argument type annotation for JAX function: {signature}."
            )

    def is_valid_result(result: Any) -> bool:
        if typing.get_origin(result) is tuple:
            return all(_is_valid_base_type(arg) for arg in typing.get_args(result))
        return _is_valid_base_type(result)

    invalid_arg_indices = [
        idx
        for idx, param in enumerate(signature.parameters.values())
        if not _is_valid_argument(param)
    ]
    result_valid = signature.return_annotation is None or is_valid_result(
        signature.return_annotation
    )

    if not invalid_arg_indices and result_valid:
        return

    error_messages: list[str] = []
    if invalid_arg_indices:
        error_messages.append(
            f"Arguments at indices {invalid_arg_indices} are invalid."
        )
    if not result_valid:
        error_messages.append("The return annotation is invalid.")

    raise ValueError(
        f"Invalid signature for JAX function: {signature}. "
        f"{' '.join(error_messages)} Only jax.Arrays and POD types are supported."
    )


def _infer_static_argnums(signature: inspect.Signature) -> tuple[int, ...]:
    """Infer which arguments are compile-time constants (non-tensor)."""
    static_argnums = []
    for idx, param in enumerate(signature.parameters.values()):
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            continue
        if _is_jax_array(param.annotation):
            continue
        if (
            _get_underlying_type_from_optional(param.annotation)
            and _is_jax_array(_get_underlying_type_from_optional(param.annotation))  # type: ignore[arg-type]
        ):
            continue
        static_argnums.append(idx)
    return tuple(static_argnums)


def _get_torch_signature(signature: inspect.Signature) -> inspect.Signature:
    """Convert a JAX-annotated signature to a torch-annotated one."""

    def _map_jax_to_torch(typ: Any) -> Any:
        if _is_jax_array(typ):
            return torch.Tensor
        if (underlying := _get_underlying_type_from_optional(typ)) is not None:
            return _map_jax_to_torch(underlying) | types.NoneType
        if typing.get_origin(typ) is tuple:
            mapped_args = tuple(
                _map_jax_to_torch(arg) for arg in typing.get_args(typ)
            )
            return tuple.__class_getitem__(mapped_args)
        return typ

    new_parameters = []
    for param in signature.parameters.values():
        if not _is_valid_argument_type(param.annotation):
            continue
        new_parameters.append(
            param.replace(annotation=_map_jax_to_torch(param.annotation))
        )
    new_return_annotation = _map_jax_to_torch(signature.return_annotation)
    return inspect.Signature(new_parameters, return_annotation=new_return_annotation)


# ---------------------------------------------------------------------------
# pallas_op
# ---------------------------------------------------------------------------


@exposed_in("torch.library")
def pallas_op(
    name: str,
    fn: Callable | None = None,
    /,
    *,
    donate_argnums: Sequence[int] | None = None,
    schema: str | None = None,
) -> Callable:
    """Create a custom operator whose implementation is a JAX/Pallas kernel.

    This is the Pallas/TPU analog of :func:`torch.library.triton_op`.  It wraps
    a JAX function as a :func:`torch.library.custom_op` and handles fake-kernel
    registration and ``FunctionalTensorMode`` decomposition automatically.

    The decorated function must use **JAX type annotations**: ``jax.Array`` for
    tensor arguments, and ``int`` / ``float`` / ``bool`` / ``str`` for
    compile-time constants.  The schema is inferred automatically.

    JAX functions are inherently immutable — even donated arguments produce new
    outputs rather than mutating inputs.  ``mutates_args`` is therefore always
    ``()``.  Use ``donate_argnums`` to indicate arguments whose backing storage
    may be reused by the kernel; donated tensors are left in an invalid state
    after the call.

    Args:
        name (str): Operator name in ``"namespace::name"`` format.
        fn: The JAX/Pallas function.  If ``None``, ``pallas_op`` returns a
            decorator.
        donate_argnums: Indices of arguments whose storage may be donated to
            the kernel.  Donated tensors must not be read after the call.
        schema: An explicit schema string.  If ``None`` (recommended) the
            schema is inferred from the function's type annotations.

    Example::

        >>> import jax
        >>> import torch
        >>> from torch.library import pallas_op
        >>>
        >>> @pallas_op("mylib::add_vectors")
        ... def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
        ...     return x + y

    """
    if "::" not in name or len(name.split("::")) != 2:
        raise ValueError(f"Op name must be in 'namespace::name' format, got: {name}")

    def dec(fn: Callable[..., object]) -> CustomOpDef:
        signature = inspect.signature(fn, follow_wrapped=False)
        _verify_signature(signature)
        static_argnums = _infer_static_argnums(signature)
        torch_sig = _get_torch_signature(signature)

        # Create a thin wrapper whose __signature__ is the torch-typed version
        # so that infer_schema produces the correct schema.
        def _torch_sig_fn(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        _torch_sig_fn.__signature__ = torch_sig  # type: ignore[attr-defined]

        result = custom_op(
            name,
            fn,
            mutates_args=(),
            schema=schema or infer_schema(_torch_sig_fn, mutates_args=()),
        )

        # -- Fake kernel via JAX tracing --------------------------------
        def fake_fn(*args: Any, **kwargs: Any) -> Any:
            for arg in args:
                if isinstance(arg, torch.Tensor) and any(
                    isinstance(d, torch.SymInt) for d in arg.shape
                ):
                    raise RuntimeError(
                        "Symbolic dimensions are not supported in the default "
                        "pallas_op fake kernel. Either remove symbolic "
                        "dimensions or override this default implementation "
                        "via result.register_fake()."
                    )

            import jax
            import jax.export

            jax_args = [
                jax_placeholder(a) if isinstance(a, torch.Tensor) else a
                for a in args
            ]

            jit_fn = jax.jit(fn, static_argnums=static_argnums)
            exported = jax.export.export(jit_fn, platforms=["tpu"])
            lowered = exported(*jax_args, **kwargs)

            return lowered.out_tree.unflatten(
                torch_placeholder(aval) for aval in lowered.out_avals
            )

        result.register_fake(fake_fn)

        # Store metadata for downstream consumers.
        result._pallas_donate_argnums = donate_argnums or []  # type: ignore[attr-defined]
        result._pallas_static_argnums = static_argnums  # type: ignore[attr-defined]

        return result

    if fn is None:
        return dec
    else:
        return dec(fn)
