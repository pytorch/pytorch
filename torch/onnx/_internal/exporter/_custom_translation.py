"""Process-wide native ONNX translations, snapshotted for each export."""

from __future__ import annotations

import dataclasses
import inspect
import threading
from collections.abc import Callable, Sequence
from typing import Any, overload, TYPE_CHECKING, TypeVar

import torch


if TYPE_CHECKING:
    import onnx_ir as ir

    from torch.onnx._internal.exporter._ir_builder import GraphBuilder


_F = TypeVar("_F", bound=Callable[..., Any])


@dataclasses.dataclass(frozen=True)
class NativeTranslation:
    target: Callable | str
    function: Callable
    is_complex: bool
    opset_introduced: int
    signature: ir.schemas.OpSignature | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.signature is None:
            return self.function(*args, **kwargs)
        import onnx_ir as ir

        arguments = inspect.signature(self.function).bind(*args, **kwargs)
        arguments.apply_defaults()
        bindings: dict[ir.schemas.TypeConstraintParam, ir.TypeProtocol] = {}

        def check_value(param: ir.schemas.Parameter, value: Any) -> None:
            if not isinstance(value, ir.Value):
                return
            constraint = param.type_constraint
            if value.type is None or value.type not in constraint.allowed_types:
                raise TypeError(
                    f"Translation value {param.name!r} has type {value.type}, "
                    f"outside constraint {constraint}. Insert an explicit Cast "
                    "or register a signature accepting this value type."
                )
            if param.homogeneous:
                previous = bindings.setdefault(constraint, value.type)
                if value.type != previous:
                    raise TypeError(
                        f"Translation value {param.name!r} has type {value.type}, "
                        f"but shared constraint {constraint.name!r} binds {previous}. "
                        "Insert explicit tensor casts; native translations do not promote tensors."
                    )

        for param in self.signature.params:
            if isinstance(param, ir.schemas.Parameter):
                check_value(param, arguments.arguments.get(param.name))
        outputs = self.function(*args, **kwargs)
        values = outputs if isinstance(outputs, Sequence) else [outputs]
        for param, value in zip(self.signature.outputs, values):
            check_value(param, value)
        return outputs


_LOCK = threading.Lock()
_TRANSLATIONS: dict[tuple[Callable, bool, int], NativeTranslation] = {}


def _validate_key(target: Callable, is_complex: bool, opset_version: int) -> None:
    if isinstance(target, torch._ops.OpOverloadPacket) or not callable(target):
        raise TypeError(
            f"Translation target must be an ATen overload or callable, got {target!r}. "
            "Use torch.ops.<namespace>.<op>.default instead of an overload packet."
        )
    try:
        hash(target)
    except TypeError as error:
        raise TypeError(
            "Translation target must be hashable. Pass an ATen overload or a "
            "hashable callable matching the FX node target."
        ) from error
    if not isinstance(is_complex, bool):
        raise TypeError("is_complex must be a bool; use True for complex inputs.")
    if type(opset_version) is not int or opset_version < 1:
        raise ValueError(
            f"opset_version must be a positive integer minimum opset, got {opset_version!r}."
        )


@overload
def register_custom_translation(
    target: Callable,
    function: _F,
    *,
    opset_version: int = 18,
    is_complex: bool = False,
    replace: bool = False,
    signature: ir.schemas.OpSignature | None = None,
) -> _F: ...


@overload
def register_custom_translation(
    target: Callable,
    function: None = None,
    *,
    opset_version: int = 18,
    is_complex: bool = False,
    replace: bool = False,
    signature: ir.schemas.OpSignature | None = None,
) -> Callable[[_F], _F]: ...


def register_custom_translation(
    target: Callable,
    function: _F | None = None,
    *,
    opset_version: int = 18,
    is_complex: bool = False,
    replace: bool = False,
    signature: ir.schemas.OpSignature | None = None,
) -> _F | Callable[[_F], _F]:
    """Register a native translation for subsequent ``dynamo=True`` exports.

    Use as a decorator or pass ``function`` directly. Both forms return the
    original callable. The translation receives ONNX IR values and ordinary
    Python attributes and returns an IR value or a sequence of IR values.
    Obtain the scoped builder with :func:`torch.onnx.get_translation_builder`.
    No ONNX Script evaluation is installed for native translations.

    Args:
        target: Hashable callable FX target, such as ``torch.ops.aten.add.Tensor``.
            Overload packets and operator-name strings are not accepted.
        function: Plain Python translation callable, or None for decorator use.
        opset_version: Minimum supported ONNX opset. The highest compatible
            registration is selected independently for real and complex inputs.
        is_complex: Whether this variant translates complex-valued inputs.
        replace: Replace only the exact (target, is_complex, opset_version) key.
            Otherwise duplicate registration raises ValueError.
        signature: Optional native ``onnx_ir.schemas.OpSignature`` metadata.
            Tensor constraints can be expressed using IR types, without ONNX
            Script annotations. Builder calls validate ONNX operator constraints.

    Export-local ``custom_translation_table`` entries take precedence over these
    registrations, which take precedence over built-ins. An export snapshots
    registrations before running model code. Updates do not affect that export.
    Registration does not import ONNX Script or populate the built-in library.

    Example::

        @torch.onnx.register_custom_translation(torch.ops.aten.add.Tensor)
        def custom_add(self, other, alpha=1):
            builder = torch.onnx.get_translation_builder()
            if alpha != 1:
                (other,) = builder.call("Mul", other, alpha)
            return builder.call("Add", self, other)[0]


        # Remove exactly the default real/opset-18 variant.
        torch.onnx.unregister_custom_translation(torch.ops.aten.add.Tensor)
    """
    _validate_key(target, is_complex, opset_version)
    if not isinstance(replace, bool):
        raise TypeError(
            "replace must be a bool; use True to replace this exact variant."
        )
    if signature is not None:
        import onnx_ir as ir

        if not isinstance(signature, ir.schemas.OpSignature):
            raise TypeError(
                "signature must be onnx_ir.schemas.OpSignature or None. "
                "Use native IR parameter/type constraints, not ONNX Script types."
            )

    def register(translation: _F) -> _F:
        if not callable(translation):
            raise TypeError(
                f"Translation must be callable, got {translation!r}. "
                "Pass a Python function that returns ONNX IR values."
            )
        if signature is not None:
            parameters = inspect.signature(translation).parameters
            unknown = {param.name for param in signature.params} - parameters.keys()
            if unknown:
                raise ValueError(
                    f"Signature parameters {sorted(unknown)} are absent from "
                    f"translation {translation!r}. Match IR signature names to "
                    "the Python callable's parameters."
                )
        key = (target, is_complex, opset_version)
        entry = NativeTranslation(
            target, translation, is_complex, opset_version, signature
        )
        with _LOCK:
            if key in _TRANSLATIONS and not replace:
                raise ValueError(
                    f"A translation for {target!r}, is_complex={is_complex}, "
                    f"opset_version={opset_version} is already registered. "
                    "Use replace=True to replace only this variant, or unregister it first."
                )
            _TRANSLATIONS[key] = entry
        return translation

    return register if function is None else register(function)


def unregister_custom_translation(
    target: Callable, *, opset_version: int = 18, is_complex: bool = False
) -> None:
    """Remove exactly one globally registered native translation.

    Other complex/opset variants and built-ins are unchanged. Raises KeyError
    if the key is absent; use the same metadata passed during registration.
    In-flight exports retain their snapshots.
    """
    _validate_key(target, is_complex, opset_version)
    key = (target, is_complex, opset_version)
    with _LOCK:
        if key not in _TRANSLATIONS:
            raise KeyError(
                f"No global translation for {target!r}, is_complex={is_complex}, "
                f"opset_version={opset_version}. Use the metadata from registration."
            )
        del _TRANSLATIONS[key]


def snapshot() -> tuple[NativeTranslation, ...]:
    with _LOCK:
        return tuple(_TRANSLATIONS.values())


def get_translation_builder() -> GraphBuilder:
    """Return the active native ONNX translation builder.

    Only valid inside a native translation (or a builder's ``activate`` scope).
    The builder provides ``call``, ``constant``, ``node`` and ``call_function``.
    ``call`` converts Python constants according to ONNX schemas; tensor casts,
    symbolic shape/list construction and control flow must be explicit.
    """
    from torch.onnx._internal.exporter._ir_builder import current_builder

    return current_builder()
