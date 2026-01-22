"""
This module implements variable tracking for TorchScript objects during Dynamo tracing.

The TorchScriptObjectVariable class provides specialized handling for TorchScript
objects with strong safety guarantees by:
- Enforcing method-call-only access to prevent unsafe attribute manipulation
- Converting graph breaks into hard errors via _raise_hard_error_if_graph_break
- Proper proxy and source tracking for TorchScript method calls
- Integration with higher-order operators for method call handling

Key safety features:
- Strict validation that only method calls are allowed (no direct attribute access)
- Immediate error reporting for potentially unsafe operations
- Proper source tracking for debugging and guard installation
- Safe handling of TorchScript object method calls through torchbind

The module ensures that TorchScript objects are handled safely during tracing
by limiting operations to known-safe patterns and failing fast for unsafe usage.
"""

import functools
import inspect
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.utils._pytree as pytree
from torch._guards import Source
from torch._library.opaque_object import (
    get_member_type,
    is_opaque_reference_type,
    is_opaque_type,
    is_opaque_value_type,
    MemberType,
)
from torch.fx.proxy import Proxy

from .. import graph_break_hints
from ..eval_frame import skip_code
from ..exc import unimplemented, UnsafeScriptObjectError, Unsupported
from ..source import AttrSource
from ..utils import proxy_args_kwargs
from .base import VariableTracker
from .constant import ConstantVariable
from .dicts import ConstDictVariable
from .lists import TupleVariable
from .misc import LambdaVariable
from .user_defined import UserDefinedObjectVariable, UserDefinedVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

_P = ParamSpec("_P")
_T = TypeVar("_T")


def _raise_hard_error_if_graph_break(
    reason: str,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def deco(fn: Callable[_P, _T]) -> Callable[_P, _T]:
        @functools.wraps(fn)
        def graph_break_as_hard_error(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            try:
                return fn(*args, **kwargs)
            except Unsupported as e:
                raise UnsafeScriptObjectError(e.msg) from e

        return graph_break_as_hard_error

    return deco


class OpaqueObjectClassVariable(UserDefinedVariable):
    """
    A variable that represents an opaque object class (not instance).
    Since UserDefinedClassVariable has some special handling for side effects,
    we have a separate class here which will directly return the object when
    __init__ is called.
    """

    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def as_python_constant(self) -> Any:
        return self.value

    def is_python_constant(self) -> bool:
        # prevents constant folding of attribute accesses on
        # opaque classes. this ensures var_getattr is called,
        # allowing for proper validation and error handling
        return False

    def is_python_hashable(self) -> bool:
        return is_opaque_value_type(self.value)

    def as_proxy(self) -> Any:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        obj = None
        try:
            obj = inspect.getattr_static(self.value, name)
        except AttributeError:
            unimplemented(
                gb_type="Attribute not found on opaque class",
                context=f"class={self.value}, attr={name}",
                explanation=f"The attribute '{name}' does not exist on opaque class {self.value}.",
                hints=[
                    f"Ensure '{name}' is a valid attribute of {type(self.value)}.",
                ],
            )

        if isinstance(obj, staticmethod):
            obj = obj.__get__(self.value)
        elif isinstance(obj, property):
            obj = obj.__get__(None, self.value)  # pyrefly: ignore[no-matching-overload]
        elif hasattr(obj, "__get__"):
            # Check for pybind11 static properties (common in PyTorch C++ bindings)
            # Reference: https://github.com/python/mypy/blob/131f9d92da58294bb2f273425e8778bd7d5b861f/mypy/stubgenc.py#L590
            type_name = type(obj).__name__
            if type_name == "pybind11_static_property":
                obj = obj.__get__(None, self.value)
            else:
                unimplemented(
                    gb_type="Unsupported descriptor on opaque class",
                    context=f"class={self.value}, attr={name}, descriptor={type_name}",
                    explanation=f"The attribute '{name}' is a descriptor of type '{type_name}' which is not supported.",
                    hints=[
                        "Only staticmethod, property, and pybind11_static_property are supported.",
                        "Consider accessing this attribute outside of the compiled region.",
                    ],
                )

        if ConstantVariable.is_literal(obj):
            return ConstantVariable.create(obj)

        source = AttrSource(self.source, name) if self.source else None
        return VariableTracker.build(tx, obj, source)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # disallow creating reference-type opaque objects in the middle of the
        # program
        if is_opaque_reference_type(self.value):
            # Skip __init__ to prevent dynamo from tracing it during resume
            skip_code(self.value.__init__.__code__)

            unimplemented(
                gb_type="An opaque object was created in the middle of the program.",
                context=f"Opaque object type: {self.value}.",
                explanation=(
                    "Opaque objects cannot be created inside the torch.compile region. "
                    "They must be created before entering the compiled function."
                ),
                hints=[
                    "Please create the opaque object before calling torch.compile "
                    "and pass it in as an argument or as a global variable."
                ],
            )

        var_args = TupleVariable(list(args))
        var_kwargs = ConstDictVariable(
            {ConstantVariable(k): v for k, v in kwargs.items()}
        )
        opaque_obj = self.value(  # pyrefly: ignore[not-callable]
            *(var_args.as_python_constant()),
            **(var_kwargs.as_python_constant()),
        )
        fake_script_obj = torch._library.fake_class_registry.maybe_to_fake_obj(
            tx.output.fake_mode, opaque_obj
        )

        return TorchScriptObjectVariable.create(opaque_obj, fake_script_obj)


class TorchScriptObjectVariable(UserDefinedObjectVariable):
    _fake_script_object_cache: dict[int, "TorchScriptObjectVariable"] = {}

    @classmethod
    def is_matching_cls(cls, user_cls: type) -> bool:
        return issubclass(user_cls, torch.ScriptObject) or is_opaque_type(user_cls)

    @staticmethod
    def create(proxy: Proxy, value: Any, **options: Any) -> "TorchScriptObjectVariable":
        return TorchScriptObjectVariable(proxy, value, **options)

    def __init__(
        self, proxy: Proxy, value: Any, source: Optional[Source] = None, **kwargs: Any
    ) -> None:
        super().__init__(value, **kwargs)
        self.proxy = proxy
        if isinstance(self.proxy, torch.fx.Proxy):
            self.proxy.node.meta["example_value"] = value
        self.source = source

    def as_proxy(self) -> Proxy:
        return self.proxy

    @_raise_hard_error_if_graph_break(
        "Dynamo cannot safely trace script object due to graph break."
    )
    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        from torch._higher_order_ops.torchbind import call_torchbind

        from .higher_order_ops import TorchHigherOrderOperatorVariable

        if hasattr(self.value, "script_class_name") and is_opaque_type(
            self.value.script_class_name
        ):
            real_obj = self.value.real_obj  # pyrefly: ignore[missing-attribute]

            member_type = get_member_type(
                type(real_obj),
                name,
            )
            if member_type is None:
                unimplemented(
                    gb_type="Attempted to access unregistered member on an OpaqueObject",
                    context=f"value={real_obj}, attr={name}",
                    explanation=f"Member '{name}' is not registered for this opaque object type.",
                    hints=[
                        f"Register '{name}' with a MemberType in register_opaque_type(members=...).",
                    ],
                )

            if member_type == MemberType.USE_REAL:
                value = getattr(real_obj, name)
                if inspect.ismethod(value):
                    return LambdaVariable(
                        lambda *args, **kwargs: self.call_method(tx, name, args, kwargs)
                    )
                else:
                    return super().var_getattr(tx, name)

            elif member_type == MemberType.INLINED:
                value = getattr(real_obj, name)
                if inspect.ismethod(value) and self.source is None:
                    # When we don't have a source, fall back to call_method
                    # which creates a proxy node.
                    return LambdaVariable(
                        lambda *args, **kwargs: self.call_method(tx, name, args, kwargs)
                    )
                return super().var_getattr(tx, name)

        method = getattr(self.value, name, None)
        if method is None:
            unimplemented(
                gb_type="FakeScriptObject missing method implementation",
                context=f"value={self.value}, method={name}",
                explanation=f"TorchScript object {self.value} doesn't define the method {name}.",
                hints=[
                    f"Ensure the method {name} is implemented in {self.value}.",
                    *graph_break_hints.USER_ERROR,
                ],
            )

        if not callable(method):
            unimplemented(
                gb_type="Attempted to access non-callable attribute of TorchScript object",
                context=f"value={self.value}, method={name}",
                explanation="Attribute accesses of TorchScript objects to non-callable attributes are not supported.",
                hints=[
                    "Use method calls instead of attribute access.",
                ],
            )

        assert self.source is not None
        return TorchHigherOrderOperatorVariable.make(
            call_torchbind,
            source=AttrSource(self.source, name),
            script_obj_var=self,
            method_name=name,
        )

    # We only support method calls on script objects. Interpreting the bytecodes
    # should go through var_getattr then call_function instead of call_method.
    #
    # However, it's possible for call_method to be used directly e.g. for __setattr__.
    @_raise_hard_error_if_graph_break(
        "Dynamo cannot safely trace script object due to graph break."
    )
    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: Iterable[Any],
        kwargs: dict[str, Any],
    ) -> VariableTracker:
        from .builder import wrap_fx_proxy

        if hasattr(self.value, "script_class_name") and is_opaque_type(
            self.value.script_class_name
        ):
            real_obj = self.value.real_obj  # pyrefly: ignore[missing-attribute]
            value_type = type(real_obj)

            member_type = get_member_type(
                value_type,
                name,
            )
            if member_type is None:
                unimplemented(
                    gb_type="Attempted to access unregistered member on an OpaqueObject",
                    context=f"value={real_obj}, attr={name}",
                    explanation=f"Member '{name}' is not registered for this opaque object type.",
                    hints=[
                        f"Register '{name}' with a MemberType in register_opaque_type(members=...).",
                    ],
                )

            if member_type == MemberType.INLINED:
                proxy_args, proxy_kwargs = proxy_args_kwargs(args, kwargs)

                proxy = tx.output.create_proxy(
                    "call_method",
                    name,
                    args=(self.proxy, *proxy_args),
                    kwargs=proxy_kwargs,
                )

                return wrap_fx_proxy(tx=tx, proxy=proxy)

            elif member_type == MemberType.USE_REAL:
                if inspect.getattr_static(value_type, "__getattr__", None) is not None:
                    unimplemented(
                        gb_type="Opaque object with custom __getattr__ not supported",
                        context=f"{value_type.__name__} with custom __getattr__",
                        explanation="Dynamo does not support opaque objects types with custom __getattr__ methods",
                        hints=[],
                    )

                args_const = [x.as_python_constant() for x in args]
                kwargs_const = {k: v.as_python_constant() for k, v in kwargs.items()}

                method = getattr(real_obj, name)

                if name == "__setattr__":
                    method(*args_const, **kwargs_const)
                    return real_obj  # pyrefly: ignore[bad-return]

                constant_val = method(*args_const, **kwargs_const)

                if any(
                    is_opaque_reference_type(type(r))
                    for r in pytree.tree_leaves(constant_val)
                ):
                    unimplemented(
                        gb_type="Opaque object member with method-type USE_REAL returned a reference-type opaque object.",
                        context=f"Opaque object type: {value_type}. Method name: '{name}'",
                        explanation=(
                            "To properly guard reference-type opaque objects, "
                            "we must lift them as inputs to the graph. In order "
                            "to do this, they must all have a source, meaning they "
                            "come from a global value or are an attribute of an input."
                        ),
                        hints=[
                            f"Register member '{name}' with MemberType.INLINED in "
                            "register_opaque_type({value_type}, members=...).",
                        ],
                    )

                return VariableTracker.build(tx, constant_val)

            else:
                unimplemented(
                    gb_type="Unsupported member type on OpaqueObject",
                    context=f"value={real_obj}, attr={name}, member_type={member_type}",
                    explanation=f"Member type '{member_type}' is not supported for this operation.",
                    hints=[],
                )

        unimplemented(
            gb_type="Weird method call on TorchScript object",
            context=f"value={self.value}, method={name}",
            explanation=(
                f"This particular method call ({name}) is not supported (e.g. calling `__setattr__`). "
                "Most method calls to TorchScript objects should be supported."
            ),
            hints=[
                "Avoid calling this method.",
            ],
        )

    def as_python_constant(self) -> Any:
        if is_opaque_value_type(
            type(self.value.real_obj)  # pyrefly: ignore[missing-attribute]
        ):
            return self.value.real_obj  # pyrefly: ignore[missing-attribute]
        return super().as_python_constant()

    def is_python_hashable(self) -> bool:
        return is_opaque_value_type(
            type(self.value.real_obj)  # pyrefly: ignore[missing-attribute]
        )
