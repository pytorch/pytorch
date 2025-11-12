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
from collections.abc import Callable, Iterable
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch
from torch._guards import Source
from torch.fx.proxy import Proxy

from .. import graph_break_hints
from ..exc import unimplemented, UnsafeScriptObjectError, Unsupported
from .base import VariableTracker
from .user_defined import UserDefinedObjectVariable


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


class TorchScriptObjectVariable(UserDefinedObjectVariable):
    _fake_script_object_cache: dict[int, "TorchScriptObjectVariable"] = {}

    @classmethod
    def is_matching_cls(cls, user_cls: type) -> bool:
        return issubclass(user_cls, torch.ScriptObject)

    @staticmethod
    def create(proxy: Proxy, value: Any, **options: Any) -> "TorchScriptObjectVariable":
        return TorchScriptObjectVariable(proxy, value, **options)

    def __init__(self, proxy: Proxy, value: Any, source: Source, **kwargs: Any) -> None:
        super().__init__(value, **kwargs)
        self.proxy = proxy
        self.proxy.node.meta["example_value"] = value
        self.source = source

    def as_proxy(self) -> Proxy:
        return self.proxy

    @_raise_hard_error_if_graph_break(
        "Dynamo cannot safely trace script object due to graph break."
    )
    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        from torch._higher_order_ops.torchbind import call_torchbind

        from ..source import AttrSource
        from .higher_order_ops import TorchHigherOrderOperatorVariable

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
