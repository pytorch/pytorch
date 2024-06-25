# mypy: allow-untyped-defs
import functools
from typing import Dict

import torch
from ..exc import unimplemented, UnsafeScriptObjectError, Unsupported

from .base import VariableTracker
from .user_defined import UserDefinedObjectVariable


def _raise_hard_error_if_graph_break(reason):
    def deco(fn):
        @functools.wraps(fn)
        def graph_break_as_hard_error(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Unsupported as e:
                raise UnsafeScriptObjectError(e.msg) from e

        return graph_break_as_hard_error

    return deco


class TorchScriptObjectVariable(UserDefinedObjectVariable):
    _fake_script_object_cache: Dict[int, "TorchScriptObjectVariable"] = {}

    @classmethod
    def is_matching_cls(cls, user_cls: type):
        return issubclass(user_cls, torch.ScriptObject)

    @staticmethod
    def create(proxy, value, **options):
        return TorchScriptObjectVariable(proxy, value, **options)

    def __init__(self, proxy, value, source, **kwargs):
        super().__init__(value, **kwargs)
        self.proxy = proxy
        self.proxy.node.meta["example_value"] = value
        self.source = source

    def as_proxy(self):
        return self.proxy

    @_raise_hard_error_if_graph_break(
        "Dynamo cannot safely trace script object due to graph break."
    )
    def var_getattr(self, tx, name: str) -> VariableTracker:
        from torch._higher_order_ops.torchbind import call_torchbind
        from ..source import AttrSource
        from .higher_order_ops import TorchHigherOrderOperatorVariable

        method = getattr(self.value, name, None)
        if method is None:
            unimplemented(
                f"FakeScriptObject doesn't define method {name}. Did you forget to implement it in the fake class?"
            )

        if not callable(method):
            unimplemented(
                "Only method calls on TorchScript objects can be supported safely."
                " Please use method calls instead of attribute access."
            )

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
    def call_method(self, tx, name, args, kwargs):
        unimplemented(f"call method {name} on script object is not safe.")
