from typing import Dict

import torch

from .base import VariableTracker
from .user_defined import UserDefinedObjectVariable


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

    def var_getattr(self, tx, name: str) -> VariableTracker:
        from torch._higher_order_ops.torchbind import call_torchbind
        from ..source import AttrSource
        from .higher_order_ops import TorchHigherOrderOperatorVariable

        assert callable(getattr(self.value, name))
        return TorchHigherOrderOperatorVariable.make(
            call_torchbind,
            source=AttrSource(self.source, name),
            script_obj_var=self,
            method_name=name,
        )
