from inspect import getattr_static

from ..bytecode_transformation import create_call_function
from ..exc import Unsupported
from .base import VariableTracker


class SDPAParamsVariable(VariableTracker):
    """Represents the c++ params struct for scaled dot product attention"""

    def __init__(self, value, proxy, param_vars, **kwargs):
        self.proxy = proxy
        self.param_vars = param_vars
        super().__init__(**kwargs)

    def reconstruct(self, codegen):
        assert self.source is None
        assert self.param_vars is not None
        codegen.load_import_from("torch._C", "_SDPAParams")
        for var in self.param_vars:
            codegen(var)
        return create_call_function(len(self.param_vars), True)

    def as_proxy(self):
        return self.proxy

    def var_getattr(self, tx, name: str) -> VariableTracker:
        import torch._C
        from ..source import AttrSource
        from .builder import wrap_fx_proxy
        from .misc import GetAttrVariable

        try:
            getattr_static(torch._C._SDPAParams, name)
        except AttributeError:
            raise Unsupported(f"Unsupported torch._C._SDPAParams attribute {name}")

        proxy = GetAttrVariable.create_getattr_proxy(self.as_proxy(), name)
        if self.source is not None:
            return wrap_fx_proxy(
                tx=tx, proxy=proxy, source=AttrSource(self.source, name)
            )
        else:
            return wrap_fx_proxy(tx=tx, proxy=proxy)

    @staticmethod
    def is_sdpa_params(value):
        from torch.backends.cuda import SDPAParams

        return value is SDPAParams
