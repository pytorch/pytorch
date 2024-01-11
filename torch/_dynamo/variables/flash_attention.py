from ..bytecode_transformation import create_call_function
from .base import VariableTracker


class SDPAParamsVariable(VariableTracker):
    """Represents the c++ params struct for flash attention"""

    def __init__(self, proxy, param_vars, **kwargs):
        self.proxy = proxy
        self.param_vars = param_vars
        super().__init__(**kwargs)

    def reconstruct(self, codegen):
        codegen.load_import_from("torch._C", "_SDPAParams")
        for var in self.param_vars:
            codegen(var)
        return create_call_function(len(self.param_vars), True)

    def as_proxy(self):
        return self.proxy

    @staticmethod
    def is_sdpa_params(value):
        from torch.backends.cuda import SDPAParams

        return value is SDPAParams
