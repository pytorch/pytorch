# mypy: ignore-errors

from inspect import getattr_static
from typing import TYPE_CHECKING

from ..bytecode_transformation import create_call_function
from ..exc import Unsupported
from .base import VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


class SDPAParamsVariable(VariableTracker):
    """Represents the c++ params struct for scaled dot product attention.
    This is a read-only container."""

    @staticmethod
    def create(tx: "InstructionTranslator", value, source):
        from torch.backends.cuda import SDPAParams

        from ..source import AttrSource
        from .builder import VariableBuilder
        from .torch import TorchInGraphFunctionVariable

        query_var = VariableBuilder(tx, AttrSource(source, "query"))(value.query)
        key_var = VariableBuilder(tx, AttrSource(source, "key"))(value.key)
        value_var = VariableBuilder(tx, AttrSource(source, "value"))(value.value)
        attn_mask_var = VariableBuilder(tx, AttrSource(source, "attn_mask"))(
            value.attn_mask
        )
        dropout_var = VariableBuilder(tx, AttrSource(source, "dropout"))(value.dropout)
        is_causal_var = VariableBuilder(tx, AttrSource(source, "is_causal"))(
            value.is_causal
        )
        param_vars = [
            query_var,
            key_var,
            value_var,
            attn_mask_var,
            dropout_var,
            is_causal_var,
        ]
        return TorchInGraphFunctionVariable(SDPAParams).call_function(
            tx, param_vars, {}
        )

    def __init__(self, proxy, param_vars, **kwargs):
        self.proxy = proxy
        self.param_vars = param_vars
        super().__init__(**kwargs)

    def reconstruct(self, codegen):
        assert self.source is None
        assert self.param_vars is not None
        codegen.add_push_null(
            lambda: codegen.load_import_from("torch._C", "_SDPAParams")
        )
        codegen.foreach(self.param_vars)
        codegen.extend_output(create_call_function(len(self.param_vars), False))

    def as_proxy(self):
        return self.proxy

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        import torch._C

        from ..source import AttrSource
        from .builder import wrap_fx_proxy
        from .misc import GetAttrVariable

        try:
            getattr_static(torch._C._SDPAParams, name)
        except AttributeError:
            # Using raise from is too verbose here
            raise Unsupported(
                f"Unsupported torch._C._SDPAParams attribute {name}"
            ) from None

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
