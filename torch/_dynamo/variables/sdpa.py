from collections.abc import Sequence
from inspect import getattr_static
from typing import Any, TYPE_CHECKING, TypeGuard

from torch._guards import Source
from torch.backends.cuda import SDPAParams
from torch.fx.proxy import Proxy
from ..bytecode_transformation import create_call_function
from ..exc import unimplemented
from ..source import AttrSource
from .base import VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator

PARAM_NAMES = [
    "query",
    "key",
    "value",
    "attn_mask",
    "dropout",
    "is_causal",
    "enable_gqa",
]


class SDPAParamsVariable(VariableTracker):
    """Represents the c++ params struct for scaled dot product attention.
    This is a read-only container."""

    @staticmethod
    def create(
        tx: "InstructionTranslator", value: Any, source: Source
    ) -> VariableTracker:
        from .torch import TorchInGraphFunctionVariable

        params = [
            VariableTracker.build(tx, getattr(value, p), AttrSource(source, p))
            for p in PARAM_NAMES
        ]
        return TorchInGraphFunctionVariable(SDPAParams).call_function(tx, params, {})

    def __init__(
        self, proxy: Proxy, param_vars: Sequence[VariableTracker], **kwargs: Any
    ) -> None:
        self.proxy = proxy
        self.param_vars = param_vars
        super().__init__(**kwargs)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        assert self.source is None
        assert self.param_vars is not None
        codegen.add_push_null(
            lambda: codegen.load_import_from("torch._C", "_SDPAParams")
        )
        codegen.foreach(self.param_vars)
        codegen.extend_output(create_call_function(len(self.param_vars), False))

    def as_proxy(self) -> Proxy:
        return self.proxy

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        import torch._C
        from .builder import wrap_fx_proxy
        from .misc import GetAttrVariable

        try:
            getattr_static(torch._C._SDPAParams, name)
        except AttributeError:
            import torch._dynamo.graph_break_hints as graph_break_hints

            unimplemented(
                gb_type="unsupported torch._C._SDPAParams attribute",
                context=f"name: {name}",
                explanation=f"Unable to fetch attribute {name} from torch._C._SDPAParams.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        proxy = GetAttrVariable.create_getattr_proxy(self.as_proxy(), name)
        if self.source is not None:
            return wrap_fx_proxy(
                tx=tx, proxy=proxy, source=AttrSource(self.source, name)
            )
        else:
            return wrap_fx_proxy(tx=tx, proxy=proxy)

    @staticmethod
    def is_sdpa_params(value: Any) -> TypeGuard["SDPAParams"]:
        return value is SDPAParams
