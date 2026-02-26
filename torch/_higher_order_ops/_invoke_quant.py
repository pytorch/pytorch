# mypy: allow-untyped-defs
# need to fix prim_hop_base type annotations first

import dataclasses

import torch
from torch._higher_order_ops.base_hop import BaseHOP, FunctionWithNoFreeVars


class InvokeQuantTracer(BaseHOP):
    def __init__(self) -> None:
        super().__init__("invoke_quant_packed")

    def __call__(self, subgraph, *operands, scheme=None, quant_options=None):
        subgraph = FunctionWithNoFreeVars(subgraph)
        return super().__call__(
            subgraph, *operands, scheme=scheme, quant_options=quant_options
        )


invoke_quant_packed = InvokeQuantTracer()


class InvokeQuantUnpacked(BaseHOP):
    def __init__(self) -> None:
        super().__init__("invoke_quant")


invoke_quant = InvokeQuantUnpacked()


@dataclasses.dataclass(frozen=True, repr=True)
class InvokeQuant:
    """
    Invoke a quantization function that will be preserved as a single operator. Preservation
    as a single operator aids in pattern matching and custom lowerings.

    The operation appears as:
        torch.ops.higher_order.invoke_quant(subgraph, *args, scheme=scheme)

    Args:
        codegen_low_precision: Use observed subgraph dtypes for codegen instead of
            upcasting to fp32. Can improve performance for prologue fusion but
            requires careful testing of numerics.
    """

    codegen_low_precision: bool = True

    def __call__(
        self,
        *args,
        scheme: str | None = None,
        **kwargs,
    ):
        if not torch.compiler.is_compiling():
            return args[0](*args[1:], **kwargs)

        if scheme is not None:
            kwargs["scheme"] = scheme

        return invoke_quant_packed(*args, **kwargs, quant_options=self)  # type: ignore[call-arg]
