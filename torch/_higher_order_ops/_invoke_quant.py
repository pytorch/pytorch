from torch._higher_order_ops.prim_hop_base import FunctionWithNoFreeVars, PrimHOPBase

import dataclasses

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._subclasses import FakeTensorMode
from torch.fx.graph_module import GraphModule


class InvokeQuantTracer(PrimHOPBase):
    def __init__(self):
        super().__init__("invoke_quant_packed")

    def __call__(self, subgraph, operands, *, scheme=None, quant_options=None):
        return super().__call__(
            subgraph, operands, scheme=scheme, quant_options=quant_options
        )

invoke_quant_packed = InvokeQuantTracer()


class InvokeQuantUnpacked(PrimHOPBase):
    def __init__(self):
        super().__init__("invoke_quant")

    def __call__(self, subgraph, *operands, scheme=None):
        return super().__call__(
            subgraph, operands, scheme=scheme
        )

    def _call_FakeTensorMode(self, mode, subgraph, operands, **kwargs):
        # TODO: this should probably route through FakeTensorMode to reuse caching
        with mode:
            return subgraph(*operands[0])


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

        force_fuse_mm: Force fusion to Triton matrix multiplications even without
            max-autotune enabled.
    """

    codegen_low_precision: bool = True
    force_fuse_mm: bool = False

    def __call__(
        self,
        *args,
        **kwargs,
    ):
        if not torch._utils.is_compiling():
            return args[0](*args[1])


        return invoke_quant_packed(*args, **kwargs, quant_options=self)  # type: ignore[call-arg]
