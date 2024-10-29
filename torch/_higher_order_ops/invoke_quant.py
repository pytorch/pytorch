# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import dataclasses

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch.fx.graph_module import GraphModule


class InvokeQuantHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("invoke_quant")

    def __call__(
        self,
        subgraph: GraphModule,
        *operands,
        scheme=None,
    ):
        return super().__call__(subgraph, *operands)


invoke_quant = InvokeQuantHOP()


@invoke_quant.py_impl(FakeTensorMode)
def _(mode, subgraph, *operands, schema=None):
    return subgraph(*operands)


@invoke_quant.py_impl(DispatchKey.CompositeExplicitAutograd)
def _(subgraph, *operands, schema=None):
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return subgraph(*operands)


@invoke_quant.py_impl(DispatchKey.Autograd)
def _(subgraph, *operands, schema=None):
    # invoke_quant should only be invoked post autograd
    assert not torch.is_grad_enabled() or pytree.tree_all_only(
        torch.Tensor,
        lambda t: not t.requires_grad,  # type: ignore[union-attr]
        operands,
    )
    with torch._C._AutoDispatchBelowAutograd():
        return subgraph(*operands)


@dataclasses.dataclass(frozen=True)
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

        from torch._higher_order_ops import invoke_quant_tracer

        return invoke_quant_tracer(*args, **kwargs, quant_options=self)  # type: ignore[call-arg]
