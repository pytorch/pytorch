# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
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
