# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs

import abc

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import reenter_make_fx
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class PrimHOPBase(HigherOrderOperator, abc.ABC):
    """
    This is the "Base" HOP implementation for a HOP that looks like:

        call_subgraph_hop(subgraph, operands, **kwargs)

    That is:
    1) the HOP is a "prim" (it stays alive until Inductor)
    2) the HOP's semantics are subgraph(*operands)

    To use this, please subclass this class and override methods as necessary:
    ```
    class InvokeQuant(PrimHOPBase):
        def __init__(self):
            return super().__init__("invoke_quant")

    invoke_quant = InvokeQuant()

    def g(x):
        return x.sin().cos()

    @torch.compile(backend="aot_eager")
    def f(x):
        return invoke_quant(g, (x,), scheme="nf4")
    ```

    NOTE: don't subclass PrimHOPBase out of tree! That is not allowed. All
    usages must be in tree.
    """

    def __init__(self, hop_name) -> None:
        super().__init__(hop_name)

        # Set up the registrations
        # If you want to override any of these, override them in your subclass.
        self.py_impl(DispatchKey.Autograd)(self._call_Autograd)
        self.py_functionalize_impl(self._call_Functionalize)
        self.py_impl(ProxyTorchDispatchMode)(self._call_ProxyTorchDispatchMode)
        self.py_impl(FakeTensorMode)(self._call_FakeTensorMode)
        self.py_impl(DispatchKey.CompositeExplicitAutograd)(
            self._call_CompositeExplicitAutograd
        )

    def __call__(self, subgraph, operands, *unused, **kwargs):
        # We accept *unused (and *_) to make mypy happy. Otherwise mypy
        # complains that we're violating LSP. We are violating LSP, but it's
        # OK for the purposes of implementation-sharing (end users should never
        # subclass these methods; only in-tree PyTorch developers are allowed to).
        assert len(unused) == 0
        if not isinstance(subgraph, (torch.fx.GraphModule, FunctionWithNoFreeVars)):
            raise RuntimeError(
                f"{self._name}: when calling this API without torch.compile, "
                f"we require that the subgraph be a torch.fx.GraphModule (or "
                f"a function we know doesn't have free variables)."
            )
        return super().__call__(subgraph, operands, **kwargs)

    def _call_Autograd(self, subgraph, operands, *_, **kwargs):
        if isinstance(subgraph, torch.fx.GraphModule):
            pass
        if not torch.is_grad_enabled() or pytree.tree_all_only(
            torch.Tensor,
            lambda t: not t.requires_grad,
            operands,
        ):
            with torch._C._AutoDispatchBelowAutograd():
                return self(subgraph, operands, **kwargs)

        # We assume the subgraph doesn't mutate inputs and there is no aliasing.
        # In the PT2 stack, this is Dynamo's responsibility to figure out.
        return PrimHOPBaseFunction.apply(self, subgraph, kwargs, *operands)

    def _call_CompositeExplicitAutograd(self, subgraph, operands, *_, **kwargs):
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        mode = _get_current_dispatch_mode()
        assert mode is None, "Mode should never be enabled for CPU/CUDA key"
        return subgraph(*operands)

    def _call_ProxyTorchDispatchMode(
        self, proxy_mode, subgraph, operands, *_, **kwargs
    ):
        traced_graph = reenter_make_fx(subgraph)(*operands)
        assert isinstance(proxy_mode.tracer, torch.fx.Tracer)
        qualname = proxy_mode.tracer.get_fresh_qualname("subgraph")
        proxy_mode.tracer.root.register_module(qualname, traced_graph)

        node_args = (traced_graph, operands)
        proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)  # type: ignore[attr-defined]
        proxy_kwargs = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, kwargs)  # type: ignore[attr-defined]
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", self, proxy_args, proxy_kwargs
        )

        out = self(subgraph, operands, **kwargs)
        return track_tensor_tree(
            out, out_proxy, constant=None, tracer=proxy_mode.tracer  # type: ignore[arg-type]
        )

    def _call_FakeTensorMode(self, mode, subgraph, operands, *_, **kwargs):
        # TODO: this should probably route through FakeTensorMode to reuse caching
        with mode:
            return subgraph(*operands)

    def _call_Functionalize(self, ctx, subgraph, operands, *_, **kwargs):
        unwrapped_operands = ctx.unwrap_tensors(operands)
        with ctx.redispatch_to_next():
            # We assume the subgraph doesn't mutate inputs and there is no aliasing.
            # In the PT2 stack, this is Dynamo's responsibility to figure out.
            functionalized_subgraph = FunctionWithNoFreeVars(
                ctx.functionalize(subgraph)
            )
            out = self(functionalized_subgraph, unwrapped_operands, **kwargs)
        return ctx.wrap_tensors(out)


class PrimHOPBaseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hop, subgraph, kwargs, *operands):
        ctx.hop = hop
        ctx.operands = operands
        ctx.subgraph = subgraph
        ctx.kwargs = kwargs

        with torch._C._AutoDispatchBelowAutograd():
            return hop(subgraph, operands, **kwargs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        subgraph = ctx.subgraph
        operands = ctx.operands
        kwargs = ctx.kwargs

        # TODO: Something special needs to happen with min cut partitioner
        with suspend_functionalization(), disable_functional_mode(), torch.enable_grad():
            with disable_proxy_modes_tracing():
                from .invoke_subgraph import create_fw_bw_graph
                from .utils import _from_fun

                fw_inputs = pytree.tree_map(_from_fun, operands)
                _, joint_graph, _ = create_fw_bw_graph(
                    subgraph, fw_inputs, grad_outputs
                )

        # The joint graph returns (*grad_inputs, *fwd_outputs).
        # We only need the grad_inputs.
        def bwd_fn(*args):
            operands = args[: -len(grad_outputs)]
            grad_outs = args[-len(grad_outputs) :]
            result = joint_graph(*operands, *grad_outs)
            grad_inputs = result[: -len(grad_outputs)]
            return grad_inputs

        return (
            None,
            None,
            None,
            *ctx.hop(
                FunctionWithNoFreeVars(bwd_fn), (*operands, *grad_outputs), **kwargs
            ),
        )


class FunctionWithNoFreeVars:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
