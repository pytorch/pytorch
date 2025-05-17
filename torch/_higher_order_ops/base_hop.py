# mypy: allow-untyped-defs

import abc

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    check_input_alias_and_mutation_return_ouputs,
    reenter_make_fx,
)
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class BaseHOP(HigherOrderOperator, abc.ABC):
    """
    This is the "Base" HOP implementation for a HOP that looks like:

        call_subgraph_hop(subgraph, *operands, **kwargs)

    That is:
    1) the HOP stays alive until Inductor
    2) the HOP's semantics are subgraph(*operands)
    3) kwargs may be some config options but aren't passed directly to the subgraph.

    To use this, please subclass this class and override methods as necessary:
    ```
    class InvokeQuant(BaseHOP):
        def __init__(self):
            return super().__init__("invoke_quant")

    invoke_quant = InvokeQuant()

    def g(x):
        return x.sin().cos()

    @torch.compile(backend="aot_eager")
    def f(x):
        return invoke_quant(g, x, scheme="nf4")
    ```

    NOTE: don't subclass BaseHOP out of tree! That is not allowed. All
    usages must be in tree.
    """

    def __init__(self, hop_name) -> None:
        super().__init__(hop_name)

        # Set up the registrations
        # If you want to override any of these, override them in your subclass.
        self.py_autograd_impl(self._call_Autograd)
        self.py_functionalize_impl(self._call_Functionalize)
        self.py_impl(ProxyTorchDispatchMode)(self._call_ProxyTorchDispatchMode)
        self.py_impl(FakeTensorMode)(self._call_FakeTensorMode)
        self.py_impl(DispatchKey.CompositeExplicitAutograd)(
            self._call_CompositeExplicitAutograd
        )

    def __call__(self, subgraph, *operands, **kwargs):
        if not isinstance(subgraph, (torch.fx.GraphModule, FunctionWithNoFreeVars)):
            raise RuntimeError(
                f"{self._name}: when calling this API without torch.compile, "
                f"we require that the subgraph be a torch.fx.GraphModule (or "
                f"a function we know doesn't have free variables)."
            )
        return super().__call__(subgraph, *operands, **kwargs)

    def _call_Autograd(self, subgraph, *operands, **kwargs):
        if isinstance(subgraph, torch.fx.GraphModule):
            pass

        # We assume the subgraph doesn't mutate inputs and there is no aliasing.
        # In the PT2 stack, this is Dynamo's responsibility to figure out.
        return BaseHOPFunction.apply(self, subgraph, kwargs, *operands)

    def _call_CompositeExplicitAutograd(self, subgraph, *operands, **kwargs):
        from torch.utils._python_dispatch import _get_current_dispatch_mode

        mode = _get_current_dispatch_mode()
        assert mode is None, "Mode should never be enabled for CPU/CUDA key"
        return subgraph(*operands)

    def _call_ProxyTorchDispatchMode(self, proxy_mode, subgraph, *operands, **kwargs):
        traced_graph = reenter_make_fx(subgraph)(*operands)
        assert isinstance(proxy_mode.tracer, torch.fx.Tracer)
        qualname = proxy_mode.tracer.get_fresh_qualname("subgraph")
        proxy_mode.tracer.root.register_module(qualname, traced_graph)

        node_args = (traced_graph, *operands)
        proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)  # type: ignore[attr-defined]
        proxy_kwargs = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, kwargs)  # type: ignore[attr-defined]
        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", self, proxy_args, proxy_kwargs
        )

        out = self(subgraph, *operands, **kwargs)
        return track_tensor_tree(
            out, out_proxy, constant=None, tracer=proxy_mode.tracer  # type: ignore[arg-type]
        )

    def _call_FakeTensorMode(self, mode, subgraph, *operands, **kwargs):
        # TODO: this should probably route through FakeTensorMode to reuse caching
        with mode:
            return subgraph(*operands)

    def _call_Functionalize(self, ctx, subgraph, *operands, **kwargs):
        unwrapped_operands = ctx.unwrap_tensors(operands)
        with ctx.redispatch_to_next():
            # We assume the subgraph doesn't mutate inputs and there is no aliasing.
            # In the PT2 stack, this is Dynamo's responsibility to figure out.
            functionalized_subgraph = FunctionWithNoFreeVars(
                ctx.functionalize(subgraph)
            )
            out = self(functionalized_subgraph, *unwrapped_operands, **kwargs)
        return ctx.wrap_tensors(out)

    def gen_schema(self, *args, **kwargs):
        from .schema import CFunctionSchemaGen, HopArgumentInfoGen

        subgraph, *operands = args

        assert isinstance(
            subgraph, torch.fx.GraphModule
        ), f"NYI non GraphModule subgraph got {subgraph}"

        fake_args = [
            ph.meta["example_value"]
            for ph in subgraph.graph.find_nodes(op="placeholder")
        ]
        (
            inp_inp_alias,
            inp_out_alias,
            out_out_alias,
            mutated_inp_idx,
            output,
        ) = check_input_alias_and_mutation_return_ouputs(subgraph, fake_args)

        assert (
            len(inp_inp_alias) == 0
            and len(inp_out_alias) == 0
            and len(out_out_alias) == 0
        ), "Aliasing is not suppported for HOP subgraph."
        args = [
            HopArgumentInfoGen.from_example(
                subgraph, name="subgraph", default_value=None, is_mutated=False
            )
        ]
        for idx, arg in enumerate((*operands, *kwargs.items())):
            if isinstance(arg, tuple):
                # kwargs value are treated as default argument
                arg_name, example_value = arg
                default = example_value
                kw_only = True
            else:
                arg_name = f"arg{idx}"
                example_value = arg
                default = None
                kw_only = False
            args.append(
                HopArgumentInfoGen.from_example(
                    example_value=example_value,
                    name=arg_name,
                    default_value=default,
                    is_mutated=idx in mutated_inp_idx,
                    kw_only=kw_only,
                )
            )

        # The output is represented as a single argument
        out = HopArgumentInfoGen.from_example(
            example_value=output,
            name="out",
            default_value=None,
            is_mutated=False,
        )
        return CFunctionSchemaGen.from_hop_argument_info(str(self), args, out)


class BaseHOPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hop, subgraph, kwargs, *operands):
        ctx.hop = hop
        ctx.operands = operands
        ctx.subgraph = subgraph
        ctx.kwargs = kwargs

        with torch._C._AutoDispatchBelowAutograd():
            return hop(subgraph, *operands, **kwargs)

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
                (
                    _,
                    joint_graph,
                    _,
                ) = create_fw_bw_graph(subgraph, fw_inputs, grad_outputs)

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
                FunctionWithNoFreeVars(bwd_fn), *operands, *grad_outputs, **kwargs
            ),
        )


class FunctionWithNoFreeVars:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
