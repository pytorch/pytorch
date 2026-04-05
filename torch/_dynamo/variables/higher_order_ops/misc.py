from typing import TYPE_CHECKING

from .common import (
    _call_function_and_unflatten_output,
    Any,
    ConstDictVariable,
    graph_break_hints,
    HigherOrderOperator,
    LazyVariableTracker,
    ListVariable,
    make_attr,
    pytree,
    Sequence,
    Source,
    speculate_subgraph,
    torch,
    TorchHigherOrderOperatorVariable,
    TupleVariable,
    unimplemented,
    VariableTracker,
)


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


class CallTorchbindHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.call_torchbind"

    def __init__(
        self,
        hop: HigherOrderOperator,
        source: Source | None,
        script_obj_var: Any,
        method_name: str,
    ) -> None:
        super().__init__(hop, source)
        self.script_obj_var = script_obj_var
        self.method_name = method_name

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..builder import wrap_fx_proxy

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        args_proxy = [arg.as_proxy() for arg in args]
        kwargs_proxy = {k: v.as_proxy() for k, v in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(
                    [self.script_obj_var.as_proxy(), self.method_name] + args_proxy
                ),
                kwargs=kwargs_proxy,
            ),
        )


class PrintHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.print"

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..builder import wrap_fx_proxy

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        args_proxy = [arg.as_proxy() for arg in args]
        kwargs_proxy = {k: v.as_proxy() for k, v in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(args_proxy),
                kwargs=kwargs_proxy,
            ),
        )


class OutDtypeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.out_dtype"

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..builder import wrap_fx_proxy

        if len(kwargs) > 0:
            unimplemented(
                gb_type="out_dtype: unexpected kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"out_dtype expects no keyword arguments (got {len(kwargs)}).",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        p_args = tuple(arg.as_proxy() for arg in args)
        op = p_args[0]
        output_dtype = p_args[1]
        fake_sub_args = pytree.tree_map_only(
            torch.fx.Proxy, lambda a: a.node.meta["example_value"], p_args[2:]
        )
        # This is a simplified implementation of this operator just for tracing.
        # Actual implementation may also first promote the arguments
        example_value = op(*fake_sub_args).to(dtype=output_dtype)

        # Store the invocation as a call
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(p_args),
                kwargs={},
            ),
            example_value=example_value,
        )


class StrictModeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.strict_mode"
    _ALLOW_FALLBACK_TO_EAGER = False

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        unpacked_sequence = args[1].unpack_var_sequence(tx)
        # TODO (tmanlaibaatar) support pytree here
        for arg in unpacked_sequence:
            if isinstance(arg, (ListVariable, TupleVariable, ConstDictVariable)):
                unimplemented(
                    gb_type="strict_mode: improper args",
                    context=f"args: {args}, kwargs: {kwargs}",
                    explanation="strict_mode higher order op expects flat inputs (list/tuple/dict)",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )

        if kwargs:
            unimplemented(
                gb_type="strict_mode: unexpected kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"strict_mode higher order op expects no keyword arguments (got {len(kwargs)}).",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        assert self._HOP_NAME is not None
        (
            (ret_val, ret_spec),
            ret_graph,
            ret_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            args[0],
            unpacked_sequence,
            {},
            self._HOP_NAME,
            source_target=self.value,
            should_flatten_outputs=True,
        )

        strict_mode_nn_modules = dict(tx.output.nn_modules)

        strict_mode_name = tx.output.install_subgraph(
            "strict_mode_body",
            torch.fx.GraphModule(strict_mode_nn_modules, ret_graph),
        )

        strict_mode_node = make_attr(tx, strict_mode_name)
        p_args = (
            strict_mode_node,
            tuple(ret_lifted_freevars.keys()),
        )

        flat_example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            ret_val.as_proxy(),
        )

        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.strict_mode,
            p_args,
            {},
            flat_example_value,
            ret_spec,
            ret_val,
        )


class ExportTracepointHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order._export_tracepoint"

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..builder import wrap_fx_proxy

        p_args = tuple(arg.as_proxy() for arg in args)
        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class RunWithRNGStateHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.run_with_rng_state"

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..builder import wrap_fx_proxy

        p_args = tuple(arg.as_proxy() for arg in args)
        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class InlineAsmElementwiseHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "inline_asm_elementwise"

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..builder import wrap_fx_proxy

        p_args = tuple(arg.as_proxy() for arg in args)
        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class AutoFunctionalizeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.auto_functionalized"

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..builder import wrap_fx_proxy

        p_args = tuple(arg.as_proxy() for arg in args)
        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            example_value=None,
        )


class TraceWrappedHigherOrderOperatorVariable(TorchHigherOrderOperatorVariable):
    """
    Handles torch._dynamo._trace_wrapped_higher_order_op.inner_trace
    by unwrapping the higher order op and inlining through it.  This op
    is created by dynamo to survive through AotAutograd, then unwrapped
    here in the call to dynamo from compiled autograd.
    """

    _HOP_NAME = "torch._dynamo._trace_wrapped_higher_order_op.inner_trace"

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        kwargs = dict(kwargs)
        fn = kwargs.pop("fn")
        return fn.call_function(tx, args, kwargs)
