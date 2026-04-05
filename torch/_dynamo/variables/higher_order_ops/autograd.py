from typing import TYPE_CHECKING

from .common import (
    _call_function_and_unflatten_output,
    _call_function_with_auto_output_flattening,
    _check_supported_callable_arg,
    _get_fake_value,
    _make_inlined,
    add_call_function,
    add_hop_context,
    Any,
    AttrSource,
    CONSTANT_VARIABLE_NONE,
    ConstDictVariable,
    copy,
    discard_graph_changes,
    enable_python_dispatcher,
    graph_break_hints,
    GraphModule,
    HigherOrderOperator,
    itertools,
    LazyVariableTracker,
    ListVariable,
    make_attr,
    OrderedSet,
    overwrite_tensor_vt_proxy,
    overwrite_tensor_vt_requires_grad,
    Proxy,
    proxy_args_kwargs,
    pytree,
    RepararametrizeModuleContextVariable,
    Sequence,
    set_example_value,
    Source,
    SubgraphTracingInfo,
    torch,
    TorchHigherOrderOperatorVariable,
    TupleVariable,
    types,
    unimplemented,
    UserFunctionVariable,
    variables,
    VariableTracker,
)


if TYPE_CHECKING:
    from torch._dynamo.output_graph import SubgraphTracer
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from .. import AutogradFunctionContextVariable


def _higher_order_ops():
    # Preserve the old monolithic module lookup so package-level monkeypatches
    # keep affecting speculative tracing after the file split.
    return torch._dynamo.variables.higher_order_ops


class CustomFunctionHigherOrderOperatorVariable(TorchHigherOrderOperatorVariable):
    """
    Wraps torch._functorch.autograd_function.custom_function_call
    """

    _HOP_NAME = "torch.ops.higher_order.custom_function_call"

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        assert self.source is not None
        return torch._dynamo.variables.UserMethodVariable(
            self.value.__call__.__func__,
            torch._dynamo.variables.UserDefinedObjectVariable(
                self.value, source=self.source
            ),
            source=AttrSource(self.source, "__call__"),
        ).call_function(tx, args, kwargs)


class FunctorchHigherOrderVariable(UserFunctionVariable):
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return super().call_function(tx, args, kwargs)

    def should_allow_nested_graph_breaks(self) -> bool:
        return False


class FunctionalCallVariable(FunctorchHigherOrderVariable):
    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return super().call_function(tx, args, kwargs)


class ReparametrizeModuleCallVariable(FunctorchHigherOrderVariable):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        ctx_manager_vt = super().call_function(tx, args, kwargs)
        return RepararametrizeModuleContextVariable(ctx_manager_vt, args[0])  # type: ignore[arg-type]


class WrapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.wrap"
    supports_input_mutation = True
    supports_aliasing = True
    allow_side_effects = False

    def install_subgraph_in_output_graph(
        self,
        tx: "InstructionTranslator",
        fn_vt: VariableTracker,
        fn_args_vt: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
        body_gmod: GraphModule,
        attr_name: str = "wrap_body",
    ) -> str:
        return tx.output.install_subgraph(
            f"{attr_name}",
            body_gmod,
        )

    def create_wrapped_node(
        self,
        tx: "InstructionTranslator",
        fn_vt: VariableTracker,
        fn_args_vt: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
        description: str,
        *,
        subgraph_name: str = "wrap_body",
    ) -> tuple[
        tuple[Proxy, ...],
        dict[str, VariableTracker],
        Any,
        VariableTracker,
        GraphModule,
        str,
        VariableTracker | tuple[VariableTracker, ...],
        SubgraphTracingInfo,
    ]:
        # See NOTE [HigherOrderOperator tracing design] for more details
        (
            body_r,
            body_graph,
            body_lifted_freevars,
            body_graph_output_vts,
            tracing_info,
        ) = _higher_order_ops().speculate_subgraph_with_auto_output_flattening(
            tx,
            fn_vt,
            fn_args_vt,
            kwargs,
            description,
            source_target=self.value,
            allow_side_effects=self.allow_side_effects,
            filter_aliased_intermediates=getattr(
                self, "filter_aliased_intermediates", False
            ),
            supports_input_mutation=self.supports_input_mutation,
            supports_aliasing=self.supports_aliasing,
        )

        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = self.install_subgraph_in_output_graph(
            tx,
            fn_vt,
            fn_args_vt,
            kwargs,
            body_gmod,
            attr_name=subgraph_name,
        )
        body_node = make_attr(tx, body_name)

        # Since, we call `speculate_subgraph` with `set_subgraph_inputs="automatic`,
        # all the arguments are lifted.
        lifted_args = tuple(arg for arg in body_lifted_freevars)

        proxy_args = (body_node,) + lifted_args

        example_value = pytree.tree_map_only(
            torch.fx.Node,
            lambda a: a.meta["example_value"],
            body_graph.find_nodes(op="output")[0].args[0],
        )

        return (
            proxy_args,
            {},
            example_value,
            body_r,
            body_gmod,
            body_name,
            body_graph_output_vts,
            tracing_info,
        )

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # This flattens the kwargs into lifted args
        (
            p_args,
            p_kwargs,
            _example_value,
            body_r,
            _,
            _,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(tx, args[0], args[1:], kwargs, "wrap")

        if len(p_kwargs) > 0:
            unimplemented(
                gb_type="WrapHigherOrderVariable: kwargs unexpected",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation="kwargs should have been flattened into lifted args.",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        return _call_function_with_auto_output_flattening(  # type: ignore[return-value]
            tx,
            self.value,
            tuple(p_args),
            p_kwargs,
            _example_value,
            body_r,
            body_graph_output_vts,
        )


class WrapWithSetGradEnabledHigherOrderVariable(TorchHigherOrderOperatorVariable):
    """
    This hop is not exposed to users but is inserted into the graph
    after export as a post-processing step.
    """

    _HOP_NAME = "torch.ops.higher_order.wrap_with_set_grad_enabled"

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        if kwargs:
            unimplemented(
                gb_type="wrap_with_set_grad_enabled: unexpected kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"wrap_with_set_grad_enabled expects no keyword arguments (got {len(kwargs)}).",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        grad_enabled, fn_var, *rest_args = args

        if not grad_enabled.is_python_constant():
            unimplemented(
                gb_type="wrap_with_set_grad_enabled: non-constant grad_enabled",
                context=str(grad_enabled),
                explanation="wrap_with_set_grad_enabled expects grad_enabled argument to be a constant.",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        _check_supported_callable_arg(tx, fn_var, "enable_grad_fn")

        with torch.set_grad_enabled(grad_enabled.as_python_constant()):
            assert self._HOP_NAME is not None
            (
                (body_r, treespec),
                body_graph,
                body_lifted_freevars,
            ) = _higher_order_ops().speculate_subgraph(
                tx,
                fn_var,
                [*rest_args],
                {},
                self._HOP_NAME,
                source_target=self.value,
                set_subgraph_inputs="manual",
                should_flatten_outputs=True,
            )

        if len(body_lifted_freevars) > 0:
            unimplemented(
                gb_type="wrap_with_set_grad_enabled: unexpected freevars",
                context=str(body_lifted_freevars),
                explanation="wrap_with_set_grad_enabled expects no freevars.",
                hints=[],
            )

        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = tx.output.install_subgraph(
            "wrap_body",
            body_gmod,
        )

        body_node = make_attr(tx, body_name)

        proxy_args = tuple(
            [
                grad_enabled.as_python_constant(),
                body_node,
            ]
            + [operand.as_proxy() for operand in rest_args]
        )
        example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )
        return _call_function_and_unflatten_output(
            tx, self.value, proxy_args, {}, example_value, treespec, body_r
        )


class WrapWithAutocastHigherOrderVariable(TorchHigherOrderOperatorVariable):
    """
    This hop is not exposed to users but is inserted into the graph
    after export as a post-processing step.
    """

    _HOP_NAME = "torch.ops.higher_order.wrap_with_autocast"

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        if kwargs:
            unimplemented(
                gb_type="wrap_with_autocast: unexpected kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"wrap_with_autocast expects no keyword arguments (got {len(kwargs)}).",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        device_type, dtype, enabled, cache_enabled, fn_var, *rest_args = args

        for arg in [device_type, dtype, enabled, cache_enabled]:
            if not arg.is_python_constant():
                unimplemented(
                    gb_type="wrap_with_autocast: expected constant arg",
                    context=str(args),
                    explanation="wrap_with_autocast expects device_type, dtype, enabled, "
                    "and cache_enabled arguments to be constants.",
                    hints=[
                        *graph_break_hints.DYNAMO_BUG,
                    ],
                )

        _check_supported_callable_arg(tx, fn_var, "autocast")

        python_constants = [
            arg.as_python_constant()
            for arg in [device_type, dtype, enabled, cache_enabled]
        ]

        with torch.autocast(*python_constants):
            assert self._HOP_NAME is not None
            (
                (body_r, treespec),
                body_graph,
                body_lifted_freevars,
            ) = _higher_order_ops().speculate_subgraph(
                tx,
                fn_var,
                [*rest_args],
                {},
                self._HOP_NAME,
                source_target=self.value,
                set_subgraph_inputs="manual",
                should_flatten_outputs=True,
            )

        if len(body_lifted_freevars) > 0:
            unimplemented(
                gb_type="wrap_with_autocast: unexpected freevars",
                context=str(body_lifted_freevars),
                explanation="wrap_with_autocast expects no freevars.",
                hints=[],
            )

        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = tx.output.install_subgraph(
            "wrap_body",
            body_gmod,
        )

        body_node = make_attr(tx, body_name)

        proxy_args = tuple(
            [
                *python_constants,
                body_node,
            ]
            + [operand.as_proxy() for operand in rest_args]
        )
        example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )

        return _call_function_and_unflatten_output(
            tx, self.value, proxy_args, {}, example_value, treespec, body_r
        )


class HintsWrapperHigherOrderVariable(WrapHigherOrderVariable):
    _HOP_NAME = "torch.ops.higher_order.hints_wrapper"
    _ALLOW_FALLBACK_TO_EAGER = False

    def install_subgraph_in_output_graph(
        self,
        tx: "InstructionTranslator",
        fn_vt: VariableTracker,
        fn_args_vt: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
        body_gmod: GraphModule,
        attr_name: str = "wrap_body",
    ) -> str:
        return tx.output.install_subgraph(
            "hints_wrapper_body",
            body_gmod,
        )

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        _check_supported_callable_arg(tx, args[0], "body_fn")

        # inputs
        if (
            len(args) != 3
            or not isinstance(args[1], (ListVariable, TupleVariable))
            or not isinstance(args[2], ConstDictVariable)
            or len(kwargs) != 1
            or "hints" not in kwargs
        ):
            unimplemented(
                gb_type="hints_wrapper: improper args/kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"hints_wrapper expects 3 positional arguments (got {len(args)}) "
                f"and 1 keyword argument (got {len(kwargs)}). "
                "Usage: hints_wrapper(body_fn, args, kwargs, hints=...). "
                "args is expected to be list/tuple and kwargs is expected to be a dict.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        operands = args[1].unpack_var_sequence(tx)
        fn_kwargs = args[2].as_python_constant()
        assert self._HOP_NAME is not None
        # Use create_wrapped_node from WrapHigherOrderVariable
        (
            p_args,
            _,
            example_value,
            body_r,
            body_gmod,
            _,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(
            tx,
            args[0],  # function
            operands,
            fn_kwargs,
            self._HOP_NAME,
        )

        # hints_wrapper expects (body_node, args, kwargs) as positional args
        # So we need to restructure p_args from (body_node, *lifted_args)
        # to (body_node, lifted_args_tuple, {})
        body_node = p_args[0]
        lifted_args = p_args[1:]
        # pyrefly: ignore [implicit-any]
        p_args = (body_node, tuple(lifted_args), {})

        # add hints into p_kwargs
        p_kwargs = {}
        p_kwargs["hints"] = kwargs["hints"].as_python_constant()

        return _call_function_with_auto_output_flattening(  # type: ignore[return-type]
            tx,
            self.value,
            p_args,
            p_kwargs,
            example_value,
            body_r,
            body_graph_output_vts,
        )


class CheckpointHigherOrderVariable(WrapHigherOrderVariable):
    _HOP_NAME = "torch.utils.checkpoint.checkpoint"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.allow_side_effects = (
            torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint
        )

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from torch._higher_order_ops.wrap import TagActivationCheckpoint
        from torch.utils.checkpoint import noop_context_fn

        context_fn = None
        if "context_fn" in kwargs and kwargs["context_fn"] is not noop_context_fn:
            ctx = kwargs.pop("context_fn")
            if isinstance(ctx, torch._dynamo.variables.UserFunctionVariable):
                context_fn = ctx.fn
            elif isinstance(
                ctx, torch._dynamo.variables.functions.FunctoolsPartialVariable
            ):
                context_fn = ctx.guard_as_python_constant()
            else:
                raise NotImplementedError(
                    f"checkpoint not implemented for {type(ctx)} context_fn"
                )

        checkpoint_kwargs, gmod_kwargs = TagActivationCheckpoint.divide_kwargs(kwargs)

        # Here we use checkpoint_kwargs (and not gmod kwargs). gmod_kwargs are
        # already flattened above and managed inside the fx graph.
        (
            p_args,
            _,
            example_value,
            _body_r,
            checkpointed_gmod,
            _,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(
            tx,
            args[0],
            args[1:],
            gmod_kwargs,
            "torch.utils.checkpoint.checkpoint",
        )
        if context_fn is not None:
            checkpointed_gmod.meta["_checkpoint_context_fn"] = context_fn

        _, checkpoint_kwargs = proxy_args_kwargs([], checkpoint_kwargs)

        return _call_function_with_auto_output_flattening(  # type: ignore[return-value]
            tx,
            self.value,
            p_args,
            checkpoint_kwargs,
            example_value,
            _body_r,
            body_graph_output_vts,
        )


class DynamoBypassingWrapperHigherOrderVariable(WrapHigherOrderVariable):
    _HOP_NAME = "torch.ops.higher_order.dynamo_bypassing_wrapper"

    def __init__(self, hop: HigherOrderOperator, source: Source | None) -> None:
        super().__init__(hop, source)

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        func_var = args[0]

        if isinstance(func_var, torch._dynamo.variables.UserFunctionVariable):
            func = func_var.fn
        elif isinstance(
            func_var, torch._dynamo.variables.functions.FunctoolsPartialVariable
        ):
            func = func_var.as_python_constant()
        else:
            raise RuntimeError(
                f"DynamoBypassingWrapperHigherOrderVariable: Unsupported function {type(func_var)}"
            )
        (
            p_args,
            _,
            example_value,
            _body_r,
            gmod,
            _,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(
            tx,
            args[1],
            args[2:],
            kwargs,
            str(func),
        )

        # Alternatively, we could've stored only the function's fqn and
        # reconstructed, but that requires the function to be a global.
        gmod_meta_key = "_dynamo_bypassing_wrapper_fn"
        gmod.meta[gmod_meta_key] = func

        return _call_function_with_auto_output_flattening(  # type: ignore[return-value]
            tx,
            self.value,
            (gmod_meta_key,) + tuple(p_args),
            {},
            example_value,
            _body_r,
            body_graph_output_vts,
        )


@add_hop_context
class AutogradFunctionApplyVariable(VariableTracker):
    _HOP_NAME: str = "autograd.Function"
    _ALLOW_FALLBACK_TO_EAGER = True

    def __init__(
        self, fwd_fn: Any, bwd_fn: Any, parent_source: Source | None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.fwd_fn = fwd_fn
        self.bwd_fn = bwd_fn
        self.parent_source = parent_source

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        """
        At the highest level, the goal of tracing an autograd.Function is to
        essentially emit a new autograd.Function object. To do this, Dynamo
        traces fwd and bwd graph and then inserts a AutogradFunctionApply HOP in
        the graph that call the traced fwd and bwd graph in the `forward` and
        `backward` methods respectively. AOTDispatcher desugars this HOP and
        just inlines the hop fwd and bwd into the main graph during its tracing.

        However, the traced forward and backward graphs cannot be directly
        placed in the new autograd.Function because autograd.Function has some
        requirements.

        a) # fwd graph inputs = # bwd graph outputs
        b) # fwd graph outputs = # bwd graph inputs
        c) Since the graphs do not have ctx variable, we have to manually return
        the saved_tensors from the forward and have additional inputs in the
        backward, and wire the connections.

        Unfortunately, reworking the initial traced fwd and bwd graphs to
        satisfy the above 3 conditions leads to a very tedious codebase.

        Lets look at an example

        class Foo:
            def __init__(self):
                self.a = 4

        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, foo):
                ctx.save_for_backward(x)
                return x.sin() + foo.a

            @staticmethod
            def backward(ctx, grad):
                x, = ctx.saved_tensors
                return grad * x.cos()

        We want the resulting graphs to look like:

        # Note that Dynamo lifts the foo_a directly as an input.
        def fwd(ctx, x, foo_a):
            # (output, saved tensors / attrs)
            return (x.sin() + foo_a, (x))

        # Note that backward graph has None as the second output to match the
        # fwd requirements (even though the original backward function has just
        # output)
        def bwd(ctx, grad, x):
            return grad * x.cos(), None


        To accomplish this, we're going to:
        1. Construct a ctx object
        2. Speculate subgraph forward
        3. Speculate subgraph backward
        4. rewired_bwd_graph_inputs - Use the traced fwd graph as the anchor point, and rewire the backward graph outputs
        5. handle_saved_tensors_wiring - Handle the saved tensors, as mentioned in (c)
        """

        fwd_tracer = torch._dynamo.output_graph.SubgraphTracer(
            tx.output,
            parent=tx.output.current_tracer,
            source_target=self._HOP_NAME,
        )

        ctx = self.prepare_ctx_vt(tx, args, kwargs)

        fwd_fn, fwd_out, fwd_graph, fwd_freevars, fwd_graph_output_vts = (
            self.trace_forward_graph(tx, ctx, fwd_tracer, args, kwargs)
        )

        bwd_args, bwd_out, bwd_graph, bwd_freevars, bwd_graph_output_vts = (
            self.trace_backward_graph(tx, ctx, fwd_tracer, fwd_out, fwd_fn)
        )

        self.rewire_bwd_graph_outputs(
            fwd_freevars, bwd_out, bwd_graph, bwd_freevars, args
        )

        fwd_graph, bwd_graph = self.handle_saved_tensors_wiring(
            fwd_out,
            fwd_graph,
            fwd_freevars,
            fwd_graph_output_vts,  # type: ignore[arg-type]
            bwd_graph,
            bwd_freevars,
        )

        # If users call ctx.mark_non_differentiable, we should capture these output tensors who
        # are marked as non-differentiable and pass them to ApplyTemplate
        # at torch._functorch.autograd_function.AutogradFunctionApply for reconstruction.
        non_differentiable_idx = []
        if ctx.non_differentiable is not None:
            non_differentiable_set = set(ctx.non_differentiable)
            assert isinstance(fwd_out, variables.BaseListVariable)
            for i, x in enumerate(fwd_out.items):
                if x.is_tensor() and x.as_proxy() in non_differentiable_set:
                    non_differentiable_idx.append(i)

        # See Note [Activations with no version counter checks in eager]
        # Compute which tensors in bwd_freevars came from ctx.save_for_backward.
        # This allows AOT autograd to distinguish between tensors saved via
        # save_for_backward vs those stashed directly on ctx (e.g., ctx.x = x).
        saved_for_backward_idx = []
        if ctx.saved_tensors is not None and len(ctx.saved_tensors.tensors) > 0:
            # Build a set of proxies that were passed to save_for_backward
            saved_tensor_proxies = OrderedSet()
            for tensor_vt in ctx.saved_tensors.tensors:
                if tensor_vt.is_tensor():
                    saved_tensor_proxies.add(tensor_vt.as_proxy())

            # bwd_freevars is a dict of outer-graph proxy -> inner-graph proxy
            # for all tensors passed from fwd to bwd. Find which indices
            # correspond to save_for_backward tensors.
            for i, fwd_proxy in enumerate(bwd_freevars.keys()):
                if fwd_proxy in saved_tensor_proxies:
                    saved_for_backward_idx.append(i)

        # Store fwd_body
        fwd_nn_modules = tx.output.tracing_context.module_context.copy_graphstate()
        fwd_name = tx.output.install_subgraph(
            "fwd_body",
            torch.fx.GraphModule(fwd_nn_modules.nn_modules, fwd_graph),
        )
        fwd_node = make_attr(tx, fwd_name)

        # Store bwd_body
        bwd_nn_modules = tx.output.tracing_context.module_context.copy_graphstate()
        bwd_name = tx.output.install_subgraph(
            "bwd_body",
            torch.fx.GraphModule(bwd_nn_modules.nn_modules, bwd_graph),
        )
        bwd_node = make_attr(tx, bwd_name)

        p_args = (
            fwd_node,
            bwd_node,
            *list(fwd_freevars.keys()),
        )
        kwargs_for_fn = {
            "non_differentiable_idx": non_differentiable_idx,
            "saved_for_backward_idx": saved_for_backward_idx,
        }

        # Store the invocation as a call
        from torch._functorch.autograd_function import autograd_function_apply

        # We use speculate_subgraph to get the fwd graph, but it's always under no grad mode like what eager mode does.
        # The fwd outputs (tensor's example_value) need to be inferred from fake tensor prop to get the correct attributes
        # (e.g, tensor.requires_grad), which would be used by downstream Dynamo tracing.
        # Since there can be other ops like Triton kernels, which depends on python dispatcher, we have to enable it.
        # TODO - revisit if we need the python dispatcher
        with enable_python_dispatcher():
            with tx.output.fake_mode:
                fwd_freevars_args = [_get_fake_value(arg) for arg in fwd_freevars]

                example_value = autograd_function_apply(
                    tx.output.nn_modules[fwd_node.node.name],
                    tx.output.nn_modules[bwd_node.node.name],
                    *fwd_freevars_args,
                    **kwargs_for_fn,
                )

        flat_variable = add_call_function(
            tx, autograd_function_apply, p_args, kwargs_for_fn, example_value
        )
        # type: ignore[arg-type]
        overwrite_tensor_vt_proxy(fwd_graph_output_vts, flat_variable)
        # type: ignore[arg-type]
        overwrite_tensor_vt_requires_grad(fwd_graph_output_vts, flat_variable)
        return fwd_out

    def prepare_ctx_vt(
        self, tx: "InstructionTranslator", args: Any, kwargs: Any
    ) -> "AutogradFunctionContextVariable":
        from .. import AutogradFunctionContextVariable

        ctx = AutogradFunctionContextVariable.create(tx, args, kwargs)
        with discard_graph_changes(tx):
            # A little hacky, but we need a dummy ctx proxy for speculate_subgraph.
            # We should clean this up at some point.
            proxy = tx.output.create_proxy(
                "call_function", torch.autograd.function.FunctionCtx, (), {}
            )
            # type: ignore[attr-defined]
            set_example_value(proxy.node, ctx.value)
            # type: ignore[attr-defined]
            ctx.proxy = proxy
        # pyrefly: ignore[bad-return]
        return ctx

    def trace_forward_graph(
        self,
        tx: "InstructionTranslator",
        ctx: "AutogradFunctionContextVariable",
        fwd_tracer: "SubgraphTracer",
        args: Any,
        kwargs: Any,
    ) -> tuple[
        VariableTracker,
        VariableTracker,
        torch.fx.Graph,
        dict[Proxy, Proxy],
        VariableTracker | tuple[VariableTracker, ...],
    ]:
        """
        Traces the forward method of the autograd.Function object.
        """
        from torch._functorch.autograd_function import DynamoAutogradFunctionTraceHelper

        fwd_fn, fwd_args = self.prepare_fn_vt(tx, ctx, "forward", args)

        # autograd.Function forward does a few things like running in no_grad
        # mode and also applying view_as for input tensors that are returned as
        # outputs. Therefore, we wrap the original forward in a helper that have
        # those extra bits for Dynamo to trace.
        fwd_fn = _make_inlined(tx, DynamoAutogradFunctionTraceHelper.fwd_trace_helper)(
            fwd_fn
        )

        # Speculate subgraph on the fwd
        fwd_out, fwd_graph, fwd_freevars, fwd_graph_output_vts, _ = (
            _higher_order_ops().speculate_subgraph_with_auto_output_flattening(
                tx,
                fwd_fn,
                fwd_args,
                kwargs,
                self._HOP_NAME,
                enable_grad=None,
                set_subgraph_inputs="automatic",
                allow_side_effects=True,
                tracer=fwd_tracer,
            )
        )

        # There could be unused inputs in the forward, and Dynamo might not
        # capture them. We must lift them as inputs, because even though they
        # are not used in forward, we still need to account for their gradients
        # in the backward.
        for arg in args:
            if arg.is_tensor():
                fwd_tracer.maybe_lift_tracked_freevar_to_input(arg.as_proxy())

        if ctx in tx.output.side_effects.store_attr_mutations:
            if (
                "_materialize_non_diff_grads"
                in tx.output.side_effects.store_attr_mutations[ctx]
            ):
                unimplemented(
                    gb_type="autograd.Function.apply: _materialize_non_diff_grads mutation",
                    context="",
                    explanation="Mutations to autograd.Function.ctx._materialize_non_diff_grads are not supported.",
                    hints=[
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

        return fwd_fn, fwd_out, fwd_graph, fwd_freevars, fwd_graph_output_vts

    def trace_backward_graph(
        self,
        tx: "InstructionTranslator",
        ctx: "AutogradFunctionContextVariable",
        fwd_tracer: "SubgraphTracer",
        fwd_out: VariableTracker,
        fwd_fn: VariableTracker,
    ) -> tuple[
        Sequence[VariableTracker],
        VariableTracker,
        torch.fx.Graph,
        dict[Proxy, Proxy],
        VariableTracker | tuple[VariableTracker, ...],
    ]:
        """
        Traces the backward method of the autograd.Function object.
        """
        from .. import UserMethodVariable

        # Note that for the forward, we do not restore side effects, because we
        # want the later tracing to see the side-effects. But for backward, we
        # are just trying to capture the graph, and therefore we must restore
        # the side effects.
        prev_side_effects = tx.output.side_effects

        # Speculate subgraph on the backward. We make the bwd tracer a child of
        # the fwd tracer, because backward may rely on tensors/attrs created in
        # the fwd tracer.
        bwd_tracer = torch._dynamo.output_graph.SubgraphTracer(
            tx.output,
            parent=fwd_tracer,
            source_target=self._HOP_NAME,
        )

        bwd_args = []
        if fwd_out.is_tensor():
            bwd_args.append(fwd_out)
        else:
            assert isinstance(fwd_out, variables.BaseListVariable)
            for i in fwd_out.items:
                if i.is_tensor():
                    bwd_args.append(i)
                else:
                    bwd_args.append(CONSTANT_VARIABLE_NONE)

        bwd_fn, bwd_args = self.prepare_fn_vt(tx, ctx, "backward", bwd_args)

        def is_strict_for(v: VariableTracker) -> bool:
            if v.is_tensor():
                # we can be more lax for stuff from forward
                return v.proxy.tracer is not fwd_tracer  # type: ignore[attr-defined]
            return True

        # automatic_with_forced_inputs relies on the function arg names to
        # create a new proxy. Also, it will always INSERT a tensor placeholder
        # as input, even though it might not be used in the graph. This allows
        # us to make a mapping for the backward graph.
        with (
            tx.output.subtracer(fwd_fn, fwd_tracer),  # type: ignore[arg-type]
            tx.strict_translation_mode(is_strict_for),
        ):
            try:
                bwd_out, bwd_graph, bwd_freevars, bwd_graph_output_vts, _ = (
                    _higher_order_ops().speculate_subgraph_with_auto_output_flattening(
                        tx,
                        bwd_fn,
                        bwd_args,
                        {},
                        self._HOP_NAME,
                        # TODO - revisit if we need enable_grad
                        enable_grad=False,
                        set_subgraph_inputs="automatic_with_forced_inputs",
                        allow_side_effects=False,
                        tracer=bwd_tracer,
                    )
                )
            except torch._dynamo.exc.UnknownPropertiesDuringBackwardTrace as e:
                # TODO - Do not support this path because of eager
                # divergence forced by contiguous calls. Instead suggested
                # nonstrict_trace.
                from unittest import mock

                bwd_tracer = torch._dynamo.output_graph.SubgraphTracer(
                    tx.output,
                    parent=fwd_tracer,
                    source_target=self._HOP_NAME,
                )
                from ..._trace_wrapped_higher_order_op import (
                    autograd_function_backward_rewritten,
                )
                from ..builder import SourcelessBuilder

                if isinstance(self.bwd_fn, types.FunctionType):
                    bwd_fn = SourcelessBuilder.create(
                        tx, autograd_function_backward_rewritten(self.bwd_fn)
                    )
                elif isinstance(self.bwd_fn, types.MethodType):
                    bwd_fn = UserMethodVariable(
                        autograd_function_backward_rewritten(self.bwd_fn.__func__),
                        VariableTracker.build(tx, self.bwd_fn.__class__),
                    )
                else:
                    unimplemented(
                        gb_type="autograd.Function.apply: non-function or method backward (2)",
                        context=str(self.bwd_fn),
                        explanation="Expected backward function to be a function or method.",
                        hints=[],
                        from_exc=e,
                    )

                with mock.patch(
                    "torch._dynamo.config._autograd_backward_strict_mode_conditional_banned_ops",
                    [],
                ):
                    bwd_out, bwd_graph, bwd_freevars, bwd_graph_output_vts, _ = (
                        _higher_order_ops().speculate_subgraph_with_auto_output_flattening(
                            tx,
                            bwd_fn,
                            bwd_args,
                            {},
                            self._HOP_NAME,
                            enable_grad=False,
                            set_subgraph_inputs="automatic_with_forced_inputs",
                            allow_side_effects=False,
                            tracer=bwd_tracer,
                        )
                    )

        # Restore the side effects
        tx.output.side_effects = prev_side_effects

        return bwd_args, bwd_out, bwd_graph, bwd_freevars, bwd_graph_output_vts

    def rewire_bwd_graph_outputs(
        self,
        fwd_freevars: dict[Proxy, Proxy],
        bwd_out: VariableTracker,
        bwd_graph: torch.fx.Graph,
        bwd_freevars: dict[Proxy, Proxy],
        orig_fwd_args: Sequence[VariableTracker],
    ) -> None:
        # ---------------------------------------------------------------------
        # Forward–Backward Input/Output Alignment
        #
        # autograd.Function requires that the outputs of backward() correspond
        # exactly to the inputs of forward(). Normally this alignment is the
        # user’s responsibility. However, when Dynamo synthesizes a new
        # autograd.Function for a traced region, Dynamo must perform this
        # alignment automatically.
        #
        # To do this, Dynamo uses the *original* forward call site as the anchor
        # that defines how forward inputs map to backward outputs.
        #
        # ---------------------------------------------------------------------
        # Terminology
        #
        # fwd_freevars / bwd_freevars:
        #     Maps from *outer-graph proxies* to *inner-graph placeholder
        #     proxies*. Keys are always outer-graph proxies (these may be actual
        #     user inputs or intermediate values lifted into the subgraph).
        #
        # orig_fwd_args:
        #     VariableTrackers for the forward() inputs. Since these correspond
        #     to user-exposed arguments, each tracker points to an *outer-graph*
        #     proxy.
        #
        # bwd_outs:
        #     VariableTrackers for the backward() outputs. These usually point to
        #     *inner-graph* proxies, except for cases where a forward input is
        #     passed directly through to a backward output—in which case the
        #     tracker may still refer to an outer-graph proxy.
        #
        # ---------------------------------------------------------------------
        # Goal
        #
        # To ensure forward–backward consistency, we must rewire the backward
        # graph outputs so that they line up with the forward graph inputs.
        #
        # We build a mapping from outer-graph proxy → inner-graph proxy using
        # orig_fwd_args and bwd_outs, then iterate over the fwd_graph inputs to
        # determine which backward outputs must be generated (or padded with
        # None) to satisfy autograd’s calling convention.
        #
        # ---------------------------------------------------------------------
        # Example
        #
        # Suppose the forward receives a user-defined object:
        #
        # @dataclass
        # class Weird:
        #     x: int
        #     b: torch.Tensor
        #     c: torch.Tensor
        #
        # class Foo(torch.autograd.Function):
        #     @staticmethod
        #     def forward(ctx, x: torch.Tensor, weird: Weird, z: torch.Tensor):
        #         ctx.save_for_backward(weird.b, weird.c)
        #         return weird.b * weird.c * x.clone()
        #
        #     @staticmethod
        #     def backward(ctx, grad):
        #         b, c = ctx.saved_tensors
        #         return grad * b * c, None, grad * 2
        #
        # Dynamo lifts the tensor fields of the user-defined object for the trace:
        #
        # fwd_graph():
        #     %l_weird_b : FakeTensor = placeholder[target=l_weird_b]
        #     %l_weird_c : FakeTensor = placeholder[target=l_weird_c]
        #     %l_x_      : FakeTensor = placeholder[target=l_x_]
        #     %l_z_      : FakeTensor = placeholder[target=l_z_]
        #     ...
        #     return (outs,)
        #
        # The initial backward graph:
        #
        # bwd_graph():
        #     %grad       : Tensor    = placeholder[target=grad]
        #     %l_weird_b  : FakeTensor = placeholder[target=l_weird_b]
        #     %l_weird_c  : FakeTensor = placeholder[target=l_weird_c]
        #     ...
        #     return (mul_1, mul_2)
        #
        # The forward graph has 4 inputs, but the backward graph produces only 2
        # outputs, and their ordering does not match the forward argument order.
        #
        # So Dynamo rewires the backward graph outputs to align with the forward
        # inputs:
        #
        # bwd_graph():
        #     ...
        #     return (None, None, mul_1, mul_2)
        #
        # This ensures the synthesized autograd.Function conforms to PyTorch’s
        # forward/backward contract.
        # ---------------------------------------------------------------------

        def get_bwd_node(vt: VariableTracker) -> torch.fx.Node:
            # Backward tensor vt here can be - (1) an intermediate, or (2) input
            # to the backward graph. If it is an input to the backward graph, we have to lookup bwd_freevars to get the inner proxy.
            return bwd_freevars.get(vt.proxy, vt.proxy).node  # type: ignore[attr-defined]

        # Find the mapping between orig_fwd_args and bwd_out
        # pyrefly: ignore [implicit-any]
        outer_fwd_proxy_to_bwd_node = {}
        if isinstance(bwd_out, variables.BaseListVariable):
            bwd_outs = bwd_out.items
            for idx, fwd_arg in enumerate(orig_fwd_args):
                # We care about tensor args. For non-tensor args, the bwd output returns None.
                if fwd_arg.is_tensor():
                    bwd_out_at_idx = bwd_outs[idx]
                    if bwd_out_at_idx.is_tensor():
                        # type: ignore[attr-defined]
                        outer_fwd_proxy_to_bwd_node[fwd_arg.proxy] = get_bwd_node(
                            bwd_out_at_idx
                        )
                    else:
                        # backward can return None at the output
                        assert (
                            isinstance(bwd_out_at_idx, variables.ConstantVariable)
                            and bwd_out_at_idx.value is None
                        )
                        # type: ignore[attr-defined]
                        outer_fwd_proxy_to_bwd_node[fwd_arg.proxy] = None

        elif bwd_out.is_tensor():
            # type: ignore[attr-defined]
            outer_fwd_proxy_to_bwd_node[orig_fwd_args[0].proxy] = get_bwd_node(bwd_out)

        # Ideally, we should have walked through the fwd placeholders. But we
        # can instead walk through the fwd_freevars, which is a insertion sorted
        # dictionary and therefore represents the outer_proxies for the
        # placeholder in the same order as that as placeholders.
        rewired_bwd_outputs = [
            outer_fwd_proxy_to_bwd_node.get(fwd_proxy) for fwd_proxy in fwd_freevars
        ]

        for node in bwd_graph.find_nodes(op="output"):
            bwd_graph.erase_node(node)
            break
        bwd_graph.output(tuple(rewired_bwd_outputs))
        bwd_graph.lint()

    def handle_saved_tensors_wiring(
        self,
        fwd_out: VariableTracker,
        fwd_graph: torch.fx.Graph,
        fwd_freevars: dict[Proxy, Proxy],
        fwd_graph_body_outputs: Sequence[VariableTracker],
        bwd_graph: torch.fx.Graph,
        bwd_freevars: dict[Proxy, Proxy],
    ) -> tuple[torch.fx.Graph, torch.fx.Graph]:
        # ---------------------------------------------------------------------
        # Rewiring Forward Outputs to Backward Inputs (and Handling Saved Tensors)
        #
        # In `rewire_bwd_graph_outputs`, we aligned the *forward inputs* with the
        # *backward outputs*. This method performs the complementary task:
        # aligning the *forward outputs* with the *backward inputs*, while also
        # incorporating all tensors saved via ctx.save_for_backward.
        #
        # There are two main issues we must resolve:
        #
        # (1) Forward outputs may contain non-tensor values.
        #     This means the number of tensors visible in fwd_out may not match
        #     the number of tensors produced by the traced forward graph. As a
        #     result, the backward graph’s placeholders may not line up with the
        #     actual tensor outputs.
        #
        # (2) The backward graph may require intermediate tensors saved during
        #     the forward pass (via save_for_backward), but those intermediates
        #     might not currently be included among the forward graph’s outputs.
        #
        # Together, these issues mean that the bwd_graph input signature may be
        # inconsistent with what fwd_graph outputs, and we need to rewrite both.
        #
        # Lets look at an example to understand the transformation
        #
        # class Add(torch.autograd.Function):
        #     @staticmethod
        #     def forward(ctx, x, y):
        #         a = torch.sin(x)
        #         b = torch.cos(y)
        #         ctx.save_for_backward(a)
        #         return Foo(a, b), x * y

        #     @staticmethod
        #     def backward(ctx, grad_a, grad_b):
        #         (a,) = ctx.saved_tensors
        #         return grad_b * 2, a * grad_b * 3

        # Before
        # fwd_graph():
        #     %l_x_ : torch._subclasses.fake_tensor.FakeTensor [num_users=2] = placeholder[target=l_x_]
        #     %l_y_ : torch._subclasses.fake_tensor.FakeTensor [num_users=2] = placeholder[target=l_y_]
        #     ....
        #     return (a, b, out)
        #
        # bwd_graph():
        #     %grad_b : torch.Tensor [num_users=2] = placeholder[target=grad_b]
        #     %a : torch._subclasses.fake_tensor.FakeTensor [num_users=1] = placeholder[target=a]
        #     ....
        #     return (mul, mul_2)
        #
        # The problems here:
        #   (1) fwd_graph has 3 tensor outputs (a, b, out), but bwd_graph has
        #       only 1 gradient input - grad_b. We need 3.
        #
        #   (2) bwd_graph uses `a` (a saved tensor) as an input, but fwd_graph
        #       does not currently return `a`. To make `a` available to the
        #       backward graph, the forward graph must expose it as part of its
        #       output signature.
        #
        # After this transformation
        # fwd_graph():
        #     %l_x_ : torch._subclasses.fake_tensor.FakeTensor [num_users=2] = placeholder[target=l_x_]
        #     %l_y_ : torch._subclasses.fake_tensor.FakeTensor [num_users=2] = placeholder[target=l_y_]
        #     .....
        #     return ((a, b, out), (a,))
        # bwd_graph():
        #     %unused_0 : [num_users=0] = placeholder[target=unused_0]
        #     %unused_1 : [num_users=0] = placeholder[target=unused_1]
        #     %grad_b : [num_users=2] = placeholder[target=grad_b]
        #     %a : [num_users=1] = placeholder[target=a]
        #     .....
        #     return (mul, mul_2)
        #
        # Key changes:
        #
        #   1) The forward graph now returns:
        #           (existing_outputs), (saved_tensors)
        #      This exposes saved intermediates (`a`) as part of the fwd output
        #      structure, making them available to backward.
        #
        #   2) The backward graph input signature is rewritten to:
        #           (*grads_for_existing_outputs, *saved_tensors)
        #      This ensures the counts and ordering match the new fwd_graph
        #      output structure. Placeholders corresponding to tensors whose
        #      gradients are unused (e.g., `a`, `b`) appear as `%unused_*`.
        #
        # This alignment ensures that the synthesized autograd.Function follows
        # PyTorch’s forward/backward calling convention and that all required
        # saved tensors are available to the backward graph.
        # ---------------------------------------------------------------------

        # To address Problem (1), we must determine which backward-graph inputs
        # correspond to the forward-graph outputs.
        #
        # We use two facts:
        #   • `fwd_out` preserves the original forward output order.
        #   • Backward-graph inputs are also ordered according to the backward()
        #     method signature, thanks to automatic_with_forced_inputs.
        #
        # For any forward output that is *not* a tensor, there is no
        # corresponding tensor placeholder in the backward graph. During tracing,
        # we intentionally inserted a `None` VariableTracker for these positions,
        # so the backward graph contains no placeholder for them.
        bwd_input_nodes = list(bwd_graph.find_nodes(op="placeholder"))
        # pyrefly: ignore [implicit-any]
        fwd_vt_to_bwd_node = {}
        bwd_idx = 0
        if isinstance(fwd_out, variables.BaseListVariable):
            for fwd_vt in fwd_out.items:
                if fwd_vt.is_tensor():
                    fwd_vt_to_bwd_node[fwd_vt] = bwd_input_nodes[bwd_idx]
                    bwd_idx += 1
        else:
            if fwd_out.is_tensor():
                fwd_vt_to_bwd_node[fwd_out] = bwd_input_nodes[bwd_idx]
                bwd_idx += 1

        rewired_bwd_graph_inputs = []
        for fwd_graph_vt in fwd_graph_body_outputs:
            # for tensor vts that were part of a user-defined object (like in
            # the above example), we just set None for now. Later, we will use
            # these None to insert a unused placeholder.
            # type: ignore[arg-type]
            rewired_bwd_graph_inputs.append(fwd_vt_to_bwd_node.get(fwd_graph_vt))

        # To address Problem (2), we must incorporate any tensors that were saved
        # (or otherwise smuggled) from the forward pass into the backward graph.
        #
        # Fortunately, these are easy to identify: they appear in `bwd_freevars`.
        # `bwd_freevars` maps outer-graph lifted proxies to inner-graph placeholder
        # proxies. Because the backward graph is traced using proxies originating
        # from `fwd_out`, any value lifted into the backward graph represents a
        # saved/smuggled tensor.
        #
        # Once we identify these saved tensors, we must also locate their
        # corresponding forward-graph proxies so that the forward graph can return
        # these tensors as part of its output signature.
        extra_fwd_output_nodes = []
        for fwd_proxy, bwd_inner_proxy in bwd_freevars.items():
            # For backward, its easy, just get the node from bwd_inner_proxy
            rewired_bwd_graph_inputs.append(bwd_inner_proxy.node)

            # For the fwd_proxy, it could be a proxy from the outer graph, or it
            # could be an intermediate.
            # First ensure that's its inner fwd proxy
            inner_fwd_proxy = fwd_freevars.get(fwd_proxy, fwd_proxy)
            extra_fwd_output_nodes.append(inner_fwd_proxy.node)

        # Mechanical steps from here on. We have the extra_fwd_outputs and rewired_bwd_inputs. Lets make the changes.
        # Lets change the fwd graph outputs.
        # pyrefly: ignore [implicit-any]
        fwd_output_nodes = []
        for node in fwd_graph.find_nodes(op="output"):
            fwd_output_nodes = node.args[0]
            fwd_graph.erase_node(node)
            break

        # The signature is now ((*existing_outputs), (*extra_outputs)). Please
        # take a look at AutogradFunctionApply where we take the saved_tensors
        # out in the forward method to save for backward.
        new_fwd_graph_outputs = (fwd_output_nodes, tuple(extra_fwd_output_nodes))
        fwd_graph.output(new_fwd_graph_outputs)
        fwd_graph.lint()

        # Now lets change the bwd graph.
        new_graph = torch.fx.Graph()
        env = {}

        count = itertools.count()

        for node in rewired_bwd_graph_inputs:
            if node is None:
                new_node = new_graph.placeholder(f"unused_{next(count)}")
            else:
                new_node = new_graph.placeholder(node.name)
                new_node.meta = copy.copy(node.meta)
            env[node] = new_node

        for node in bwd_graph.nodes:
            if node.op == "placeholder":
                assert node in env
            else:
                env[node] = new_graph.node_copy(node, lambda x: env[x])
                env[node].meta = copy.copy(node.meta)

        new_graph.lint()
        return fwd_graph, new_graph

    def prepare_fn_vt(
        self,
        tx: "InstructionTranslator",
        ctx: "AutogradFunctionContextVariable",
        method_name: str,
        args: Sequence[VariableTracker],
    ) -> tuple[VariableTracker, Sequence[VariableTracker]]:
        from .. import UserMethodVariable

        source = None
        if self.parent_source:
            source = AttrSource(self.parent_source, member=method_name)

        if method_name == "forward":
            fn = self.fwd_fn
        else:
            fn = self.bwd_fn

        fn_vt, fn_args = None, None
        if isinstance(fn, types.FunctionType):
            fn_vt = VariableTracker.build(tx, fn, source=source)
            fn_args = [ctx, *args]
        elif isinstance(fn, types.MethodType):
            cls_vt = VariableTracker.build(tx, fn.__class__)
            fn_vt = UserMethodVariable(
                fn.__func__,
                cls_vt,
                source=source,
            )
            fn_args = [cls_vt, ctx, *args]
        else:
            unimplemented(
                gb_type="autograd.Function.apply: non-function or method forward",
                context=str(fn),
                explanation=f"Expected {method_name} to be a function or method.",
                hints=[],
            )
        assert fn_vt is not None and fn_args is not None
        return fn_vt, fn_args


class BaseHOPVariable(WrapHigherOrderVariable):
    # Generic fallback for BaseHOP instances not explicitly mapped
    # The actual HOP name comes from self.value._name at runtime
    _HOP_NAME = "base HOP (name not yet determined)"
    supports_input_mutation = False
    supports_aliasing = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._HOP_NAME = self.value._name

    def python_type(self) -> type:
        return type(self.value)

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        (
            p_args,
            p_kwargs,
            example_value,
            body_r,
            _,
            _,
            body_graph_output_vts,
            _,
        ) = self.create_wrapped_node(
            tx, args[0], args[1:], {}, self.value._name, subgraph_name="subgraph"
        )
        assert len(p_kwargs) == 0

        p_kwargs = {key: value.as_proxy() for key, value in kwargs.items()}
        return _call_function_with_auto_output_flattening(  # type: ignore[return-value]
            tx,
            self.value,
            p_args,
            p_kwargs,
            example_value,
            body_r,
            body_graph_output_vts,
        )
