from typing import TYPE_CHECKING

from .common import (
    Any,
    cast,
    DictGetItemSource,
    discard_graph_changes,
    graph_break_hints,
    GraphModule,
    ListVariable,
    make_attr,
    Proxy,
    proxy_args_kwargs,
    pytree,
    Sequence,
    set_example_value,
    speculate_subgraph,
    torch,
    TorchHigherOrderOperatorVariable,
    TupleVariable,
    unimplemented,
    UnspecializedNNModuleVariable,
    Unsupported,
    VariableTracker,
)


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


class FlexAttentionBackwardHighOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.flex_attention_backward"

    @staticmethod
    def _uses_pretraced_graphs(
        fw_graph: VariableTracker, joint_graph: VariableTracker
    ) -> bool:
        return not joint_graph.is_constant_none() or isinstance(
            fw_graph, UnspecializedNNModuleVariable
        )

    def proxy_submod(
        self, tx: "InstructionTranslator", arg: UnspecializedNNModuleVariable
    ) -> Proxy:
        assert arg.source and isinstance(arg.source.base, DictGetItemSource)  # type: ignore[attr-defined]
        submod_name = tx.output.install_subgraph(arg.source.base.index, arg.value)  # type: ignore[arg-type]
        p_submod = make_attr(tx, submod_name)
        set_example_value(p_submod.node, arg.value)
        return p_submod

    def to_proxy(self, tx: "InstructionTranslator", arg: VariableTracker) -> Any:
        if isinstance(arg, UnspecializedNNModuleVariable):
            return self.proxy_submod(tx, arg)
        elif isinstance(arg, (ListVariable, TupleVariable)):
            return arg.python_type()(
                self.to_proxy(tx, nested_arg) for nested_arg in arg.items
            )
        else:
            return arg.as_proxy()

    def create_wrapped_node(
        self,
        tx: "InstructionTranslator",
        query: VariableTracker,
        fn: VariableTracker,
        fn_name: str,
        other_buffers: Sequence[VariableTracker],
    ) -> tuple[Proxy, tuple[Proxy, ...], torch.fx.GraphModule]:
        from ..._trace_wrapped_higher_order_op import TransformGetItemToIndex

        def create_scalar() -> VariableTracker:
            return query.call_method(
                tx,
                "new_empty",
                [VariableTracker.build(tx, [])],
                {"dtype": VariableTracker.build(tx, torch.int32)},
            )

        with discard_graph_changes(tx):
            bhmn = [create_scalar() for _ in range(4)]
            if fn_name == "score_mod":
                scores_require_grad: bool = query.requires_grad  # type: ignore[attr-defined]
                score = query.call_method(
                    tx,
                    "new_empty",
                    [VariableTracker.build(tx, [])],
                    {"requires_grad": VariableTracker.build(tx, scores_require_grad)},
                )
                new_args = [score, *bhmn, *other_buffers]
            else:
                assert fn_name == "mask_fn", "Illegal function name: " + fn_name
                new_args = [*bhmn, *other_buffers]

        with TransformGetItemToIndex():
            (
                (_body_output, _body_spec),
                body_graph,
                body_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                fn,
                new_args,
                {},
                description=f"{self._HOP_NAME}: {fn_name}",
                source_target=self.value,
                set_subgraph_inputs="flatten_manual",
            )

        gm = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = tx.output.install_subgraph(fn_name, gm)
        return make_attr(tx, body_name), tuple(body_lifted_freevars), gm

    @staticmethod
    def _buffer_example_value(buffer: VariableTracker) -> Any:
        proxy = buffer.as_proxy()
        if isinstance(proxy, Proxy):
            return proxy.node.meta["example_value"]
        return proxy

    def _derive_joint_graph(
        self,
        tx: "InstructionTranslator",
        query: VariableTracker,
        fw_graph_gm: torch.fx.GraphModule,
        score_mod_other_buffers: TupleVariable,
        fw_graph_lifted_args: tuple[Proxy, ...],
    ) -> Proxy:
        from torch._higher_order_ops.flex_attention import create_fw_bw_graph

        query_example = query.as_proxy().node.meta["example_value"]
        example_vals = (
            query_example.new_zeros((), requires_grad=True),
            query_example.new_zeros((), dtype=torch.int),
            query_example.new_zeros((), dtype=torch.int),
            query_example.new_zeros((), dtype=torch.int),
            query_example.new_zeros((), dtype=torch.int),
        )
        all_buffer_examples = tuple(
            self._buffer_example_value(buf) for buf in score_mod_other_buffers.items
        ) + tuple(buf.node.meta["example_value"] for buf in fw_graph_lifted_args)

        _, joint_gm = create_fw_bw_graph(fw_graph_gm, example_vals, all_buffer_examples)
        joint_gm = cast(GraphModule, joint_gm)

        submod_name = tx.output.install_subgraph("joint_graph", joint_gm)
        p_submod = make_attr(tx, submod_name)
        set_example_value(p_submod.node, joint_gm)
        return p_submod

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..builder import wrap_fx_proxy

        if len(args) != 14:
            return self._call_function_fallback(tx, args, kwargs)

        (
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            fw_graph,
            joint_graph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        ) = args

        if (
            not isinstance(block_mask, TupleVariable)
            or not isinstance(score_mod_other_buffers, TupleVariable)
            or not isinstance(mask_mod_other_buffers, TupleVariable)
            or len(block_mask.items) < 1
        ):
            return self._call_function_fallback(tx, args, kwargs)

        if self._uses_pretraced_graphs(fw_graph, joint_graph):
            return self._call_function_fallback(tx, args, kwargs)

        fw_graph_node, fw_graph_lifted_args, fw_graph_gm = self.create_wrapped_node(
            tx, query, fw_graph, "score_mod", score_mod_other_buffers.items
        )

        joint_graph_node = self._derive_joint_graph(
            tx,
            query,
            fw_graph_gm,
            score_mod_other_buffers,
            fw_graph_lifted_args,
        )

        mask_fn = block_mask.items[-1]
        if mask_fn.is_python_constant() and mask_fn.as_python_constant() is None:
            mask_fn = VariableTracker.build(
                tx,
                torch.nn.attention.flex_attention.noop_mask,
                source=mask_fn.source,
            )
        mask_fn_node, mask_fn_lifted_args, _ = self.create_wrapped_node(
            tx, query, mask_fn, "mask_fn", mask_mod_other_buffers.items
        )

        proxied_args = [
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            TupleVariable(block_mask.items[:-1], source=block_mask.source),
            scale,
            kernel_options,
        ]
        inp_args, _ = proxy_args_kwargs(proxied_args, {})
        proxied_score_mod_other_buffers = tuple(
            self.to_proxy(tx, arg) for arg in score_mod_other_buffers.items
        )
        proxied_mask_mod_other_buffers = tuple(
            self.to_proxy(tx, arg) for arg in mask_mod_other_buffers.items
        )

        (
            inp_q,
            inp_k,
            inp_v,
            inp_out,
            inp_lse,
            inp_grad_out,
            inp_grad_lse,
            inp_block_mask,
            inp_scale,
            inp_kernel_options,
        ) = inp_args

        block_mask_proxy = tuple(inp_block_mask + (mask_fn_node,))

        with torch.fx.experimental.proxy_tensor.set_original_aten_op(self.value):
            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    self.value,
                    args=(
                        inp_q,
                        inp_k,
                        inp_v,
                        inp_out,
                        inp_lse,
                        inp_grad_out,
                        inp_grad_lse,
                        fw_graph_node,
                        joint_graph_node,
                        block_mask_proxy,
                        inp_scale,
                        inp_kernel_options,
                        proxied_score_mod_other_buffers + fw_graph_lifted_args,
                        proxied_mask_mod_other_buffers + mask_fn_lifted_args,
                    ),
                    kwargs={},
                ),
                example_value=None,
            )

    def _call_function_fallback(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..builder import wrap_fx_proxy

        p_args, p_kwargs = None, None
        try:
            p_args = tuple(self.to_proxy(tx, arg) for arg in args)
            p_kwargs = {key: self.to_proxy(tx, arg) for key, arg in kwargs.items()}
        except (NotImplementedError, Unsupported) as err:
            unimplemented(
                gb_type="failed to handle argument for FlexAttentionBackward HOP",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation="Missing Dynamo support for FlexAttentionBackward HOP argument.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
                from_exc=err,
            )
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


class FlexAttentionHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.flex_attention"

    @staticmethod
    def normalize_to_args(
        args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> list[VariableTracker]:
        # input signature is (query, key, value, score_mod, block_mask, *other_buffers),
        # block_mask is a tuple, and we don't want to flatten it.
        # only flatten kwargs into lists
        flat_kwargs = pytree.tree_flatten(kwargs)[0]

        # Combine the flattened lists
        all_args = args + flat_kwargs
        return all_args

    def create_wrapped_node(
        self,
        tx: "InstructionTranslator",
        query: VariableTracker,
        fn: VariableTracker,
        fn_name: str,
    ) -> tuple[Proxy, tuple[Proxy, ...]]:
        from ..._trace_wrapped_higher_order_op import TransformGetItemToIndex

        def create_scalar() -> VariableTracker:
            return query.call_method(
                tx,
                "new_empty",
                [
                    VariableTracker.build(tx, []),
                ],
                {
                    "dtype": VariableTracker.build(tx, torch.int32),
                },
            )

        with discard_graph_changes(tx):
            bhmn = [create_scalar() for _ in range(4)]
            if fn_name == "score_mod":
                scores_require_grad: bool = query.requires_grad  # type: ignore[attr-defined]
                score = query.call_method(
                    tx,
                    "new_empty",
                    [
                        VariableTracker.build(tx, []),
                    ],
                    {"requires_grad": VariableTracker.build(tx, scores_require_grad)},
                )
                new_args = [score, *bhmn]
            else:
                assert fn_name == "mask_fn", "Illegal function name: " + fn_name
                new_args = [*bhmn]

        with TransformGetItemToIndex():
            (
                (_body_output, _body_spec),
                body_graph,
                body_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                fn,
                new_args,
                {},  # expect only args no kwargs for now
                description=f"{self._HOP_NAME}: {fn_name}",
                source_target=self.value,
                set_subgraph_inputs="flatten_manual",
            )

        body_name = tx.output.install_subgraph(
            fn_name,
            torch.fx.GraphModule(tx.output.nn_modules, body_graph),
        )

        body_node = make_attr(tx, body_name)

        # It is possible that the score-mod function captures some free variables that are not
        # passed in as arguments. In this case, we need to lift them, which is handled by speculate_subgraph.
        # We then need to create proxies for this + the inputs.

        lifted_args = tuple(arg for arg in body_lifted_freevars)

        proxy_args = (body_node, lifted_args)

        return proxy_args

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..builder import wrap_fx_proxy

        (
            query,
            key,
            value,
            score_mod,
            block_mask,
            scale,
            kernel_options,
        ) = self.normalize_to_args(list(args), kwargs)

        score_mod_node, score_mod_lifted_args = self.create_wrapped_node(
            tx, query, score_mod, "score_mod"
        )
        mask_fn = block_mask.items[-1]  # type: ignore[attr-defined]
        if mask_fn.is_python_constant() and mask_fn.as_python_constant() is None:
            mask_fn = VariableTracker.build(
                tx,
                torch.nn.attention.flex_attention.noop_mask,
                source=mask_fn.source,
            )
        mask_fn_node, mask_fn_lifted_args = self.create_wrapped_node(
            tx, query, mask_fn, "mask_fn"
        )

        proxied_args = [
            query,
            key,
            value,
            TupleVariable(block_mask.items[:-1], source=block_mask.source),  # type: ignore[attr-defined]
            scale,
            kernel_options,
        ]

        # Store the invocation as a call
        # Norm_kwargs contains the score_function and we dont want to proxy this because
        # Proxying user defined functions is not supported.
        inp_args, _ = proxy_args_kwargs(proxied_args, {})

        # Compose the ordered HOO args:
        # - inp_args: [query, key, value, block_mask, scale, kernel_options]
        # - subgraph node: [score_mod, mask_fn_node]
        # - lifted args from tracing subgraph: [score_mod_other_buffers, mask_fn_other_buffers]
        _, _, _, inp_arg_block_mask, inp_arg_scale, inp_arg_kernel_options = inp_args
        block_mask = tuple(inp_arg_block_mask + (mask_fn_node,))
        with torch.fx.experimental.proxy_tensor.set_original_aten_op(self.value):
            proxy = wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    self.value,
                    args=inp_args[:3]
                    + (
                        score_mod_node,
                        block_mask,
                        inp_arg_scale,
                        inp_arg_kernel_options,
                        score_mod_lifted_args,
                        mask_fn_lifted_args,
                    ),
                    kwargs={},
                ),
                example_value=None,
            )
        return proxy
