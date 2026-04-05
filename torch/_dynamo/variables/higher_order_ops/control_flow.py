from typing import TYPE_CHECKING

from .common import (
    _call_function_and_unflatten_output,
    _call_while_loop,
    _check_all_tensorvariable,
    _check_supported_callable_arg,
    _make_inlined,
    _merge_graph_inputs,
    check_meta_consistency_vt,
    ConstantVariable,
    discard_graph_changes,
    functools,
    get_fake_value,
    graph_break_hints,
    itertools,
    LazyVariableTracker,
    ListVariable,
    make_attr,
    only_consist_of,
    OutputSpec,
    Proxy,
    pytree,
    Sequence,
    speculate_subgraph,
    SymNodeVariable,
    TensorVariable,
    torch,
    TorchHigherOrderOperatorVariable,
    TupleVariable,
    unimplemented,
    variables,
    VariableTracker,
    warnings,
)


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


class CondHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.cond"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .. import ListVariable

        self.supports_input_mutation = not torch.is_grad_enabled()
        self.supports_aliasing = not torch.is_grad_enabled()

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        for i, k in enumerate(["pred", "true_fn", "false_fn", "operands"]):
            if v := kwargs.pop(k, None):
                assert i == len(args), (
                    "did not provide the right number of non-keyword args"
                )
                args.append(v)

        # TODO(voz): Support fake tensor dispatch for recursive
        # ops - see torch/dispatch/_dispatcher.py
        if len(args) != 4 or kwargs:
            unimplemented(
                gb_type="torch.cond: improper args/kwargs",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"torch.cond expects 4 positional arguments (got {len(args)}) "
                f"and no keyword arguments (got {len(kwargs)}) "
                "Usage: cond(pred, cond_fn, body_fn, operands)",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # Specialize into one of the branches since pred is constant
        pred, true_fn, false_fn, operands = args
        if type(args[0]) is ConstantVariable:
            warnings.warn(
                "Pred is a Python constant. When used with torch.cond, it specializes on one of the branches."
                " If you want torch.cond to preserve two branches, please make the predicate a boolean tensor or a SymBool.",
                UserWarning,
            )
            if pred.as_python_constant():
                return true_fn.call_function(tx, operands.unpack_var_sequence(tx), {})
            else:
                return false_fn.call_function(tx, operands.unpack_var_sequence(tx), {})

        # predicate
        if type(pred.realize()) not in (
            ConstantVariable,
            TensorVariable,
            SymNodeVariable,
        ):
            unimplemented(
                gb_type="torch.cond: improper predicate",
                context=str(pred),
                explanation="Expected `pred` to be a bool or a boolean tensor with a single item "
                f"but got {str(type(pred))} with original python type {str(pred.python_type())}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # operands
        if not isinstance(operands, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.cond: improper operands",
                context=str(operands),
                explanation="Expected `operands` to be a list/tuple "
                f"but got {operands.python_type()}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        operands_seq = operands.unpack_var_sequence(tx)
        if not only_consist_of(
            operands, (TensorVariable, ConstantVariable, SymNodeVariable)
        ):
            unimplemented(
                gb_type="torch.cond: improper operands contents",
                context=str(operands),
                explanation="Expected `operands` to be a list/tuple of pytrees that only consists of tensor leaves.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # branches
        _check_supported_callable_arg(tx, true_fn, "true_fn")
        _check_supported_callable_arg(tx, false_fn, "false_fn")

        # Our strategy for tracing the true/false branches of cond
        # are to checkpoint our graphstate, run the true branch,
        # roll it back to the checkpoint, and run the false
        # branch, and then merge the graphstates.  Well, perhaps
        # "merge" is too strong a word: we mostly assert that
        # the resulting graphstates have to be the same.
        #
        # We only permit guards to diverge (we union the guards from
        # both branches).  In particular, this means that side
        # effects are NOT permitted inside true/false branches; this
        # would be difficult to implement, because of the path
        # explosion problem.

        def speculate_branch(
            branch: bool,
        ) -> tuple[VariableTracker, OutputSpec, torch.fx.Graph, dict[Proxy, Proxy]]:
            # NB: 0 is predicate
            ix = 1 if branch else 2
            assert self._HOP_NAME is not None
            # TODO: Support kwargs
            (
                (ret_val, ret_spec),
                ret_graph,
                ret_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                args[ix],
                operands_seq,
                {},
                self._HOP_NAME,
                source_target=self.value,
                should_flatten_outputs=True,
                # TODO - removing consts from control flow ops need more work
                remove_consts_from_outputs=False,
                supports_input_mutation=self.supports_input_mutation,
                supports_aliasing=self.supports_aliasing,
            )

            # need to ensure we increase epoch so we don't memoize unbacked bindings
            # across different subgraphs which can interfere with runtime assertion
            # generation.
            assert tx.fake_mode is not None
            tx.fake_mode.epoch += 1

            if not only_consist_of(ret_val, (TensorVariable, ConstantVariable)):
                unimplemented(
                    gb_type="torch.cond: unsupported branch return type",
                    context=str(ret_val),
                    explanation="Expected branches to return a possibly nested pytree of tensors or constant ints.",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            for ret in ret_val.unpack_var_sequence(tx):
                if ret.is_python_constant() and not isinstance(
                    ret.as_python_constant(), int
                ):
                    unimplemented(
                        gb_type="torch.cond: unsupported branch return type (constant non-int)",
                        context=str(ret_val),
                        explanation="Constants returned from branches must be ints.",
                        hints=[
                            *graph_break_hints.USER_ERROR,
                        ],
                    )
            return ret_val, ret_spec, ret_graph, ret_lifted_freevars

        (true_r, true_spec, true_graph, true_lifted_freevars) = speculate_branch(True)
        true_nn_modules = dict(tx.output.nn_modules)

        (
            false_r,
            false_spec,
            false_graph,
            false_lifted_freevars,
        ) = speculate_branch(False)
        false_nn_modules = dict(tx.output.nn_modules)

        same_spec = _make_inlined(tx, pytree.TreeSpec.__eq__)(
            true_spec.treespec, false_spec.treespec
        ).as_python_constant()
        # 3.14: NotImplemented cannot be converted to bool
        if same_spec is not NotImplemented and not same_spec:
            unimplemented(
                gb_type="torch.cond: differing branch outputs",
                context=f"true_spec: {true_spec.treespec}, false_spec: {false_spec.treespec}, same_spec: {same_spec}",
                explanation="Expected branches to return the same pytree structure.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        (
            true_graph,
            false_graph,
            true_shared,
            _false_shared,
            unique_true,
            unique_false,
        ) = _merge_graph_inputs(
            true_graph,
            true_lifted_freevars,
            "true_branch",
            false_graph,
            false_lifted_freevars,
            "false_branch",
        )

        true_name = tx.output.install_subgraph(
            "cond_true",
            torch.fx.GraphModule(true_nn_modules, true_graph),
        )
        false_name = tx.output.install_subgraph(
            "cond_false",
            torch.fx.GraphModule(false_nn_modules, false_graph),
        )

        true_node = make_attr(tx, true_name)
        false_node = make_attr(tx, false_name)

        p_args = (
            pred.as_proxy(),
            true_node,
            false_node,
            # We pick true_shared but it shouldn't matter
            tuple(true_shared + unique_true + unique_false),
        )

        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.cond,
            p_args,
            {},
            None,
            true_spec,
            true_r,
        )


class WhileLoopHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.while_loop"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        assert self._HOP_NAME is not None
        return _call_while_loop(
            self, tx, args, kwargs, stack_output=False, hop_name=self._HOP_NAME
        )


class WhileLoopStackOutputHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.while_loop"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        assert self._HOP_NAME is not None
        return _call_while_loop(
            self, tx, args, kwargs, stack_output=True, hop_name=self._HOP_NAME
        )


class AssociativeScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.associative_scan"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from torch._higher_order_ops.utils import first_slice_copy

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        def arg_extractor(
            combine_fn: VariableTracker,
            xs: VariableTracker,
            additional_inputs: VariableTracker,
        ) -> tuple[VariableTracker, VariableTracker, VariableTracker]:
            return combine_fn, xs, additional_inputs

        combine_fn, xs, additional_inputs = arg_extractor(*args, **kwargs)

        if args[0].python_type() is functools.partial:
            # This is the standard case when the user calls the frontend
            # and the frontend invokes dynamo
            if len(args) != 2:
                unimplemented(
                    gb_type="torch.associative_scan: improper args",
                    context=f"args: {args}",
                    explanation=f"torch.associative_scan expects 2 positional arguments (got {len(args)}) "
                    "Usage: associative_scan(combine_fn, xs)",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )

            xs_treespec = args[0].keywords["spec"]

            # combine_fn input check
            # We need to get the pure combine_fn from the functools.partial
            _check_supported_callable_arg(
                tx,
                combine_fn.keywords["combine_fn"],  # type: ignore[attr-defined]
                "combine_fn",
            )
        else:
            # This case is hit during re-tracing, for example in export tests
            # In this case, the combine_fn is a callable and not a functools.partial
            xs_treespec = _make_inlined(tx, pytree.tree_structure)(xs)

            _check_supported_callable_arg(tx, combine_fn, "combine_fn")

        # xs input check
        if not isinstance(xs, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.associative_scan: improper xs",
                context=str(xs),
                explanation=f"Expected xs to be a list/tuple but got {xs.python_type()}",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )
        xs_vars = xs.unpack_var_sequence(tx)
        _check_all_tensorvariable(xs_vars)

        # additional_inputs input check
        if not isinstance(additional_inputs, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.associative_scan: improper additional_inputs",
                context=str(additional_inputs),
                explanation=f"Expected additional_inputs to be a list/tuple but got {additional_inputs.python_type()}",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )
        additional_inputs_vars = additional_inputs.unpack_var_sequence(tx)
        _check_all_tensorvariable(additional_inputs_vars)

        scan_length = get_fake_value(xs_vars[0].as_proxy().node, tx).size()[0]
        if scan_length == 0:
            unimplemented(
                gb_type="torch.associative_scan: zero-sized tensor",
                context=str(xs_vars[0]),
                explanation="associative_scan() operator doesn't support zero-sized tensors during tracing.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # Trace the subgraph
        # The sub_args is a slice of original input, e.g. if input.size is (3, 4), and scan dim=0
        # the sub_args shape will be (4, ).
        with discard_graph_changes(tx):
            sub_args = [
                _make_inlined(tx, first_slice_copy)(leaf)
                for leaf in itertools.chain(xs_vars, xs_vars)
            ]
            sub_args_additional_inputs = [
                t.call_method(tx, "clone", args=[], kwargs={})
                for t in additional_inputs_vars
            ]

        sub_args = sub_args + sub_args_additional_inputs
        assert self._HOP_NAME is not None
        (
            (combine_result, _combine_spec),
            combine_graph,
            combine_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            combine_fn,
            sub_args,
            sub_kwargs={},
            description=self._HOP_NAME,
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
            supports_input_mutation=self.supports_input_mutation,
            supports_aliasing=self.supports_aliasing,
        )

        # Ensure that the output of scan is a flattened list of elements,
        # because downstream operations assume that the output of HOPs
        # is flattened
        output_node = combine_graph.find_nodes(op="output")[0]
        output_node.args = (pytree.tree_leaves(output_node.args),)
        combine_graph.lint()

        # Collect the results from the combine_fn
        results, _combine_treespec = _make_inlined(tx, pytree.tree_flatten)(
            combine_result
        ).unpack_var_sequence(tx)

        # Check whether the combine_fn returns one child tree for the output.
        if _combine_treespec.as_python_constant().num_leaves < 1:
            unimplemented(
                gb_type="torch.associative_scan: combine_fn improper number of leaves",
                context=str(_combine_treespec.as_python_constant()),
                explanation="combine_fn needs to produce one pytree for the output "
                f"but combine_fn produces the pytree {_combine_treespec.as_python_constant()}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # Check whether the outs produced by combine_fn has the same treespec as xs
        # We need to have this check this way, because in case init is a TreeSpec and carry
        # but carry is only a LeafSpec, these two cannot be compared correctly.
        if (
            xs_treespec.as_python_constant().is_leaf()
            != _combine_treespec.as_python_constant().is_leaf()
        ) or not _make_inlined(tx, pytree.TreeSpec.__eq__)(
            xs_treespec, _combine_treespec
        ).as_python_constant():
            unimplemented(
                gb_type="torch.associative_scan: mismatched input/output tree structure",
                context=f"xs: {xs_treespec.as_python_constant()}, output: {_combine_treespec.as_python_constant()}",
                explanation="The tree structure of the xs and the outs of the combine_fn are are expected to be identical, but got "
                f"xs: {xs_treespec.as_python_constant()} vs output: {_combine_treespec.as_python_constant()}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # We set include contiguity=False because we have vmap x HOP tests, where if
        # include_contiguity=True will call t.is_contiguous inside of vmap and get an error
        # "querying is_contiguous inside of vmap for memory_format other than
        # torch.contiguous_format is not yet implemented". This is okay because stride
        # is still checked.
        check_meta_consistency_vt(
            [_make_inlined(tx, first_slice_copy)(t) for t in xs_vars],
            results.items,  # type: ignore[attr-defined]
            "initial_xs",
            "combine_fn_output",
            include_contiguity=False,
        )

        combine_gm = torch.fx.GraphModule(dict(tx.output.nn_modules), combine_graph)
        combine_freevars_proxy = tuple(combine_lifted_freevars.keys())

        # Compute the proxies for the input check
        proxy_vars_inputcheck = (
            tuple(sarg.as_proxy() for sarg in sub_args) + combine_freevars_proxy
        )

        from torch._higher_order_ops.utils import _maybe_fake_tracing
        from torch._inductor.utils import is_pointwise_use

        assert tx.fake_mode is not None
        with tx.fake_mode:
            sub_args_fake = [
                (
                    leaf.node.meta["example_value"].clone()
                    if hasattr(leaf.node.meta["example_value"], "clone")
                    else leaf.node.meta["example_value"]
                )
                for leaf in pytree.tree_leaves(proxy_vars_inputcheck)
            ]
            pre_dispatch = False

            fx = _maybe_fake_tracing(
                combine_gm, sub_args_fake, pre_dispatch=pre_dispatch
            )

            for node in fx.graph.nodes:
                # Check that the combine_fn is pointwise, if combine_mode='pointwise'
                if not all(
                    is_pointwise_use(use) or use.op == "output" for use in node.users
                ):
                    raise RuntimeError(
                        "For combine_mode='pointwise', the combine_fn needs to be pointwise"
                    )

        combine_fn_name = tx.output.install_subgraph(
            "associative_scan_combine_fn", combine_gm
        )

        # Compute the proxies
        xs_proxy = xs.as_proxy()
        combine_freevars_proxy = tuple(combine_lifted_freevars.keys())
        additional_inputs_proxy = additional_inputs.as_proxy() + combine_freevars_proxy

        p_args = (
            make_attr(tx, combine_fn_name),
            xs_proxy,
            additional_inputs_proxy,
        )

        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.associative_scan,
            p_args,
            {},
            None,
            OutputSpec(xs_treespec),  # type: ignore[arg-type]
            None,
        )


class ScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.scan"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from torch._higher_order_ops.scan import _extract_carry_and_out
        from torch._higher_order_ops.utils import first_slice_copy

        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        # combine_fn input check
        def _check_combine_fn_is_normalized(combine_fn_var: VariableTracker) -> bool:
            if not isinstance(
                combine_fn_var,
                (
                    variables.nn_module.NNModuleVariable,
                    variables.nn_module.UnspecializedNNModuleVariable,
                    variables.FunctoolsPartialVariable,
                ),
            ):
                unimplemented(
                    gb_type="torch.scan: improper combine_fn",
                    context=str(combine_fn_var),
                    explanation="Expected combine_fn to be wrapped as functools.partial in scan user-facing api "
                    f"or a graph module if we're re-exporting but got {combine_fn_var.python_type()}.",
                    hints=[
                        *graph_break_hints.DIFFICULT,
                    ],
                )
            return isinstance(
                combine_fn_var,
                (
                    variables.nn_module.NNModuleVariable,
                    variables.nn_module.UnspecializedNNModuleVariable,
                ),
            )

        def arg_extractor(
            combine_fn: VariableTracker,
            init: VariableTracker,
            xs: VariableTracker,
            additional_inputs: VariableTracker,
        ) -> tuple[VariableTracker, VariableTracker, VariableTracker, VariableTracker]:
            return combine_fn, init, xs, additional_inputs

        combine_fn, init, xs, additional_inputs = arg_extractor(*args, **kwargs)
        init_vars = init.unpack_var_sequence(tx)
        xs_vars = xs.unpack_var_sequence(tx)
        additional_inputs_vars = additional_inputs.unpack_var_sequence(tx)

        # combine_fn input check
        combine_fn_is_normalized = _check_combine_fn_is_normalized(combine_fn)
        if combine_fn_is_normalized:
            combine_gm = combine_fn.value  # type: ignore[attr-defined]
            assert isinstance(combine_gm, torch.fx.GraphModule), (
                combine_fn,
                combine_gm,
            )
        else:
            # combine_fn input check
            # We need to get the pure combine_fn from the functools.partial
            _check_supported_callable_arg(
                tx,
                combine_fn.keywords["combine_fn"],  # type: ignore[attr-defined]
                "combine_fn",
            )
        # xs input check
        if not isinstance(xs, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.scan: improper xs",
                context=str(xs),
                explanation=f"Expected xs to be a list/tuple but got {xs.python_type()}",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )
        # init input check
        if not isinstance(init, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.scan: improper init",
                context=str(init),
                explanation=f"Expected init to be a list/tuple with at least one element but got {init.python_type()}",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        if len(init_vars) == 0:
            unimplemented(
                gb_type="torch.scan: no init leaves",
                context="",
                explanation="Expected init leaves.",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )

        # additional_inputs input check
        if not isinstance(additional_inputs, (ListVariable, TupleVariable)):
            unimplemented(
                gb_type="torch.scan: improper additional_inputs",
                context=str(additional_inputs),
                explanation=f"Expected additional_inputs to be a list/tuple but got {additional_inputs.python_type()}",
                hints=[
                    *graph_break_hints.DYNAMO_BUG,
                ],
            )
        # scan_length check
        scan_length = get_fake_value(xs_vars[0].as_proxy().node, tx).size()[0]
        if scan_length == 0:
            unimplemented(
                gb_type="torch.scan: zero-sized tensor",
                context=str(xs_vars[0]),
                explanation="associative_scan() operator doesn't support zero-sized tensors during tracing.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                    *graph_break_hints.SUPPORTABLE,
                ],
            )
        _check_all_tensorvariable(init_vars)
        _check_all_tensorvariable(xs_vars)
        _check_all_tensorvariable(additional_inputs_vars)

        with discard_graph_changes(tx):
            sub_args_init = [
                ini.call_method(tx, "clone", args=[], kwargs={}) for ini in init_vars
            ]
            # The sub_args_inp is a slice of original input, e.g. if input.size is (3, 4), and scan dim=0
            # the sub_args_inp shape will be (4, ).
            sub_args_inp = [_make_inlined(tx, first_slice_copy)(inp) for inp in xs_vars]
            sub_args_additional_inputs = [
                t.call_method(tx, "clone", args=[], kwargs={})
                for t in additional_inputs_vars
            ]

        sub_args = sub_args_init + sub_args_inp + sub_args_additional_inputs
        assert self._HOP_NAME is not None
        (
            (combine_result, _combine_spec),
            combine_graph,
            combine_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            combine_fn,
            sub_args,
            sub_kwargs={},
            description=self._HOP_NAME,
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
            supports_input_mutation=self.supports_input_mutation,
            supports_aliasing=self.supports_aliasing,
        )

        # Ensure that the output of scan is a flattened list of elements,
        # because downstream operations assume that the output of HOPs
        # is flattened
        output_node = combine_graph.find_nodes(op="output")[0]
        output_node.args = (pytree.tree_leaves(output_node.args),)
        combine_graph.lint()
        combine_freevars_proxy = list(combine_lifted_freevars.keys())
        combine_result_vars = combine_result.unpack_var_sequence(tx)

        if combine_fn_is_normalized:
            carry_vars, out_vars = _extract_carry_and_out(
                combine_result_vars, len(init_vars)
            )
        else:
            if len(combine_result_vars) != 2:
                unimplemented(
                    gb_type="torch.scan: improper combine_fn number of returns",
                    context=str(combine_result_vars),
                    explanation=f"Expect combine_fn to return a tuple (next_carry, y) but got {combine_result_vars}.",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            carry_tree, out_vars = combine_result_vars
            carry_vars, _ = _make_inlined(tx, pytree.tree_flatten)(
                carry_tree
            ).unpack_var_sequence(tx)
            carry_vars = carry_vars.unpack_var_sequence(tx)
            out_vars = _make_inlined(tx, pytree.tree_leaves)(
                out_vars
            ).unpack_var_sequence(tx)

            # additional output checking
            _combine_spec = OutputSpec(
                _make_inlined(tx, pytree.tree_structure)(combine_result)  # type: ignore[arg-type]
            )

            check_meta_consistency_vt(
                init_vars,
                carry_vars,
                "init",
                "carry",
            )

        # Check meta data of carries and inits. If we pass this stage, we are sure that the init and carries
        # have the same tree structure.
        # We set include contiguity=False because we have vmap x HOP tests, where if
        # include_contiguity=True will call t.is_contiguous inside of vmap and get an error
        # "querying is_contiguous inside of vmap for memory_format other than
        # torch.contiguous_format is not yet implemented". This is okay because stride
        # is still checked.
        check_meta_consistency_vt(
            init_vars,
            carry_vars,
            "init",
            "carry",
            include_contiguity=False,
        )

        xs_proxy = xs.as_proxy()
        init_proxy = init.as_proxy()
        additional_inputs_proxy = list(additional_inputs.as_proxy()) + list(
            combine_freevars_proxy
        )

        combine_gm = torch.fx.GraphModule(dict(tx.output.nn_modules), combine_graph)
        combine_fn_name = tx.output.install_subgraph("scan_combine_fn", combine_gm)

        p_args = (
            make_attr(tx, combine_fn_name),
            init_proxy,
            xs_proxy,
            additional_inputs_proxy,
        )

        return _call_function_and_unflatten_output(
            tx,
            torch.ops.higher_order.scan,
            p_args,
            {},
            None,
            _combine_spec,
            None,
        )


class MapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    _HOP_NAME = "torch.ops.higher_order.map_impl"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = False
    supports_aliasing = False

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        if len(kwargs) > 0:
            unimplemented(
                gb_type="torch.map: kwargs not supported",
                context=f"args: {args}, kwargs: {kwargs}",
                explanation=f"torch.map expects no keyword arguments (got {len(kwargs)})",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        _check_supported_callable_arg(tx, args[0], "map_fn")

        # args = f, flat_xs, flat_args
        assert isinstance(args[1], (ListVariable, TupleVariable)), args[1]
        assert isinstance(args[2], (ListVariable, TupleVariable)), args[2]
        unpacked_xs = args[1].unpack_var_sequence(tx)
        unpacked_args = args[2].unpack_var_sequence(tx)

        sample_shape = get_fake_value(unpacked_xs[0].as_proxy().node, tx).size()

        if len(sample_shape) < 1 or sample_shape[0] == 0:
            unimplemented(
                gb_type="torch.map: improper inputs",
                context=str(sample_shape),
                explanation="torch.map doesn't support scalar or non-zero sized tensors during tracing.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # To get the example output from map() we will need to provide at least one sample to
        # the loop body. In our case we will always use xs[0], and our map() won't support zero
        # sized tensor during tracing.
        with discard_graph_changes(tx):
            sliced_xs = [
                xs.call_method(
                    tx,
                    "select",
                    args=[VariableTracker.build(tx, 0), VariableTracker.build(tx, 0)],
                    kwargs={},
                )
                for xs in unpacked_xs
            ]
        assert self._HOP_NAME is not None

        # TODO: Support kwargs
        (
            (body_r, body_spec),
            body_graph,
            body_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            args[0],
            [
                *sliced_xs,
                *unpacked_args,
            ],
            {},
            self._HOP_NAME,
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
            should_flatten_outputs=True,
            # TODO - removing consts from control flow ops need more work
            remove_consts_from_outputs=False,
            supports_input_mutation=self.supports_input_mutation,
            supports_aliasing=self.supports_aliasing,
        )

        # Check all outputs of map are tensors.
        # For map, outputting None is OK, thus ignore None values in the check
        body_r_vars = body_r.unpack_var_sequence(tx)
        none_mask = [x.is_constant_none() for x in body_r_vars]
        _check_all_tensorvariable(
            [br for bm, br in zip(none_mask, body_r_vars) if not bm]
        )

        body_nn_modules = dict(tx.output.nn_modules)

        body_name = tx.output.install_subgraph(
            "map_body",
            torch.fx.GraphModule(body_nn_modules, body_graph),
        )

        body_node = make_attr(tx, body_name)

        p_args = (
            body_node,
            [xs.as_proxy() for xs in unpacked_xs],
            [arg.as_proxy() for arg in unpacked_args]
            + list(body_lifted_freevars.keys()),
        )

        return _call_function_and_unflatten_output(
            tx, torch.ops.higher_order.map_impl, p_args, {}, None, body_spec, body_r
        )
