import contextlib
import functools
import logging

from typing import Dict, List, Optional

import torch._C
import torch.fx
import torch.nn
import torch.onnx.operators
from torch._dynamo.utils import deepcopy_to_fake_tensor, get_fake_value, get_real_value
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builtin import BuiltinVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.tensor import SymNodeVariable
from torch._guards import Source
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree

from ..exc import (
    UncapturedHigherOrderOpError,
    unimplemented,
    Unsupported,
    UserError,
    UserErrorType,
)
from ..source import FSDPNNModuleSource, GetItemSource, NNModuleSource
from ..utils import proxy_args_kwargs
from .dicts import ConstDictVariable
from .lists import ListVariable, TupleVariable
from .nn_module import NNModuleVariable, UnspecializedNNModuleVariable


log = logging.getLogger(__name__)


def raise_hard_error_if_graph_break(reason):
    def deco(fn):
        @functools.wraps(fn)
        def graph_break_as_hard_error(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Unsupported as e:
                msg = " Scroll up to find out what causes the graph break."
                raise UncapturedHigherOrderOpError(reason + msg) from e

        return graph_break_as_hard_error

    return deco


@contextlib.contextmanager
def dynamo_enable_grad(tx, enable=True):
    from . import GradModeVariable

    org_value = torch.is_grad_enabled()
    try:
        GradModeVariable.create(tx, enable, initialized=True)
        yield
    finally:
        GradModeVariable.create(tx, org_value, initialized=True)


def only_consist_of(var, types):
    if isinstance(var, types):
        return True
    if isinstance(var, (TupleVariable, ListVariable)):
        return all(only_consist_of(item, types) for item in var.items)
    if isinstance(var, ConstDictVariable):
        return all(only_consist_of(item, types) for item in var.items.values())
    return False


# A more read-able syntax sugar for creating a UserFunctionVariable for f
# and run call_function on it. Make it return a function to preserve the calling
# convention of the original f.
def _make_inlined(tx, f):
    assert callable(f), "Expect f to be a python callable."

    def inline_call(*args, **kwargs):
        return UserFunctionVariable(f).call_function(tx, args, kwargs)

    return inline_call


def _call_function_and_unflatten_output(tx, fn, args, kwargs, ret_vt, ret_treespec):
    from .builder import wrap_fx_proxy

    flat_example_value = pytree.tree_map_only(
        torch.fx.Proxy,
        lambda a: a.node.meta["example_value"],
        ret_vt.as_proxy(),
    )

    # Store the invocation as a call
    flat_variable = wrap_fx_proxy(
        tx=tx,
        proxy=tx.output.create_proxy(
            "call_function",
            fn,
            args=args,
            kwargs=kwargs,
        ),
        example_value=flat_example_value,
    )

    # Transform variable back into a list (previously made into a tuple by
    # speculate_subgraph function) so as to respect the pytree API typing.
    flat_list_variable = BuiltinVariable(list).call_function(tx, [flat_variable], {})
    return (
        _make_inlined(tx, pytree.tree_unflatten)(flat_list_variable, ret_treespec)
        if ret_treespec
        else flat_variable
    )


def _assert_tensors_nonaliasing(inputs, outputs):
    input_tensor_ids = {
        id(t) for t in pytree.tree_leaves(inputs) if isinstance(t, torch.Tensor)
    }
    output_tensor_ids = {
        id(t) for t in pytree.tree_leaves(outputs) if isinstance(t, torch.Tensor)
    }
    assert input_tensor_ids.isdisjoint(
        output_tensor_ids
    ), "inputs to function body cannot alias outputs"


def validate_args_and_maybe_create_graph_inputs(
    sub_args,
    tracer,
    tx,
    manually_set_subgraph_inputs,
    description,
):
    from . import AutogradFunctionContextVariable, ConstantVariable, EnumVariable
    from .builder import wrap_fx_proxy_cls

    assert tracer.parent is not None

    args = []
    for a in sub_args:
        assert isinstance(a, VariableTracker)
        if not manually_set_subgraph_inputs:
            args.append(a)
            continue

        if isinstance(a, (ConstantVariable, EnumVariable)):
            # This arg is not used in the body of the higher order op.
            # Currently, this new input is added to make the calls
            # happy, which expect a fixed number of arguments. In
            # future, we can clean this up.
            tracer.create_graph_input("const")
            new_arg = a
        # Weird special case, we probably want to delete it or fold it
        # into the next case (of `a` being placeable into a graph)
        elif isinstance(a, AutogradFunctionContextVariable):
            tracer.create_graph_input(a.as_proxy().node.name)
            new_arg = a
        # If `a` can be put into a graph
        elif a.maybe_fx_node() is not None:
            node = a.maybe_fx_node()
            new_proxy = tracer.create_graph_input(node.name)
            example_value = (
                node.meta["example_value"] if "example_value" in node.meta else None
            )
            new_arg = wrap_fx_proxy_cls(
                target_cls=type(a),
                tx=tx,
                proxy=new_proxy,
                example_value=example_value,
            )
        # If `a` cannot be put into a graph
        else:
            # HOPs work much better if they use speculate_subgraph(manually_set_subgraph_inputs=False).
            raise unimplemented(
                f"{description} with body that accepts non-Tensors as input. "
                f"Got: {a.python_type()}"
            )
        args.append(new_arg)
    return args


# See NOTE [HigherOrderOperator tracing design] for details of the design
def speculate_subgraph(
    tx,
    f,
    sub_args,
    sub_kwargs,
    description,
    *,
    # source_target is the .value of HigherOrderOpVariable and is the
    # target of the proxy that we created for the higherOrderOperator.
    source_target=None,
    always_restore=False,
    enable_grad=None,
    # NOTE [Temporary argument `manually_set_subgraph_inputs`]
    # If manually_set_subgraph_inputs=True, then we manually add
    # the `sub_args` to `subgraph`, if False then we rely
    # on tracer's lifting mechanism to lift these args.
    # NOTE: Default `True` is temporary and plan is
    #       to always lift args in future and remove this
    #       argument.
    manually_set_subgraph_inputs=True,
    restore_side_effects=True,
    should_flatten_outputs=False,
    # Pass in an originating tracer - this is needed for preserving context
    # across fwd-bwd for autograd.Function
    tracer=None,
):
    if sub_kwargs is None:
        sub_kwargs = {}

    # See NOTE [Temporary argument `manually_set_subgraph_inputs`]
    if sub_kwargs and manually_set_subgraph_inputs:
        unimplemented(
            "Use `manually_set_subgraph_inputs=False` when passing `sub_kwargs`."
        )

    try:
        f, sub_args, sub_kwargs = VariableTracker.apply(
            # ensure guards on args get installed in parent subgraph
            lambda x: x.realize(),
            (f, sub_args, sub_kwargs),
        )
        with tx.output.subtracer(source_target, tracer) as subtracer:
            args = validate_args_and_maybe_create_graph_inputs(
                sub_args, subtracer, tx, manually_set_subgraph_inputs, description
            )

            validate_args_and_maybe_create_graph_inputs(
                sub_kwargs.values(),
                subtracer,
                tx,
                manually_set_subgraph_inputs=False,
                description=description,
            )

            autograd_ctx = (
                dynamo_enable_grad(tx, enable_grad)
                if enable_grad is not None
                else contextlib.nullcontext()
            )

            if restore_side_effects:
                prev_side_effects = tx.output.side_effects.clone()

            with autograd_ctx:
                output = f.call_function(tx, args, sub_kwargs)

            if restore_side_effects:
                # Captured variables are tracked in side-effects
                # and they show up in output graph incorrectly.
                # It is ok to undo this side-effect tracking
                # as speculate_subgraph will allow only
                # pure functions.
                tx.output.side_effects = prev_side_effects

            treespec = None
            if should_flatten_outputs:
                # Flatten the speculated subgraph output.
                output, treespec = _make_inlined(tx, pytree.tree_flatten)(
                    output
                ).unpack_var_sequence(tx)
                # Actually, transform the list (returned by flatten) into a tuple
                # for dynamo consistency.
                output = BuiltinVariable(tuple).call_function(tx, [output], {})

            # Register output to graph
            # Modeled off of compile_and_call_fx_graph
            # TODO: support pytree output
            # We check always_restore because we dont use the output or side effects of always_restore code,
            # like bwd.
            if always_restore:
                # Nothing left to do here
                return (output, treespec), tx.output.graph, subtracer.lifted_freevars
            else:
                from . import TensorVariable

                if not only_consist_of(output, TensorVariable):
                    unimplemented(
                        "HigherOrderOperator body's output must consist of tensors only"
                    )

                # The output proxies might not belong to this SubgraphTracer
                # (if they are free variables that were never lifted)
                # so lift them here.
                output_proxies = output.as_proxy()
                output_proxies = pytree.tree_map(
                    subtracer.maybe_lift_tracked_freevar_to_input, output_proxies
                )

                tx.output.create_node(
                    "output",
                    "output",
                    (subtracer.create_arg((output_proxies,))),
                    {},
                )
                graph = tx.output.graph
                graph.lint()
                lifted_freevars = subtracer.lifted_freevars

                return (
                    (output, treespec),
                    graph,
                    lifted_freevars,
                )

    except Unsupported as ex:
        f_name = f"{type(f).__name__}"
        if isinstance(f, UserFunctionVariable):
            f_name = f.get_name()
        msg = (
            f"speculate_subgraph: while introspecting {description}, we were unable "
            f"to trace function `{f_name}` into a single graph. This means "
            f"that Dynamo was unable to prove safety for this API and will "
            f"fall back to eager-mode PyTorch, which could lead to a slowdown."
        )
        log.warning(msg)
        log.exception(ex)
        raise Unsupported(
            f"{msg} Scroll up for the stack trace "
            f"of the initial exception. The reason was: {ex.msg}"
        ) from ex


def make_attr(tx, name):
    node = tx.output.create_proxy(
        "get_attr",
        name,
        (),
        {},
    )
    return node


def add_subgraph(tx, source, name, gm):
    next_name = None
    i = 0
    while not next_name:
        candidate = f"{name}_{i}"
        if candidate in tx.output.nn_modules:
            i += 1
        else:
            next_name = candidate

    gm.__name__ = next_name
    if source.guard_source().is_fsdp_module():
        src = FSDPNNModuleSource(GetItemSource(source, next_name))
    else:
        src = NNModuleSource(GetItemSource(source, next_name))
    gm.torchdynamo_force_dynamic = False
    tx.output.register_attr_or_module(gm, next_name, source=src)
    return next_name


class TorchHigherOrderOperatorVariable(VariableTracker):
    def __init__(self, value, source: Optional[Source] = None, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.source = source

    @staticmethod
    def make(value, source=None, **kwargs):
        if value.__name__ == "cond":
            return CondHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ in ("map", "map_impl"):
            return MapHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "executorch_call_delegate":
            return ExecutorchCallDelegateHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "out_dtype":
            return OutDtypeHigherOrderVariable(value, source, **kwargs)
        elif value is torch._functorch.eager_transforms.grad_impl:
            return FunctorchGradHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ in (
            "trampoline_autograd_fwd",
            "trampoline_autograd_bwd",
            "trampoline_autograd_apply",
        ):
            return AutogradFunctionMethodHigherOrderVariable(
                value=value, source=source, **kwargs
            )
        elif value.__name__ == "wrap":
            return WrapHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ in (
            "wrap_activation_checkpoint",
            "tag_activation_checkpoint",
        ):
            return CheckpointHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "_export_tracepoint":
            return ExportTracepointHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "trace_wrapped":
            return TraceWrappedHigherOrderOperatorVariable(value, source, **kwargs)
        else:
            unimplemented(f"HigherOrderOperator {value.__name__}")

    def call_function(
        self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]
    ) -> VariableTracker:
        unimplemented(f"HigherOrderOperator {self.value.__name__}")


class CondHigherOrderVariable(TorchHigherOrderOperatorVariable):
    @raise_hard_error_if_graph_break(
        reason="Cond doesn't work unless it is captured completely with torch.compile."
    )
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from . import (
            ConstantVariable,
            ListVariable,
            NestedUserFunctionVariable,
            TensorVariable,
            UserFunctionVariable,
        )

        args, kwargs = VariableTracker.apply(lambda x: x.realize(), (args, kwargs))

        for i, k in enumerate(["pred", "true_fn", "false_fn", "operands"]):
            if v := kwargs.pop(k, None):
                assert i == len(
                    args
                ), "did not provide the right number of non-keyword args"
                args.append(v)

        if kwargs:
            unimplemented(f"torch.cond: Got unexpected kwargs: {list(kwargs.keys())}")

        # TODO(voz): Support fake tensor dispatch for recursive
        # ops - see torch/dispatch/_dispatcher.py
        if len(args) != 4:
            unimplemented(
                f"Expected 4 arguments but got {len(args)}.\n"
                f"Usage: cond(pred, true_fn, false_fn, operands)",
            )
        # predicate
        if type(args[0]) not in (ConstantVariable, TensorVariable, SymNodeVariable):
            unimplemented(
                f"Expected pred to be bool or a boolean tensor with single "
                f"item but got {str(type(args[0]))} "
                f"with original python type {str(args[0].python_type())}.",
            )

        # operands
        if not isinstance(args[3], (ListVariable, TupleVariable)):
            unimplemented(
                f"Expected a tuple but got {args[3].python_type()}",
            )
        operands = args[3].unpack_var_sequence(tx)
        if not only_consist_of(args[3], (TensorVariable,)):
            unimplemented(
                "Expect operands to be a tuple of pytrees that only consists of tensor leaves."
            )

        # branches
        assert isinstance(
            args[1],
            (
                UserFunctionVariable,
                NestedUserFunctionVariable,
                NNModuleVariable,
                UnspecializedNNModuleVariable,
            ),
        ), str(
            type(args[1])
        )  # true_fn

        assert isinstance(
            args[2],
            (
                UserFunctionVariable,
                NestedUserFunctionVariable,
                NNModuleVariable,
                UnspecializedNNModuleVariable,
            ),
        ), str(
            type(args[2])
        )  # false_fn

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

        def speculate_branch(branch):
            # NB: 0 is predicate
            ix = 1 if branch else 2
            # TODO: Support kwargs
            (
                (ret_val, ret_treespec),
                ret_graph,
                ret_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                args[ix],
                operands,
                {},
                "cond",
                source_target=self.value,
                manually_set_subgraph_inputs=False,
                should_flatten_outputs=True,
            )

            if not only_consist_of(ret_val, (TensorVariable,)):
                unimplemented(
                    "Expected branches to return a possibly nested list/tuple/dict of tensors but it consists of non tensors.",
                )
            return ret_val, ret_treespec, ret_graph, ret_lifted_freevars

        (true_r, true_treespec, true_graph, true_lifted_freevars) = speculate_branch(
            True
        )
        true_nn_modules = dict(tx.output.nn_modules)

        (
            false_r,
            false_treespec,
            false_graph,
            false_lifted_freevars,
        ) = speculate_branch(False)
        false_nn_modules = dict(tx.output.nn_modules)

        same_treespec = _make_inlined(tx, pytree.TreeSpec.__eq__)(
            true_treespec, false_treespec
        )
        if not same_treespec.as_python_constant():
            unimplemented("Expected branches to return the same pytree structure.")

        def diff_meta(tensor_vars1, tensor_vars2):
            assert all(
                isinstance(var, TensorVariable) for var in tensor_vars1 + tensor_vars2
            )
            all_diffs = []
            for i, (var1, var2) in enumerate(zip(tensor_vars1, tensor_vars2)):
                # We check the meta data associated with meta["example_value"]
                meta1 = _extract_tensor_metadata(
                    var1.proxy.node.meta["example_value"], include_contiguity=False
                )
                meta2 = _extract_tensor_metadata(
                    var2.proxy.node.meta["example_value"], include_contiguity=False
                )
                if meta1 != meta2:
                    all_diffs.append((f"pair{i}:", meta1, meta2))
            return all_diffs

        if diffs := diff_meta(
            true_r.unpack_var_sequence(tx), false_r.unpack_var_sequence(tx)
        ):
            unimplemented(
                f"Expected branches to return tensors with same metadata. [(tensor_pair, difference)...]:{diffs}"
            )

        def dedup_and_sort_lifted_freevars(true_lifted_freevars, false_lifted_freevars):
            # The nn module attributes are guaranteed to be registered into the top-level graph module during
            # higher order op speculation. Therefore, get_attr nodes in two branches with the same
            # target refer to the same attribute and we can safely deduplicate them with their target.
            #
            # Note: ideally, dynamo should just create a single proxy for the same attribute of a nn module. But
            # true_branch and false_branch belong to two separate tracing contexts, they may register the same
            # attribute to top level seperately. This creates two get_attr proxies for the same attribute
            # that have different meta data such as stack_trace (one stack trace for the true_branch,
            # and the other for false_branch). It seems better to discard the proxy explicitly in cond
            # than make dynamo create a single proxy for the same get_attr target.
            def shared_getattrs(true_lifted_proxies, false_lifted_proxies):
                true_targets = {
                    proxy.node.target: proxy
                    for proxy in true_lifted_proxies
                    if proxy.node.op == "get_attr"
                }
                true_fn_shared_getattrs = {}
                false_fn_shared_getattrs = {}

                for false_proxy in false_lifted_proxies:
                    if (
                        false_proxy.node.op == "get_attr"
                        and false_proxy.node.target in true_targets
                    ):
                        true_proxy = true_targets[false_proxy.node.target]
                        true_fn_shared_getattrs[true_proxy] = true_proxy
                        false_fn_shared_getattrs[false_proxy] = true_proxy
                return true_fn_shared_getattrs, false_fn_shared_getattrs

            true_fn_shared_getattrs, false_fn_shared_getattrs = shared_getattrs(
                true_lifted_freevars.keys(), false_lifted_freevars.keys()
            )

            true_shared_freevars = (
                true_lifted_freevars.keys() & false_lifted_freevars.keys()
            ).union(true_fn_shared_getattrs.keys())
            false_shared_freevars = (
                true_lifted_freevars.keys() & false_lifted_freevars.keys()
            ).union(false_fn_shared_getattrs.keys())
            unique_true_freevars = true_lifted_freevars.keys() - true_shared_freevars
            unique_false_freevars = false_lifted_freevars.keys() - false_shared_freevars

            def _sort_by_name(vars):
                return sorted(vars, key=lambda var: var.node.name)

            return (
                list(_sort_by_name(list(true_shared_freevars))),
                list(_sort_by_name(list(false_shared_freevars))),
                list(_sort_by_name(list(unique_true_freevars))),
                list(_sort_by_name(list(unique_false_freevars))),
            )

        (
            true_shared,
            false_shared,
            unique_true,
            unique_false,
        ) = dedup_and_sort_lifted_freevars(true_lifted_freevars, false_lifted_freevars)

        # Let's say we capture cond(pred, true_fn, false_fn, (x,))
        # With mannually_set_graph_input set to False,
        # true_fn has lifted variables x, a, b, c
        # false_fn has lifted variables x, a, b, d
        # Then fixup_branch_inps make sure both branches have the same signature, i.e.:
        # - true_fn(x, a, b, c_true_branch, d_false_branch)
        # - false_fn(x, a, b, c_true_branch, d_false_branch)
        #
        # More formally, the signature has three parts in the following order:
        # 1. used in both branches: x, a, b
        # 2. only used in true branches: c, suffixed with _true_branch
        # 3. only used in false branches: d, suffixed with _false_branch
        # Within each part, we re-order the nodes by name to have a derterministic ordering for testing.
        def fixup_branch_inps(
            graph, lifted_freevars, shared, unique_true, unique_false
        ):
            def _insert_or_replace_phs(new_args, name_suffix):
                for arg in new_args:
                    new_ph = graph.placeholder(arg.node.name + name_suffix)
                    # Override with new_ph if there exists a old placeholder.
                    if arg in lifted_freevars:
                        old_ph = lifted_freevars[arg].node
                        old_ph.replace_all_uses_with(new_ph)
                        # replace_all_uses_with doesn't clean users. Clean it mannually so that we could erase it.
                        old_ph.users = {}
                        graph.erase_node(old_ph)

            first_not_ph_node = next(
                node for node in graph.nodes if node.op != "placeholder"
            )
            with graph.inserting_before(first_not_ph_node):
                _insert_or_replace_phs(shared, "")
                _insert_or_replace_phs(unique_true, "_true_branch")
                _insert_or_replace_phs(unique_false, "_false_branch")

        fixup_branch_inps(
            true_graph, true_lifted_freevars, true_shared, unique_true, unique_false
        )
        fixup_branch_inps(
            false_graph, false_lifted_freevars, false_shared, unique_true, unique_false
        )

        true_name = add_subgraph(
            tx,
            self.source,
            "cond_true",
            torch.fx.GraphModule(true_nn_modules, true_graph),
        )
        false_name = add_subgraph(
            tx,
            self.source,
            "cond_false",
            torch.fx.GraphModule(false_nn_modules, false_graph),
        )

        true_node = make_attr(tx, true_name)
        false_node = make_attr(tx, false_name)

        p_args = (
            args[0].as_proxy(),
            true_node,
            false_node,
            # We pick true_shared but it shouldn't matter
            true_shared + unique_true + unique_false,
        )

        return _call_function_and_unflatten_output(
            tx, torch.ops.higher_order.cond, p_args, {}, true_r, true_treespec
        )


def non_single_tensor_return_unsupported(api, ret):
    from . import TensorVariable

    if not isinstance(ret, TensorVariable):
        raise Unsupported(
            f"{api} over function that returns something " f"other than one Tensor"
        )


class MapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]
    ) -> VariableTracker:
        from . import NestedUserFunctionVariable, TensorVariable, UserFunctionVariable
        from .builder import wrap_fx_proxy_cls

        if len(kwargs) > 0:
            unimplemented(
                "torch.ops.higher_order.map: kwargs are not supported in the map operator."
            )

        assert type(args[0].realize()) in (
            UserFunctionVariable,
            NestedUserFunctionVariable,
        )
        assert type(args[1].realize()) is TensorVariable

        sample_shape = get_fake_value(args[1].as_proxy().node, tx).size()

        if len(sample_shape) < 1 or sample_shape[0] == 0:
            unimplemented(
                "map() operator doesn't support scalar or zero-sized tensors during tracing."
            )

        # To get the example output from map() we will need to provide at least one sample to
        # the loop body. In our case we will always use xs[0], and our map() won't support zero
        # sized tensor during tracing.
        first_dim = wrap_fx_proxy_cls(
            target_cls=TensorVariable, tx=tx, proxy=args[1].as_proxy()[0]
        )

        # TODO: Support kwargs
        (
            (body_r, body_spec),
            body_graph,
            body_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            args[0],
            [
                first_dim,
                *args[2:],
            ],
            {},
            "torch.ops.higher_order.map",
            source_target=self.value,
            should_flatten_outputs=True,
        )

        body_nn_modules = dict(tx.output.nn_modules)

        body_name = add_subgraph(
            tx,
            self.source,
            "map_body",
            torch.fx.GraphModule(body_nn_modules, body_graph),
        )

        body_node = make_attr(tx, body_name)
        p_args = (
            body_node,
            1,  # right now we only supports num_mapped = 1
            *(arg.as_proxy() for arg in args[1:]),
            *(arg for arg in body_lifted_freevars.keys()),
        )
        return _call_function_and_unflatten_output(
            tx, torch.ops.higher_order.map_impl, p_args, {}, body_r, body_spec
        )


class ExecutorchCallDelegateHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        # This is operator for delegation within Executorch which calls a
        # specific function in the given lowered module with the given
        # operators. The actual operator is defined in the Executorch codebase.
        # This is a bad hierarchical violation since
        # executorch_call_delegate sits at a higher level than dynamo, but
        # there's no real solution to this issue yet.
        if len(kwargs) > 0:
            unimplemented(
                "executorch_call_delegate: kwargs arguments were not enabled."
            )
        lowered_module = tx.output.get_submodule(args[0].module_key)

        lowered_node = make_attr(tx, args[0].module_key)

        p_args = tuple(arg.as_proxy() for arg in args[1:])
        real_sub_args = pytree.tree_map_only(
            torch.fx.Proxy, lambda a: get_real_value(a.node, tx.output), p_args
        )

        example_res = lowered_module.original_module(*real_sub_args)

        # NOTE [Guaranteeing the 1-1 correspondence of FakeTensors and real tensors]:
        # executorch modules promise not to alias inputs and outputs.
        # Thus, output FakeTensors will correctly not alias input FakeTensors.
        _assert_tensors_nonaliasing(real_sub_args, example_res)

        example_value = deepcopy_to_fake_tensor(example_res, tx.fake_mode)

        p_args = (lowered_node,) + p_args

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


class FunctorchGradHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from . import ConstantVariable
        from .builder import wrap_fx_proxy

        # TODO: Support `fn` with kwargs.
        if not torch._dynamo.config.capture_func_transforms:
            unimplemented(
                "torch.func.grad capture is disabled, "
                "it can be turned on by setting "
                "`torch._dynamo.config.capture_func_transforms=True`"
            )
        # [NOTE] Here we are (roughly) modelling the following
        #
        #   grad_fn = torch.func.grad(fn, argnums=.., has_aux=..)
        #   grad_output = grad_fn(x)
        grad_args = (args[0], args[1], args[2])

        # get arguments
        func, argnums, has_aux = grad_args
        kwargs = args[4].items
        if len(kwargs) > 0:
            # Since speculate_subgraph doesn't support kwargs, we can't handle this for now.
            unimplemented(
                "torch.func.grad: kwargs arguments are currently unsupported."
            )

        # Trace through the `func`
        # NOTE [HACK: Enable autograd while tracing function]
        # `torch.func.grad` should not be affected by `no_grad` outside of `grad`.
        # So, we enable_grad right before the function to which `grad` is applied
        # (the parts explicitly disabled with `no_grad` inside the function are still disabled).
        # Eg.
        # def f(x):
        #     with no_grad():  # This will disable grad tracking under it.
        #        y = x * 2
        #
        #     return x ** 2 - y  # grad tracking should be enabled irrespective of outside `no_grad`.
        #
        # with no_grad():  # This will not disable grad tracking inside of grad(f).
        #     grad_o = torch.func.grad(f)(x)
        # TODO: Support kwargs
        (body_r, _), body_graph, body_lifted_freevars = speculate_subgraph(
            tx,
            func,
            args[3].items,
            {},
            "torch.func.grad",
            source_target=self.value,
            # See NOTE [HACK: Enable autograd while tracing function]
            enable_grad=True,
        )

        body_name = add_subgraph(
            tx,
            self.source,
            "grad_body",
            torch.fx.GraphModule(tx.output.nn_modules, body_graph),
        )
        body_node = make_attr(tx, body_name)
        grad_proxy_args = (
            body_node,
            *(arg.as_proxy() for arg in grad_args[1:]),
        )

        # Model `grad_fn = grad(fn, *grad_args, **grad_kwargs)`
        grad_fn = tx.output.create_proxy(
            "call_function",
            torch.func.grad,
            args=tuple(grad_proxy_args),
            kwargs={},
            name="grad_proxy",
        )

        # Pass lifted freevars to the call to `grad_fn`
        args = args[3].items
        grad_fn_args = tuple(arg.as_proxy() for arg in args) + tuple(
            body_lifted_freevars
        )

        # Call grad_fn with inputs.
        # grad_output = grad_fn(*grad_fn_args, **grad_fn_kwargs)
        grad_output = grad_fn(*grad_fn_args)

        # `grad_fn(*grad_fn_args, **grad_fn_kwargs)`
        # Output of grad_fn is
        # For has_aux=False, Tuple[gradients of inputs indicated by argnums].
        # For has_aux=True, Tuple[Tuple[gradients of inputs indicated by argnums], aux values]
        # NOTE: example_value should match `grad_output`.
        def _from_args(idx):
            return args[idx].as_proxy().node.meta["example_value"].contiguous()

        def to_python_ints(argnums):
            if not isinstance(argnums, (ConstantVariable, TupleVariable)):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"argnums is expected to be int or tuple of ints. Got {argnums}.",
                )

            if isinstance(argnums, ConstantVariable):
                if not isinstance(argnums.value, (int, tuple)):
                    raise UserError(
                        UserErrorType.INVALID_INPUT,
                        f"argnums is expected to be int or tuple of ints. Got {argnums}.",
                    )
                return argnums.value
            else:
                const_vars = argnums.unpack_var_sequence(tx)
                if not all(
                    isinstance(var, ConstantVariable) and isinstance(var.value, int)
                    for var in const_vars
                ):
                    raise UserError(
                        UserErrorType.INVALID_INPUT,
                        f"argnums is expected to contain int only. Got {const_vars}.",
                    )
                return tuple(var.value for var in const_vars)

        argnums_v = to_python_ints(argnums)
        example_value = pytree.tree_map(_from_args, argnums_v)

        if has_aux.value:
            # case : has_aux = True
            # NOTE: Currently speculate subgraph allows body_r to be
            # Tensor or Tuple/List of Tensor.
            # Since `grad` expects output with has_aux
            # to be (output, aux), only valid output currently is
            # (output, some_tensor)
            body_r_proxy = body_r.as_proxy()
            aux = body_r_proxy[1].node.meta["example_value"]
            example_value = (example_value, aux)

        fx_proxy = wrap_fx_proxy(tx=tx, proxy=grad_output, example_value=example_value)

        # Call contiguous on all the computed grads.
        if not has_aux.value:
            if isinstance(argnums_v, int):
                return fx_proxy.call_method(tx, "contiguous", (), {})
            else:
                grads = fx_proxy
                items = []
                for idx in range(len(argnums_v)):
                    proxy = grads.call_method(
                        tx, "__getitem__", (ConstantVariable.create(idx),), {}
                    ).call_method(tx, "contiguous", (), {})
                    items.append(proxy)
                return TupleVariable(items)
        else:  # case: has_aux.value = True
            # fx_proxy -> Tuple(grads, aux)
            grads = fx_proxy.call_method(
                tx, "__getitem__", (ConstantVariable.create(0),), {}
            )
            aux = fx_proxy.call_method(
                tx, "__getitem__", (ConstantVariable.create(1),), {}
            )
            if isinstance(argnums_v, int):
                return TupleVariable([grads.call_method(tx, "contiguous", (), {}), aux])
            else:
                items = []
                for idx in range(len(argnums_v)):
                    proxy = grads.call_method(
                        tx, "__getitem__", (ConstantVariable.create(idx),), {}
                    ).call_method(tx, "contiguous", (), {})
                    items.append(proxy)
                return TupleVariable([TupleVariable(items), aux])


class AutogradFunctionMethodHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def __init__(
        self, value, fwd_bwd_tracer=None, source: Optional[Source] = None, **kwargs
    ):
        super().__init__(value, source, **kwargs)
        # The fwd_bwd_tracer is owned by AutogradFunctionVariable and passed
        # in for speculation. It allows us to share tracing information about proxies
        # across fwd bwd, such as when users stash tensors on a context.
        self.fwd_bwd_tracer = fwd_bwd_tracer

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from . import UserFunctionVariable
        from .builder import wrap_fx_proxy

        tracer = self.fwd_bwd_tracer

        if len(kwargs) > 0:
            unimplemented(
                "kwargs have not been implemented for torch.autograd.Function"
            )

        from . import TorchInGraphFunctionVariable

        always_restore = self.value.__name__ == "trampoline_autograd_bwd"
        if (
            self.value.__name__ == "trampoline_autograd_bwd"
            or self.value.__name__ == "trampoline_autograd_fwd"
        ):
            fn = UserFunctionVariable(self.value, source=self.source)
        else:
            fn = TorchInGraphFunctionVariable(self.value)
        # TODO(jansel): BUG!!! we aren't copying on the line below, so the post-pre check below is pointless
        pre_guards = tx.output.guards
        # In eager-mode PyTorch, if we only compute first-order gradients,
        # then the grad_mode is False during the backward pass.
        # torch.compile assumes that we only compute first-order gradients,
        # so we want to speculate the backward pass with the grad mode disabled.
        enable_grad = (
            False if self.value.__name__ == "trampoline_autograd_bwd" else None
        )

        # TODO: Support kwargs
        (
            (body_r, _),
            body_graph,
            body_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            fn,
            [
                *args,
            ],
            {},
            "the user-defined autograd.Function",
            source_target=self.value,
            # Backwards should never, ever be stored!
            always_restore=always_restore,
            restore_side_effects=False,
            tracer=tracer,
            enable_grad=enable_grad,
        )
        post_guards = tx.output.guards
        if body_lifted_freevars:
            unimplemented("NYI - freevars in autograd function.")

        if always_restore:
            if post_guards - pre_guards:
                unimplemented("NYI - New guards discovered in a restoring state")
            # Nothing left to do here
            return None

        # don't add call module to parent graph if speculating forward
        # return the result directly
        if self.value.__name__ == "trampoline_autograd_fwd":
            return body_r

        p_args = (
            *(arg.as_proxy() for arg in args),
            *(arg for arg in body_lifted_freevars.keys()),
        )
        example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )

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


class WrapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def create_wrapped_node(self, tx, args, kwargs, description):
        # See NOTE [HigherOrderOperator tracing design] for more details

        (
            (body_r, treespec),
            body_graph,
            body_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            args[0],  # function
            [*args[1:]],
            kwargs,
            description,
            source_target=self.value,
            manually_set_subgraph_inputs=False,
            should_flatten_outputs=True,
        )

        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = add_subgraph(
            tx,
            self.source,
            "wrap_body",
            body_gmod,
        )

        body_node = make_attr(tx, body_name)

        # Since, we call `speculate_subgraph` with `manually_set_subgraph_inputs=False`,
        # all the arguments are lifted.
        lifted_args = tuple(arg for arg in body_lifted_freevars.keys())

        proxy_args = (body_node,) + lifted_args
        example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],
            body_r.as_proxy(),
        )

        return proxy_args, {}, example_value, body_r, treespec, body_gmod

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # This flattens the kwargs into lifted args
        p_args, p_kwargs, example_value, body_r, treespec, _ = self.create_wrapped_node(
            tx, args, kwargs, "wrap"
        )

        if len(p_kwargs) > 0:
            unimplemented("kwargs should have been flattened into lifted args")

        return _call_function_and_unflatten_output(
            tx, self.value, tuple(p_args), p_kwargs, body_r, treespec
        )


class OutDtypeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        if len(kwargs) > 0:
            unimplemented("out_dtype does not handle kwargs")

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


class CheckpointHigherOrderVariable(WrapHigherOrderVariable):
    def call_function(
        self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]
    ) -> VariableTracker:
        from torch._higher_order_ops.wrap import TagActivationCheckpoint
        from torch.utils.checkpoint import noop_context_fn
        from .builder import wrap_fx_proxy

        context_fn = None
        if "context_fn" in kwargs and kwargs["context_fn"] != noop_context_fn:
            context_fn = kwargs.pop("context_fn").fn

        checkpoint_kwargs, gmod_kwargs = TagActivationCheckpoint.divide_kwargs(kwargs)

        # Here we use checkpoint_kwargs (and not gmod kwargs). gmod_kwargs are
        # already flattened above and managed inside the fx graph.
        (
            p_args,
            _,
            example_value,
            body_r,
            treespec,
            checkpointed_gmod,
        ) = self.create_wrapped_node(
            tx, args, gmod_kwargs, "torch.utils.checkpoint.checkpoint"
        )
        if context_fn is not None:
            checkpointed_gmod.meta["_checkpoint_context_fn"] = context_fn

        _, checkpoint_kwargs = proxy_args_kwargs([], checkpoint_kwargs)

        # Store the invocation as a call
        variable = wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(p_args),
                kwargs=checkpoint_kwargs,
            ),
            example_value=example_value,
        )

        if treespec is None:
            return variable

        # Transform variable back into a list (previously made into a tuple by
        # speculate_subgraph function) so as to respect the pytree API typing.
        variable = BuiltinVariable(list).call_function(tx, [variable], {})

        return _make_inlined(tx, pytree.tree_unflatten)(variable, treespec)


class ExportTracepointHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

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
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from . import TensorVariable

        assert "fn" in kwargs
        fn = kwargs["fn"]
        assert len(args) == 1
        grad = args[0]
        assert isinstance(grad, TensorVariable)

        return fn.call_function(tx, args, {})
