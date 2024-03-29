from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import make_fx, track_tensor_tree
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils.weak import _WeakHashRef, WeakIdKeyDictionary, WeakTensorKeyDictionary


@dataclass
class UnsupportedAliasMutationException(RuntimeError):
    reason: str


def autograd_not_implemented_inner(
    operator: HigherOrderOperator, delayed_error: bool, *args: Any, **kwargs: Any
) -> Any:
    """If autograd is enabled and any of the arguments require grad this will either
    raise an error or return a DelayedError depending on the value of delayed.

    Args:
        operator: The HigherOrderOperator to call with the *args and **kwargs with
        op_name: The name of the HigherOrderOperator
        delayed_error: If True, return a DelayedError instead of raising an error
        args: The flattened operands to the HigherOrderOperator
        kwargs: The keyword arguments to the HigherOrderOperator

    Raises:
        RuntimeError: If autograd is enabled and any of the arguments to the HigherOrderOperator
    """
    with torch._C._AutoDispatchBelowAutograd():
        result = operator(*args, **kwargs)
        flat_operands = pytree.arg_tree_leaves(*args)
        if torch.is_grad_enabled() and any(
            f.requires_grad for f in flat_operands if isinstance(f, torch.Tensor)
        ):
            if delayed_error:
                err_fn = torch._C._functions.DelayedError(
                    f"Autograd not implemented for {str(operator)}",
                    1,
                )

                def fake_requires_grad(tensor):
                    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
                        tensor = tensor.detach()
                        tensor.requires_grad = True
                    return tensor

                return pytree.tree_map_only(
                    torch.Tensor, lambda x: err_fn(fake_requires_grad(x)), result
                )  # type: ignore[arg-type]
            else:
                raise RuntimeError(f"Autograd not implemented for {str(operator)}")
        return result


def autograd_not_implemented(op: HigherOrderOperator, deferred_error: bool) -> Callable:
    def inner(*args, **kwargs):
        return autograd_not_implemented_inner(op, deferred_error, *args, **kwargs)

    return inner


def _maybe_run_with_interpreter(fn):
    maybe_interpreted_fn = fn
    if isinstance(fn, torch.fx.GraphModule) and fx_traceback.has_preserved_node_meta():
        # Running graph with interpreter is needed for propagating the stack_trace
        def graph_with_interpreter(*args):
            with fx_traceback.preserve_node_meta():
                return torch.fx.Interpreter(fn).run(*args)

        maybe_interpreted_fn = graph_with_interpreter
    return maybe_interpreted_fn


@contextmanager
def _reset_tracer_states_temporarily(tracer):
    """_reset_tracer_states_temporarily temporarily resets the tracer states that are critical
    to sub-graph construction. The purpose is to use the current tracer to trace the subgraph.
    It creates an isolated tracing environment for the subgraph. Specifically, we reset the following:
    1. graph: it will be reset to an empty graph. Subgraph nodes will be appended to it.
    2. root: it's used to resolve the get_attr nodes of self in the subgraph.
    3. tensor_tracker: it associates tensors with their proxies.
    4. symnode_tracker: it associates SymInt/SymFloat/SymBool with their proxies.
    5. script_object_tracker: it associates script_object with their proxies.

    Args:
        tracer: the current tracer.
    """
    prev_graph = tracer.graph
    prev_root = tracer.root
    prev_symnode_tracker = tracer.symnode_tracker
    prev_tensor_tracker = tracer.tensor_tracker
    prev_script_object_tracker = tracer.script_object_tracker

    try:
        tracer.graph = torch.fx.Graph(
            prev_graph.owning_module, prev_graph._tracer_cls, prev_graph._tracer_extras
        )
        tracer.root = torch.nn.Module()
        tracer.tensor_tracker = WeakTensorKeyDictionary()
        tracer.symnode_tracker = torch.fx.experimental.proxy_tensor._SymNodeDict()
        tracer.script_object_tracker = WeakIdKeyDictionary(
            dict=None, ref_type=_WeakHashRef
        )
        yield tracer
    finally:
        tracer.graph = prev_graph
        tracer.root = prev_root
        tracer.symnode_tracker = prev_symnode_tracker
        tracer.tensor_tracker = prev_tensor_tracker
        tracer.script_object_tracker = tracer.script_object_tracker


def trace_subgraph(proxy_mode, func, args):
    """
    This function takes the current proxy_mode that's poped out of the torch dispatch stack
    and use this proxy_mode to trace `func` with `args`.

    Args:
        proxy_mode: the current proxy_mode that has been poped out of the torch dispatch stack.
        func: the function to be traced.
        args: the args to the func. Specifically, call func with func(*args).

    Returns:
        the traced graph module and the actual result of func(*args)

    Note: this function assume func has all the inputs passed in as args. For examples, parameters and buffers
    accessed inside of subgraph are lifted as inputs of subgraph.
    """
    graph_tracer = proxy_mode.tracer
    proxy_args = pytree.tree_map(graph_tracer.unwrap_proxy, args)
    with _reset_tracer_states_temporarily(graph_tracer) as subgraph_tracer:
        # create the placeholders for sub_graph.
        for i, (arg, parg) in enumerate(zip(args, proxy_args)):
            if not isinstance(parg, torch.fx.Proxy):
                # Sometimes we can have inputs that are not proxies e.g. tensor closures.
                # They're not tracked by parent graph yet so just give them a name manually.
                # Note this won't affect the correctness of the sub_graph.
                const_name = "_constant_input"
                new_proxy_arg = subgraph_tracer.create_proxy(
                    "placeholder", f"{const_name}{i}", (), {}, name=f"{const_name}{i}"
                )
            else:
                new_proxy_arg = subgraph_tracer.create_proxy(
                    "placeholder", parg.node.name, (), {}, name=parg.node.name
                )
            track_tensor_tree(arg, new_proxy_arg, constant=None, tracer=subgraph_tracer)

        with proxy_mode:
            real_out = _maybe_run_with_interpreter(func)(*args)

        proxy_out = pytree.tree_map_only(
            (
                torch.Tensor,
                torch.SymInt,
                torch.SymFloat,
                torch.SymBool,
                torch.ScriptObject,
            ),  # type: ignore[arg-type]
            subgraph_tracer.unwrap_proxy,
            real_out,
        )
        subgraph_tracer.create_node(
            "output",
            "output",
            (subgraph_tracer.create_arg(proxy_out),),
            {},
            None,
        )
        new_gm = torch.fx._lazy_graph_module._make_graph_module(
            subgraph_tracer.root, subgraph_tracer.graph
        )
    return new_gm, real_out


@contextmanager
def _set_compilation_env():
    _old_is_tracing = torch.fx._symbolic_trace._is_fx_tracing_flag
    try:
        # We need to turn off the is_fx_tracing_flag. Remove this flag check from dyanmo
        # once we are confident fx tracing works with dynamo.
        torch.fx._symbolic_trace._is_fx_tracing_flag = False
        yield
    finally:
        torch.fx._symbolic_trace._is_fx_tracing_flag = _old_is_tracing


def _has_potential_branch_input_mutation(branch, inputs, pre_dispatch=False):
    """
    Dispatch-trace the branch with inputs and check if
    producing graph has mutable op on the input. This is
    bit restrictive as the branch must be traceable.
    """
    try:
        gm = make_fx(branch, pre_dispatch=pre_dispatch)(*inputs)
    except UnsupportedAliasMutationException:
        # this can happen when nested cond_op is
        # functionalized
        return True
    except Exception as e:
        raise e

    def _detect_input_mutation(gm):
        input_nodes = set()
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                input_nodes.add(node)
            if node.op == "call_function":
                target = node.target
                if (
                    isinstance(target, torch._ops.OpOverload)
                    and target._schema.is_mutable
                ):
                    for arg in node.args:
                        if arg in input_nodes:
                            return True

        for _, module in gm.named_children():
            if isinstance(module, torch.fx.GraphModule):
                if _detect_input_mutation(module):
                    return True

        return False

    return _detect_input_mutation(gm)


def _has_potential_branch_input_alias(branch, inputs, pre_dispatch=False):
    """
    Dispatch-trace the branch with inputs and check if
    producing graph has output aliasing the branch input. This is
    bit restrictive as the branch must be traceable.
    """
    try:
        gm = make_fx(branch, pre_dispatch=pre_dispatch)(*inputs)
    except UnsupportedAliasMutationException:
        # this can happen when nested cond_op is
        # functionalized
        return True
    except Exception as e:
        raise e

    def _detect_input_alias(gm):
        input_storages = set()
        for node in gm.graph.nodes:
            # We need to check existence of "val" because we reuse the logic here
            # for map operator, where num_mapped_args is a scalar
            # and doesn't have a "val" meta.
            if node.op == "placeholder" and "val" in node.meta:
                input_storages.add(StorageWeakRef(node.meta["val"]._typed_storage()))
            if node.op == "output":

                def check_alias(out):
                    if out is not None and "val" in out.meta:
                        out_storage = StorageWeakRef(out.meta["val"]._typed_storage())
                        return out_storage in input_storages
                    return False

                if any(pytree.tree_leaves(pytree.tree_map(check_alias, node.args))):
                    return True

        for _, module in gm.named_children():
            if isinstance(module, torch.fx.GraphModule) and _detect_input_alias(module):
                return True

        return False

    return _detect_input_alias(gm)
