from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.fx.traceback as fx_traceback

import torch.utils._pytree as pytree

from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._dynamo.exc import CondOpArgsMismatchError

from torch._functorch.eager_transforms import (
    _unwrap_all_tensors_from_functional,
    _wrap_all_tensors_to_functional,
    functionalize,
)
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)


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


@dataclass
class UnsupportedAliasMutationException(RuntimeError):
    reason: str


def cond(pred, true_fn, false_fn, operands):
    r"""
    Conditionally applies ``true_fn`` or ``false_fn``.

    ``cond`` is structured control flow operator. That is, it is like a Python if-statement,
    but has limitations on ``true_fn``, ``false_fn``, and ``operands`` that enable it to be
    capturable using torch.compile and torch.export.

    Assuming the constraints on ``cond``'s arguments are met, ``cond`` is equivalent to the following::

        def cond(pred, true_branch, false_branch, operands):
            if pred:
                return true_branch(*operands)
            else:
                return false_branch(*operands)

    .. warning::
       cond is a prototype feature in PyTorch, included as a part of the torch.export release. The main limitations are that
       it may not work in eager-mode PyTorch and you may encounter various failure modes while using it.
       Please look forward to a more stable implementation in a future version of PyTorch.

    Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    Args:
        - `pred (Union[bool, torch.Tensor])`: A boolean expression or a tensor with one element,
          indicating which branch function to apply.

        - `true_fn (Callable)`: A callable function (a -> b) that is within the
          scope that is being traced.

        - `false_fn (Callable)`: A callable function (a -> b) that is within the
          scope that is being traced. The true branch and false branch must have
          consistent input and outputs, meaning the inputs have to be the same, and
          the outputs have to be the same type and shape.

        - `operands (Tuple[torch.Tensor])`: A tuple of inputs to the true/false
          branches.

    Example:

        def true_fn(x: torch.Tensor):
            return x.cos()
        def false_fn(x: torch.Tensor):
            return x.sin()
        return cond(x.shape[0] > 4, true_fn, false_fn, (x,))

    Restrictions:
        - The conditional statement (aka `pred`) must meet one of the following constraints:

          - It's a `torch.Tensor` with only one element, and torch.bool dtype

          - It's a boolean expression, e.g. `x.shape[0] > 10` or `x.dim() > 1 and x.shape[1] > 10`

        - The branch function (aka `true_fn`/`false_fn`) must meet all of the following constraints:

          - The function signature must match with operands.

          - The function must return a tensor with the same metadata, e.g. shape,
            dtype, etc.

          - The function cannot have in-place mutations on inputs or global variables. (Note: in-place tensor
            operations such as `add_` for intermediate results are allowed in a branch)

    .. warning::

    Temporal Limitations:

        - `cond` only supports **inference** right now. Autograd will be supported in the future.

        - The **operands** must be a **tuple of tensors**. Pytree of tensors will be supported in the future.

        - The **output** of branches must be a **single Tensor**. Pytree of tensors will be supported in the future.

    """

    if torch._dynamo.is_compiling():
        return cond_op(pred, true_fn, false_fn, operands)

    def _validate_input(pred, true_fn, false_fn, operands):
        if not isinstance(pred, (bool, torch.Tensor, torch.SymBool)):
            raise RuntimeError(f"Expected pred to be bool or tensor, but got {pred}.")

        if isinstance(pred, torch.Tensor) and pred.numel() != 1:
            raise RuntimeError(
                f"Expected pred to be bool or single-element tensor, but got {pred}."
            )

        if not callable(true_fn) or not callable(false_fn):
            raise RuntimeError("Expect both branches to be callbale.")

        if not isinstance(operands, (tuple, list)) or any(
            not isinstance(t, torch.Tensor) for t in operands
        ):
            raise RuntimeError(
                f"Expect operands to be a tuple of Tensors, but got {operands}."
            )

    _validate_input(pred, true_fn, false_fn, operands)

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("torch.cond requires dynamo support.")

    with _set_compilation_env():
        return torch.compile(cond_op, backend="eager", fullgraph=True)(
            pred, true_fn, false_fn, operands
        )


"""
We're going to define a `cond_op` operation.
In order to do this, we need implementations for each of the dispatch keys.
"""
cond_op = HigherOrderOperator("cond")


def _maybe_run_with_interpreter(fn):
    maybe_interpreted_fn = fn
    if isinstance(fn, torch.fx.GraphModule) and fx_traceback.has_preserved_node_meta():
        # Running graph with interpreter is needed for propagating the stack_trace
        def graph_with_interpreter(*args):
            with fx_traceback.preserve_node_meta():
                return torch.fx.Interpreter(fn).run(*args)

        maybe_interpreted_fn = graph_with_interpreter
    return maybe_interpreted_fn


def trace_cond(proxy_mode, func_overload, pred, true_fn, false_fn, operands):
    assert isinstance(
        operands, (list, tuple)
    ), "Cond operands must be a list or tuple of tensors"
    assert all(
        isinstance(o, torch.Tensor) for o in operands
    ), "Cond operands must be a list of tensors"

    pre_dispatch = getattr(proxy_mode, "pre_dispatch", False)

    with disable_proxy_modes_tracing():
        true_graph = make_fx(
            _maybe_run_with_interpreter(true_fn), pre_dispatch=pre_dispatch
        )(*operands)
        false_graph = make_fx(
            _maybe_run_with_interpreter(false_fn), pre_dispatch=pre_dispatch
        )(*operands)

    true_outs = []
    false_outs = []
    for node in true_graph.graph.nodes:
        if node.op == "output":
            true_outs.extend(node.args)

    for node in false_graph.graph.nodes:
        if node.op == "output":
            false_outs.extend(node.args)

    flat_true_outs, _ = pytree.tree_flatten(true_outs)
    flat_false_outs, _ = pytree.tree_flatten(false_outs)
    if len(flat_true_outs) != len(flat_false_outs):
        raise CondOpArgsMismatchError(
            f"Expected to return same number of outputs but got:"
            f"\n  {true_fn.__name__} returns {len(flat_true_outs)} item(s)"
            f"\n  {false_fn.__name__} returns {len(flat_false_outs)} item(s)"
        )

    for i in range(0, len(flat_true_outs)):
        true_out = flat_true_outs[i]
        false_out = flat_false_outs[i]
        if true_out.meta["tensor_meta"] != false_out.meta["tensor_meta"]:
            raise CondOpArgsMismatchError(
                f"Expected each tensor to have same metadata but got:"
                f"\n  {true_fn.__name__} returns {true_out.meta['tensor_meta']}"
                f"\n  {false_fn.__name__} returns {false_out.meta['tensor_meta']}"
            )

    # There are probably better ways - I know that create_arg has some self incrementing name
    # magic to it, but since we explicitly have to get the name for register_module,
    # I was not sure how to do that. This kinda simulates it.
    next_name = None
    i = 0
    while not next_name:
        candidate = f"true_graph_{i}"
        if hasattr(proxy_mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate

    true_name = next_name
    false_name = f"false_graph_{i}"
    assert not hasattr(proxy_mode.tracer.root, false_name)

    proxy_mode.tracer.root.register_module(true_name, true_graph)
    proxy_mode.tracer.root.register_module(false_name, false_graph)

    args = (pred, true_graph, false_graph, operands)

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="conditional"
    )

    # At this point, we're *guaranteed* that whether an output came from the
    # true or false branch is indistinguishable. So, as this is just for tracing
    # purposes, choose the true branch.

    # TODO: Uhh.... it shouldn't matter, but changing this to true_fn results in
    # a FakeTensorMode error :
    # `Current active mode <class 'torch._subclasses.fake_tensor.FakeTensorMode'> not registered`
    # TODO Sometimes the operands are not completely FakeTensor, something seems went wrong in
    # dynamo? Because of that it runs real computation sometimes and re-triggering downstream dispatch keys.
    out = false_fn(*operands)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@cond_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def cond_op_dense(pred, true_fn, false_fn, operands):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)


cond_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(cond_op, deferred_error=True)
)


@cond_op.py_impl(ProxyTorchDispatchMode)
def inner(pred, true_fn, false_fn, operands):
    # TODO Move this to proper utility function
    from torch._ops import mode_stack_per_key, temporarily_pop_mode

    # torch.cond expects ProxyTorchDispatchMode to **still** be on the stack
    # at the time that its proxy implementation is called.
    # However, the mode can live in one of two places, depending on
    # whether we're doing pre_dispatch tracing or normal tracing.
    pre_dispatch_modes = mode_stack_per_key().get(DispatchKey.PreDispatch, [])  # type: ignore[attr-defined]
    if len(pre_dispatch_modes) > 0:
        with temporarily_pop_mode(pre_dispatch_modes) as mode:
            if mode.enable_tracing:
                return trace_cond(mode, cond_op, pred, true_fn, false_fn, operands)
            else:
                return cond_op(pred, true_fn, false_fn, operands)
    mode = _get_current_dispatch_mode()
    assert mode is not None, "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        if mode.enable_tracing:
            return trace_cond(mode, cond_op, pred, true_fn, false_fn, operands)
        else:
            return cond_op(pred, true_fn, false_fn, operands)


@cond_op.py_impl(FakeTensorMode)
def cond_fake_tensor_mode(pred, true_fn, false_fn, operands):
    true_outs = true_fn(*operands)
    flat_true_outs, _ = pytree.tree_flatten(true_outs)
    flat_false_outs, _ = pytree.tree_flatten(false_fn(*operands))
    if len(flat_true_outs) != len(flat_false_outs):
        raise RuntimeError("Unmatched number of outputs from cond() branches.")

    for true_out, false_out in zip(flat_true_outs, flat_false_outs):
        true_meta = _extract_tensor_metadata(true_out)
        false_meta = _extract_tensor_metadata(false_out)
        if true_meta != false_meta:
            raise CondOpArgsMismatchError(
                f"Expected each tensor to have same metadata but got:"
                f"\n  {true_fn.__name__} returns {true_meta}"
                f"\n  {false_fn.__name__} returns {false_meta}"
            )
    return true_outs


def _has_potential_branch_input_mutation(branch, inputs):
    """
    Dispatch-trace the branch with inputs and check if
    producing graph has mutable op on the input. This is
    bit restrictive as the branch must be traceable.
    """
    try:
        gm = make_fx(branch)(*inputs)
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


def _has_potential_branch_input_alias(branch, inputs):
    """
    Dispatch-trace the branch with inputs and check if
    producing graph has output aliasing the branch input. This is
    bit restrictive as the branch must be traceable.
    """
    try:
        gm = make_fx(branch)(*inputs)

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

                if any(pytree.tree_flatten(pytree.tree_map(check_alias, node.args))[0]):
                    return True

        for _, module in gm.named_children():
            if isinstance(module, torch.fx.GraphModule) and _detect_input_alias(module):
                return True

        return False

    return _detect_input_alias(gm)


@cond_op.py_impl(DispatchKey.Functionalize)
def cond_func(pred, true_fn, false_fn, inputs):
    reapply_views = torch._C._functionalization_reapply_views_tls()
    unwrapped_inputs = _unwrap_all_tensors_from_functional(
        inputs, reapply_views=reapply_views
    )
    unwrapped_pred = _unwrap_all_tensors_from_functional(
        pred, reapply_views=reapply_views
    )
    mode = "mutations_and_views" if reapply_views else "mutations"
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        functional_true = functionalize(true_fn, remove=mode)
        functional_false = functionalize(false_fn, remove=mode)
        for branch in [true_fn, false_fn]:
            if _has_potential_branch_input_mutation(branch, unwrapped_inputs):
                raise UnsupportedAliasMutationException(
                    "One of torch.cond branch might be modifying the input!"
                )

            if _has_potential_branch_input_alias(branch, unwrapped_inputs):
                raise UnsupportedAliasMutationException(
                    "One of torch.cond branch might be aliasing the input!"
                )

        cond_return = cond_op(
            unwrapped_pred, functional_true, functional_false, unwrapped_inputs
        )
        return _wrap_all_tensors_to_functional(cond_return, level=0)


@cond_op.py_impl(torch._C._functorch.TransformType.Functionalize)
def cond_functionalize(interpreter, pred, true_fn, false_fn, inputs):
    """
    Functionalization implementation for torch.cond. Currently:
      1. We don't allow any input mutation inside the branches
      2. Our check for above condition is not exhaustive
    """
    reapply_views = interpreter.functionalize_add_back_views()
    mode = "mutations_and_views" if reapply_views else "mutations"
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_inputs = _unwrap_all_tensors_from_functional(
        inputs, reapply_views=reapply_views
    )
    unwrapped_pred = _unwrap_all_tensors_from_functional(
        pred, reapply_views=reapply_views
    )

    functional_true_fn = functionalize(true_fn, remove=mode)
    functional_false_fn = functionalize(false_fn, remove=mode)

    with interpreter.lower():
        for branch in [functional_true_fn, functional_false_fn]:
            if _has_potential_branch_input_mutation(branch, unwrapped_inputs):
                raise UnsupportedAliasMutationException(
                    "One of torch.cond branch might be modifying the input!"
                )
        for branch in [true_fn, false_fn]:
            if _has_potential_branch_input_alias(branch, unwrapped_inputs):
                raise UnsupportedAliasMutationException(
                    "One of torch.cond branch might be aliasing the input!"
                )

        cond_return = cond_op(
            unwrapped_pred, functional_true_fn, functional_false_fn, unwrapped_inputs
        )
        return _wrap_all_tensors_to_functional(cond_return, level=interpreter.level())


# TODO(voz): Make this automatic for keys, this is very ugly atm
cond_op.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
cond_op.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
cond_op.fallthrough(DispatchKey.ADInplaceOrView)
cond_op.fallthrough(DispatchKey.BackendSelect)
cond_op.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
cond_op.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
