import contextlib

import torch
import torch._subclasses.functional_tensor

import torch.utils._pytree as pytree

from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._C._functorch import (
    _add_batch_dim,
    get_unwrapped,
    is_batchedtensor,
    maybe_get_bdim,
)
from torch._functorch.utils import exposed_in
from torch._guards import detect_fake_mode

from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _set_compilation_env,
    reenter_make_fx,
    unique_graph_id,
    UnsupportedAliasMutationException,
)

from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import (
    disable_functional_mode,
)
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    _temp_remove_pre_dispatch_torch_function_mode,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._python_dispatch import _get_current_dispatch_mode
from torch.fx.experimental.proxy_tensor import _temp_remove_pre_dispatch_torch_function_mode
from .utils import _from_fun, clone_outputs_aliasing_inputs, prepare_fw_with_masks, create_fw_bw_graph

@exposed_in("torch")
def cond(pred, true_fn, false_fn, operands):
    r"""
    Conditionally applies `true_fn` or `false_fn`.

    .. warning::
        `torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types and
        doesn't support training currently. Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    `cond` is structured control flow operator. That is, it is like a Python if-statement,
    but has restrictions on `true_fn`, `false_fn`, and `operands` that enable it to be
    capturable using torch.compile and torch.export.

    Assuming the constraints on `cond`'s arguments are met, `cond` is equivalent to the following::

        def cond(pred, true_branch, false_branch, operands):
            if pred:
                return true_branch(*operands)
            else:
                return false_branch(*operands)

    Args:
        pred (Union[bool, torch.Tensor]): A boolean expression or a tensor with one element,
          indicating which branch function to apply.

        true_fn (Callable): A callable function (a -> b) that is within the
          scope that is being traced.

        false_fn (Callable): A callable function (a -> b) that is within the
          scope that is being traced. The true branch and false branch must
          have consistent input and outputs, meaning the inputs have to be
          the same, and the outputs have to be the same type and shape.

        operands (Tuple of possibly nested dict/list/tuple of torch.Tensor): A tuple of inputs to the true/false functions.

    Example::

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

          - The function cannot have in-place mutations on inputs or global variables.
            (Note: in-place tensor operations such as `add_` for intermediate results
            are allowed in a branch)

    .. warning::
        Temporal Limitations:

        - The **output** of branches must be a **single Tensor**. Pytree of tensors will be supported in the future.

    """

    if torch.compiler.is_dynamo_compiling():
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

        if not isinstance(operands, (tuple, list)) or pytree.tree_any(
            lambda t: not isinstance(t, torch.Tensor), operands
        ):
            raise RuntimeError(
                "Expect operands to be a tuple of possibly nested dict/list/tuple that only"
                f"consists of tensor leaves, but got {operands}."
            )

    _validate_input(pred, true_fn, false_fn, operands)

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("torch.cond requires dynamo support.")

    with _set_compilation_env():
        with torch._dynamo.utils.disable_cache_limit():
            with _temp_remove_pre_dispatch_torch_function_mode():
                return torch.compile(cond_op, backend="eager", fullgraph=True)(
                    pred, true_fn, false_fn, operands
                )

"""
We're going to define a `cond_op` operation.
In order to do this, we need implementations for each of the dispatch keys.
"""
cond_op = HigherOrderOperator("cond")
cond_op.__module__ = "torch.ops.higher_order"

def create_fw_bw_graph_branches(true_fn, false_fn, *operands):
    
    # from torch._functorch.aot_autograd import AOTConfig, create_joint
    # dummy_aot_config = AOTConfig(
    #     fw_compiler=None,  # type: ignore[arg-type]
    #     bw_compiler=None,  # type: ignore[arg-type]
    #     partition_fn=None,  # type: ignore[arg-type]
    #     decompositions={},
    #     num_params_buffers=0,
    #     aot_id=0,
    #     keep_inference_input_mutations=False,
    # )

    # Note:[HOP create fw_bw graph] We create "clean" environments for make_fx by suspending all dispatch keys
    # between Autograd and Python key. Currently, we only suspend functionalization but more can be
    # added when required. Will encounter two problems if we don't suspend functionalization:
    #
    # 1. make_fx fails to capture operations on input: the inputs are wrapped as _to_functional_tensor_wrapper,
    # but they will be unwrapped before entering ProxyTorchDispatchMode as part of the dispatching.
    # However, it's the outside wrapper that tracer creates proxies for. This casuses tracer fail to
    # fetch the proxy for the inputs and fail to capture any operations on them.
    #
    # 2. make_fx fails to capture output: the outputs after ProxyTorchDispatchMode are further
    # wrapped as FunctionalTensorWrapper in Functionalize key after return. However, the tracer
    # only associates the inner tensor with proxy in ProxyTorchDispatchMode. Therefore,
    # when creating the output node, it fails to associate the wrapped tensor with its proxy.
    # Instead, it will create _tensor_constant as output.

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():

            # num_mapped_args = len(operands)
            unwrapped_mapped_operands = pytree.tree_map(_from_fun, operands)
            example_operands = unwrapped_mapped_operands

            #Note, the true_fn and the false_fn produce the same output
            #shape, thus we can simply generate the example outputs from the true_fn.
            example_flat_out = pytree.tree_map(
                _from_fun, true_fn(*example_operands)
            )
            if any(
                not isinstance(out, torch.Tensor)
                for out in example_flat_out
                if out is not None
            ):
                raise RuntimeError(
                    "Expect outputs of map only contains tensors or None. "
                    f"Got types {[type(out) for out in example_flat_out]}."
                )
            # example_grad = [_from_fun(out) for out in example_flat_out]
            fw_true_graph, joint_true_graph = create_fw_bw_graph(true_fn, *example_operands)
            fw_false_graph, joint_false_graph = create_fw_bw_graph(false_fn, *example_operands)

            # fw_true_graph = make_fx(true_fn)(*example_operands)
            # fw_false_graph = make_fx(false_fn)(*example_operands)

        # def joint_f_true(*joint_mapped_args):
        #     mapped_input = joint_mapped_args[:num_mapped_args]
        #     mapped_grads = joint_mapped_args[num_mapped_args:]

        #     joint = create_joint(prepare_fw_with_masks(true_fn), aot_config=dummy_aot_config)
        #     _, grads = joint(
        #         list(mapped_input),
        #         [
        #             grad
        #             for grad in mapped_grads
        #             if grad is not None and grad.requires_grad
        #         ],
        #     )

        #     # In order to keep map functional for backward graph,
        #     # we clone outputs that are aliasing inputs           
        #     maybe_clone = clone_outputs_aliasing_inputs(joint_mapped_args)

        #     return pytree.tree_map(maybe_clone, grads)
        
        # def joint_f_false(*joint_mapped_args):
        #     mapped_input = joint_mapped_args[:num_mapped_args]
        #     mapped_grads = joint_mapped_args[num_mapped_args:]

        #     joint = create_joint(prepare_fw_with_masks(false_fn), aot_config=dummy_aot_config)
        #     _, grads = joint(
        #         list(mapped_input),
        #         [
        #             grad
        #             for grad in mapped_grads
        #             if grad is not None and grad.requires_grad
        #         ],
        #     )

        #     # In order to keep map functional for backward graph,
        #     # we clone outputs that are aliasing inputs
        #     maybe_clone = clone_outputs_aliasing_inputs(joint_mapped_args)

        #     return pytree.tree_map(maybe_clone, grads)

        # joint_operands_grads = list(example_operands) + list(example_grad)
        # # joint_true_graph = make_fx(joint_f_true)(*joint_operands_grads)
        # joint_false_graph = make_fx(joint_f_false)(*joint_operands_grads)
        return fw_true_graph, fw_false_graph, joint_true_graph, joint_false_graph


def trace_cond(proxy_mode, func_overload, pred, true_fn, false_fn, operands):
    assert isinstance(
        operands, (list, tuple)
    ), "Cond operands must be a list or tuple of tensors"
    assert all(
        isinstance(o, torch.Tensor) for o in operands
    ), "Cond operands must be a list of tensors"

    true_graph = reenter_make_fx(true_fn)(*operands)
    false_graph = reenter_make_fx(false_fn)(*operands)

    true_outs = []
    false_outs = []
    for node in true_graph.graph.nodes:
        if node.op == "output":
            true_outs.extend(node.args)

    for node in false_graph.graph.nodes:
        if node.op == "output":
            false_outs.extend(node.args)

    flat_true_outs = pytree.arg_tree_leaves(*true_outs)
    flat_false_outs = pytree.arg_tree_leaves(*false_outs)
    if len(flat_true_outs) != len(flat_false_outs):
        raise torch._dynamo.exc.CondOpArgsMismatchError(
            f"Expected to return same number of outputs but got:"
            f"\n  {true_fn.__name__} returns {len(flat_true_outs)} item(s)"
            f"\n  {false_fn.__name__} returns {len(flat_false_outs)} item(s)"
        )

    for i in range(0, len(flat_true_outs)):
        true_out = flat_true_outs[i]
        false_out = flat_false_outs[i]
        if true_out.meta["tensor_meta"] != false_out.meta["tensor_meta"]:
            raise torch._dynamo.exc.CondOpArgsMismatchError(
                f"Expected each tensor to have same metadata but got:"
                f"\n  {true_fn.__name__} returns {true_out.meta['tensor_meta']}"
                f"\n  {false_fn.__name__} returns {false_out.meta['tensor_meta']}"
            )

    i, true_name = unique_graph_id(proxy_mode, prefix="true_graph")

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

    # TODO: the unbacked symbol allocations MUST NOT leak out, if you want to
    # support this we need to arrange for the reenter_make_fx unbacked SymInts
    # to be used, AND we need to arrange for some sort of unification between
    # the two branches (but not really unification; e.g., if one branch
    # returns [u0] and the other returns [5] this is OK but you MUST NOT
    # conclude the result is 5.  Also if one branch returns [3] and another
    # branch returns [5] you can make it work by immediately allocating a new
    # unbacked SymInt here).
    ignore_fresh_unbacked = contextlib.nullcontext()
    if (fake_mode := detect_fake_mode()) and fake_mode.shape_env:
        ignore_fresh_unbacked = fake_mode.shape_env.ignore_fresh_unbacked_symbols()

    # TODO: Uhh.... it shouldn't matter, but changing this to true_fn results in
    # a FakeTensorMode error :
    # `Current active mode <class 'torch._subclasses.fake_tensor.FakeTensorMode'> not registered`
    # TODO Sometimes the operands are not completely FakeTensor, something seems went wrong in
    # dynamo? Because of that it runs real computation sometimes and re-triggering downstream dispatch keys.
    with ignore_fresh_unbacked:
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


class CondAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, fw_true_graph, fw_false_graph, joint_true_graph, joint_false_graph, *operands):
        ctx._pred = pred
        ctx._joint_true_graph = joint_true_graph
        ctx._joint_false_graph = joint_false_graph
        ctx.save_for_backward(*operands)
        
        with torch._C._AutoDispatchBelowAutograd():
            # with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            #     return torch.compile(cond_op, backend="eager", fullgraph=True)(
            #         pred, fw_true_graph, fw_false_graph, operands
            #     )
            return cond_op(
                pred, fw_true_graph, fw_false_graph, operands
            )

    @staticmethod
    def backward(ctx, *flat_grads):       
        operands = ctx.saved_tensors

        # with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
        #     grads = torch.compile(cond_op, backend="eager", fullgraph=True)(
        #         ctx._pred, ctx._joint_true_graph, ctx._joint_false_graph, operands + flat_grads
        #     )
        grads = cond_op(
            ctx._pred, ctx._joint_true_graph, ctx._joint_false_graph, operands + flat_grads
        )
        return None, None, None, None, None, *grads

@cond_op.py_impl(DispatchKey.Autograd)
def cond_autograd(pred, true_fn, false_fn, operands):
    fw_true_graph, fw_false_graph, joint_true_graph, joint_false_graph = create_fw_bw_graph_branches(true_fn, false_fn, *operands)
    flat_out = CondAutogradOp.apply(pred, fw_true_graph, fw_false_graph, joint_true_graph, joint_false_graph, *operands)
    return flat_out


@cond_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, pred, true_fn, false_fn, operands):
    if mode.enable_tracing:
        return trace_cond(mode, cond_op, pred, true_fn, false_fn, operands)
    else:
        return cond_op(pred, true_fn, false_fn, operands)


@cond_op.py_impl(FakeTensorMode)
def cond_fake_tensor_mode(mode, pred, true_fn, false_fn, operands):
    # Ignore here, because if you've gotten here but you're not manually
    # tracing the inner graphs, that means that you intend to reuse the graph
    # directly.  Which means the old unbacked symbol bindings are appropriate.
    # This strategy will not work if unbacked symbols can escape.
    ignore_fresh_unbacked = contextlib.nullcontext()
    if mode.shape_env:
        ignore_fresh_unbacked = mode.shape_env.ignore_fresh_unbacked_symbols()

    with mode, ignore_fresh_unbacked:
        true_outs = true_fn(*operands)
        flat_true_outs = pytree.tree_leaves(true_outs)
        flat_false_outs = pytree.tree_leaves(false_fn(*operands))
    if len(flat_true_outs) != len(flat_false_outs):
        raise RuntimeError("Unmatched number of outputs from cond() branches.")

    for true_out, false_out in zip(flat_true_outs, flat_false_outs):
        true_meta = _extract_tensor_metadata(true_out)
        false_meta = _extract_tensor_metadata(false_out)
        if true_meta != false_meta:
            raise torch._dynamo.exc.CondOpArgsMismatchError(
                f"Expected each tensor to have same metadata but got:"
                f"\n  {true_fn.__name__} returns {true_meta}"
                f"\n  {false_fn.__name__} returns {false_meta}"
            )
    return true_outs


@cond_op.py_functionalize_impl
def cond_func(ctx, pred, true_fn, false_fn, inputs):
    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    unwrapped_pred = ctx.unwrap_tensors(pred)
    with ctx.redispatch_to_next() as m:
        functional_true = ctx.functionalize(true_fn)
        functional_false = ctx.functionalize(false_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        for branch in [functional_true, functional_false]:
            if _has_potential_branch_input_mutation(
                branch, unwrapped_inputs, pre_dispatch=pre_dispatch
            ):
                raise UnsupportedAliasMutationException(
                    "One of torch.cond branch might be modifying the input!"
                )
        for branch in [true_fn, false_fn]:
            if _has_potential_branch_input_alias(
                branch, unwrapped_inputs, pre_dispatch=pre_dispatch
            ):
                raise UnsupportedAliasMutationException(
                    "One of torch.cond branch might be aliasing the input!"
                )

        cond_return = cond_op(
            unwrapped_pred, functional_true, functional_false, unwrapped_inputs
        )
        return ctx.wrap_tensors(cond_return)


@cond_op.py_impl(torch._C._functorch.TransformType.Vmap)
def cond_batch_rule(interpreter, pred, true_fn, false_fn, inputs):
    assert isinstance(
        inputs, (list, tuple)
    ), "Cond inputs must be a list or tuple of tensors"
    assert all(
        isinstance(i, torch.Tensor) for i in inputs
    ), "Cond inputs must be a list of tensors"

    pred_ = get_unwrapped(pred) if is_batchedtensor(pred) else pred

    # unbatched tensors are not vmapped
    tensors, in_dims = zip(
        *[
            (get_unwrapped(t), maybe_get_bdim(t)) if is_batchedtensor(t) else (t, None)
            for t in inputs
        ]
    )

    if is_batchedtensor(pred):
        # prepend "pred" and vmap everything
        tensors = (pred_,) + tensors
        in_dims = (0,) + in_dims

        def fn(p, *args):
            t = true_fn(*args)
            f = false_fn(*args)
            return torch.where(p, t[0], f[0])

        with interpreter.lower():
            result = torch.vmap(fn, in_dims=in_dims)(*tensors)

    else:
        # predicate is known at this stage and it is a boolean expression or a
        # tensor with one element.
        true_fn = torch.vmap(true_fn, in_dims=in_dims)
        false_fn = torch.vmap(false_fn, in_dims=in_dims)

        with interpreter.lower():
            result = cond_op(pred, true_fn, false_fn, tensors)

    if not isinstance(result, tuple):
        result = (result,)
    lvl = interpreter.level()
    return tuple([_add_batch_dim(r, 0, lvl) for r in result])
