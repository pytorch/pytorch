# mypy: allow-untyped-defs
import functools
from typing import Callable, Union
from typing_extensions import TypeVarTuple

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import _maybe_run_with_interpreter, reenter_make_fx
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

from .utils import (
    _from_fun,
    _stack_pytree,
    _unstack_pytree,
    clone_outputs_aliasing_inputs,
    prepare_fw_with_masks,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
)


class MapImpl(HigherOrderOperator):
    def __init__(self):
        super().__init__("map_impl")

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


map_impl = MapImpl()


def create_fw_bw_graph(f, num_mapped_args, *args):
    mapped_xs = args[:num_mapped_args]
    pos_args = args[num_mapped_args:]

    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            unwrapped_mapped_xs = pytree.tree_map(_from_fun, mapped_xs)
            example_xs = _unstack_pytree(unwrapped_mapped_xs)[0]

            example_pos_args = [
                _from_fun(arg) if isinstance(arg, torch.Tensor) else arg
                for arg in pos_args
            ]
            example_flat_out = pytree.tree_map(
                _from_fun, f(*example_xs, *example_pos_args)
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
            example_grad = [_from_fun(out) for out in example_flat_out]

            fw_graph = make_fx(f)(*example_xs, *example_pos_args)

        from torch._functorch.aot_autograd import AOTConfig, create_joint

        dummy_aot_config = AOTConfig(
            fw_compiler=None,  # type: ignore[arg-type]
            bw_compiler=None,  # type: ignore[arg-type]
            partition_fn=None,  # type: ignore[arg-type]
            decompositions={},
            num_params_buffers=0,
            aot_id=0,
            keep_inference_input_mutations=False,
        )

        def joint_f(*example_args):
            joint_mapped_args = example_args[:joint_num_mapped]
            args = example_args[joint_num_mapped:]

            mapped_input = joint_mapped_args[:num_mapped_args]
            mapped_grads = joint_mapped_args[num_mapped_args:]

            joint = create_joint(prepare_fw_with_masks(f), aot_config=dummy_aot_config)
            _, grads = joint(
                list(mapped_input) + list(args),
                [
                    grad
                    for grad in mapped_grads
                    if grad is not None and grad.requires_grad
                ],
            )

            # In order to keep map functional for backward graph,
            # we clone outputs that are aliasing inputs
            maybe_clone = clone_outputs_aliasing_inputs(example_args)

            return pytree.tree_map(maybe_clone, grads)

        joint_num_mapped = len(example_grad) + len(example_xs)
        joint_graph = make_fx(joint_f)(*example_xs, *example_grad, *example_pos_args)
        return fw_graph, joint_graph


def map(
    f: Callable[[pytree.PyTree, tuple[pytree.PyTree, ...]], pytree.PyTree],
    xs: Union[pytree.PyTree, torch.Tensor],
    *args: TypeVarTuple,
):
    r"""
    Perfoms a map of f with xs. Intuitively, you can think of the semantic being:

    out = []
    for idx in len(xs.size(0)):
        xs_sliced = xs.select(0, idx)
        out.append(f(xs_sliced, *args))
    torch.stack(out)

    .. warning::
        `torch._higher_order_ops.map` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype


    Args:
        f (Callable): a callable that takes an input x, that could either be a single Tensor
            or a nested dict, list of tensors and some additional inputs
        xs: the inputs that're to be mapped over. We'll iterate over the first dim of each x
            and perform f on each slice.

        *args: additional arguments provided to each step of f. They could also be omitted and
            map is able to automatically figure out the read dependency.

    Return:
        the stacked output for each step of f

    Example:

        def f(xs):
            return xs[0] + xs[1] + const1 + const2

        xs = [torch.randn(2, 3), torch.randn(2, 3)]
        const1 = torch.randn(2, 3)
        const2 = torch.randn(2, 3)
        # returns a tensor of shape [2, 2, 3]
        torch._higher_order_ops.map(f, xs)

    """
    flat_xs, xs_spec = pytree.tree_flatten(xs)
    flat_args, args_spec = pytree.tree_flatten(args)
    if not all(isinstance(t, torch.Tensor) for t in flat_xs):
        raise RuntimeError(f"Mapped xs can only consist of tensors. Got xs {flat_xs}.")

    shapes = [xs.shape for xs in flat_xs]
    leading_dim_size = shapes[0][0]
    if leading_dim_size == 0:
        raise RuntimeError("Leading dimensions of mapped xs cannot be 0.")

    if any(cur_shape[0] != leading_dim_size for cur_shape in shapes):
        raise RuntimeError(
            f"Leading dimensions of mapped xs must be consistent. Got shapes {shapes}."
        )

    def run_flattened_map(f, flat_xs, flat_args):
        def wrapped_fn(*flat_args, f, xs_tree_spec, args_tree_spec, num_xs):
            xs = pytree.tree_unflatten(flat_args[:num_xs], xs_tree_spec)
            args = pytree.tree_unflatten(flat_args[num_xs:], args_tree_spec)
            return f(xs, *args)

        inner_f = functools.partial(
            wrapped_fn,
            f=f,
            xs_tree_spec=xs_spec,
            args_tree_spec=args_spec,
            num_xs=len(flat_xs),
        )
        return map_impl(inner_f, flat_xs, flat_args)

    from torch._higher_order_ops.utils import _maybe_compile_and_run_fn

    return _maybe_compile_and_run_fn(run_flattened_map, f, flat_xs, flat_args)


class MapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fw_graph, joint_graph, num_mapped_args, *flat_args):
        save_tensors_and_symints_for_backward(ctx, flat_args)
        ctx._joint_graph = joint_graph
        ctx._num_mapped_args = num_mapped_args
        with torch._C._AutoDispatchBelowAutograd():
            return (
                *map_impl(
                    fw_graph, flat_args[:num_mapped_args], flat_args[num_mapped_args:]
                ),
            )

    @staticmethod
    def backward(ctx, *flat_grads):
        fw_args = saved_tensors_and_symints(ctx)
        fw_mapped_args = fw_args[: ctx._num_mapped_args]
        pos_args = fw_args[ctx._num_mapped_args :]

        grads = map_impl(
            ctx._joint_graph,
            fw_mapped_args + flat_grads,
            pos_args,
        )
        return None, None, None, *grads


def trace_map(proxy_mode, func_overload, f, xs, pos_args):
    example_input = _unstack_pytree(xs)[0]
    body_graph = f

    body_graph = reenter_make_fx(body_graph)(*example_input, *pos_args)

    next_name = proxy_mode.tracer.get_fresh_qualname("body_graph_")

    proxy_mode.tracer.root.register_module(next_name, body_graph)

    fake_outs = map_impl(body_graph, xs, pos_args)

    node_args = (body_graph, list(xs), list(pos_args))
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="map_impl"
    )
    return track_tensor_tree(
        fake_outs, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@map_impl.py_impl(DispatchKey.CompositeExplicitAutograd)
def map_dense(f, xs, pos_args):
    pytrees = [f(*inp, *pos_args) for inp in _unstack_pytree(xs)]
    return _stack_pytree(pytrees)


@map_impl.py_autograd_impl
def map_autograd(f, xs, pos_args):
    num_mapped_args = len(xs)
    fw_graph, bw_graph = create_fw_bw_graph(f, num_mapped_args, *xs, *pos_args)
    flat_out = MapAutogradOp.apply(fw_graph, bw_graph, num_mapped_args, *xs, *pos_args)
    return flat_out


@map_impl.py_impl(ProxyTorchDispatchMode)
def map_proxy_torch_dispatch_mode(mode, f, xs, args):
    return trace_map(mode, map_impl, f, xs, args)


@map_impl.py_impl(FakeTensorMode)
def map_fake_tensor_mode(mode, f, xs, args):
    with mode:
        return map_dense(f, xs, args)


@map_impl.py_functionalize_impl
def map_functionalize(ctx, f, xs, pos_args):
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_args = ctx.unwrap_tensors(pos_args)
    wrapped_fn = ctx.functionalize(_maybe_run_with_interpreter(f))

    with ctx.redispatch_to_next():
        example_inputs = (*_unstack_pytree(unwrapped_xs)[0], *unwrapped_args)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        _check_alias_and_mutation(f, example_inputs, "map", pre_dispatch)
        map_return = map_impl(wrapped_fn, unwrapped_xs, unwrapped_args)
        return ctx.wrap_tensors(map_return)


def _fake_map(f, x, *args):
    from functorch.experimental.control_flow import _stack_pytree, _unstack_pytree

    x_pytrees = _unstack_pytree(x)
    zs = []
    for xp in x_pytrees:
        zs.append(f(xp, *args))
    return _stack_pytree(zs)
