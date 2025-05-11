# mypy: allow-untyped-defs
import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _maybe_run_with_interpreter,
    reenter_make_fx,
    UnsupportedAliasMutationException,
)
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
    create_bw_fn,
    materialize_as_graph,
    prepare_fw_with_masks,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
    split_into_chunks,
)


# TODO: We add this to prevent dymamo from tracing into map_wrapper,
# remove the wrapper call when it's ready.
class MapWrapper(HigherOrderOperator):
    def __init__(self):
        super().__init__("map")

    def __call__(self, xs, *args):
        return map_wrapper(xs, *args)


class MapImpl(HigherOrderOperator):
    def __init__(self):
        super().__init__("map_impl")

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


map = MapWrapper()

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


def map_wrapper(f, xs, *args):
    flat_xs, xs_spec = pytree.tree_flatten(xs)
    if not all(isinstance(t, torch.Tensor) for t in flat_xs):
        raise RuntimeError(f"Mapped xs can only consist of tensors. Got xs {flat_xs}.")

    num_mapped_args = len(flat_xs)
    shapes = [xs.shape for xs in flat_xs]
    leading_dim_size = shapes[0][0]
    if leading_dim_size == 0:
        raise RuntimeError("Leading dimensions of mapped xs cannot be 0.")

    if any(cur_shape[0] != leading_dim_size for cur_shape in shapes):
        raise RuntimeError(
            f"Leading dimensions of mapped xs must be consistent. Got shapes {shapes}."
        )

    out_spec = None

    def flat_fn(*flat_args):
        xs = pytree.tree_unflatten(list(flat_args[:num_mapped_args]), xs_spec)
        unflattened_out = f(xs, *flat_args[num_mapped_args:])
        flat_out, tmp_out_spec = pytree.tree_flatten(unflattened_out)

        if any(
            not isinstance(out, torch.Tensor) for out in flat_out if out is not None
        ):
            raise RuntimeError(
                "Expect outputs of map only contains tensors or None. "
                f"Got types {[type(out) for out in flat_out]}."
            )

        nonlocal out_spec
        out_spec = tmp_out_spec
        return flat_out

    return pytree.tree_unflatten(
        map_impl(flat_fn, flat_xs, args), out_spec  # type: ignore[arg-type]
    )


class MapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, num_mapped_args, *flat_args):
        ctx._f = f
        ctx._num_mapped_args = num_mapped_args
        ctx._num_pos_args = len(flat_args) - num_mapped_args

        # We snapshot the dispatch keys in forward for materializing the
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()
        save_tensors_and_symints_for_backward(ctx, flat_args)
        with torch._C._AutoDispatchBelowAutograd():
            return (
                *map_impl(f, flat_args[:num_mapped_args], flat_args[num_mapped_args:]),
            )

    @staticmethod
    def backward(ctx, *flat_grads):
        fw_args = saved_tensors_and_symints(ctx)
        fw_mapped_args, pos_args = split_into_chunks(
            fw_args, [ctx._num_mapped_args, ctx._num_pos_args]
        )
        num_mapped_args = ctx._num_mapped_args
        num_pos_args = ctx._num_pos_args
        num_grads = len(flat_grads)

        ctx._bw_f = create_bw_fn(ctx._f, fw_args)

        # Create a wrapper around thefor the bw_f
        def bw_f_wrapper(*args):
            # Dissect args and re-order them for the ``ctx._bw_f``
            # args provided to the wrapper are composed of [*fw_mapped_args, *flat_grads, *pos_args]
            # The content of ``bw_f_tangents`` are the upstream gradients, i.e. flat_grads
            # The content of ``bw_f_primals`` are the fw_args, i.e., [*fw_mapped_args, *pos_args]
            # The bw_f requires *bw_f_primals, *bw_f_tangents
            fw_m_args, bw_f_tangents, pos_args = split_into_chunks(
                args, [num_mapped_args, num_grads, num_pos_args]
            )
            bw_f_primals = [*fw_m_args, *pos_args]
            return ctx._bw_f(*bw_f_primals, *bw_f_tangents)

        def construct_args_single_step_bw():
            fw_mapped_args_slice = _unstack_pytree(fw_mapped_args)[0]
            flat_grads_slice = _unstack_pytree(flat_grads)[0]
            return *fw_mapped_args_slice, *flat_grads_slice, *pos_args

        args_single_step_bw = construct_args_single_step_bw()

        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint funcion when torch.compile torch.autograd.grad.
        fn_bw_gm = materialize_as_graph(
            bw_f_wrapper,
            args_single_step_bw,
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )

        grads = map_impl(fn_bw_gm, fw_mapped_args + list(flat_grads), pos_args)

        return None, None, *grads


def trace_map(proxy_mode, func_overload, f, xs, pos_args):
    leading_dim_size = xs[0].shape[0]

    example_input = _unstack_pytree(xs)[0]
    body_graph = f

    body_graph = reenter_make_fx(body_graph)(*example_input, *pos_args)

    next_name = proxy_mode.tracer.get_fresh_qualname("body_graph_")

    proxy_mode.tracer.root.register_module(next_name, body_graph)

    with disable_proxy_modes_tracing():
        example_outs = body_graph(*example_input, *pos_args)

        def expand_tensor(t):
            if isinstance(t, torch.Tensor):
                return t.expand(leading_dim_size, *t.shape)
            return t

        expanded_outs = pytree.tree_map(expand_tensor, example_outs)

    node_args = (body_graph, list(xs), list(pos_args))
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="map_impl"
    )
    return track_tensor_tree(
        expanded_outs, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@map_impl.py_impl(DispatchKey.CompositeExplicitAutograd)
def map_dense(f, xs, pos_args):
    pytrees = [f(*inp, *pos_args) for inp in _unstack_pytree(xs)]
    return _stack_pytree(pytrees)


@map_impl.py_autograd_impl
def map_autograd(f, xs, pos_args):
    num_mapped_args = len(xs)
    flat_out = MapAutogradOp.apply(f, num_mapped_args, *xs, *pos_args)
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
    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_args = ctx.unwrap_tensors(pos_args)
    wrapped_fn = ctx.functionalize(_maybe_run_with_interpreter(f))

    with ctx.redispatch_to_next():
        with disable_proxy_modes_tracing():
            example_inputs = (*_unstack_pytree(unwrapped_xs)[0], *unwrapped_args)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        if _has_potential_branch_input_mutation(
            f, example_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException("torch.map is mutating the input!")

        if _has_potential_branch_input_alias(
            f, example_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException("torch.map is aliasing the input!")

        map_return = map_impl(wrapped_fn, unwrapped_xs, unwrapped_args)
        return ctx.wrap_tensors(map_return)
