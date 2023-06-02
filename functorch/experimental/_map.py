from functools import partial

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey, DispatchKeySet, _ExcludeDispatchKeyGuard
from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional, _wrap_all_tensors_to_functional, functionalize
from torch._functorch.aot_autograd import create_joint, AOTConfig
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.multiprocessing.reductions import StorageWeakRef
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
    unwrap_proxy,
)
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)
from torch._dispatch.python import suspend_functionalization
from ._cond import _has_potential_branch_input_alias, _has_potential_branch_input_mutation, UnsupportedAliasMutationException


# TODO: We add this to prevent dymamo from tracing into map_wrapper,
# remove the wrapper call when it's ready.
class MapWrapper(HigherOrderOperator):
    def __call__(self, xs, *args):
        return map_wrapper(xs, *args)

map = MapWrapper("map")
map_impl = HigherOrderOperator("map_impl")

dummy_aot_config = AOTConfig(fw_compiler=None,
                             bw_compiler=None,
                             partition_fn=None,
                             decompositions={},
                             num_params_buffers=0,
                             aot_id=0,
                             keep_inference_input_mutations=False)


def create_fw_bw_graph(f, num_mapped_args, *args):
    mapped_xs = args[:num_mapped_args]
    pos_args = args[num_mapped_args:]

    # Note: We create "clean" environments for make_fx by suspending all dispatch keys
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

    with suspend_functionalization():
        with disable_proxy_modes_tracing():
            def from_fun(t):
                if isinstance(t, torch.Tensor):
                    return torch.empty_strided(t.size(), t.stride(), requires_grad=t.requires_grad)
                return t

            example_xs = [from_fun(xs) for xs in _unstack_pytree(mapped_xs)[0]]
            example_pos_args = [from_fun(arg) if isinstance(arg, torch.Tensor) else arg for arg in pos_args]
            example_flat_out = pytree.tree_map(from_fun, f(*example_xs, *example_pos_args))
            if any(not isinstance(out, torch.Tensor) for out in example_flat_out if out is not None):
                raise RuntimeError("Expect outputs of map only contains tensors or None. "
                                   f"Got types {[type(out) for out in example_flat_out]}.")
            example_grad = [from_fun(out) for out in example_flat_out]


            fw_graph = make_fx(f)(*example_xs, *example_pos_args)

        def joint_f(*example_args):
            joint_mapped_args = example_args[:joint_num_mapped]
            args = example_args[joint_num_mapped:]

            mapped_input = joint_mapped_args[:num_mapped_args]
            mapped_grads = joint_mapped_args[num_mapped_args:]

            def fw_with_masks(*args):
                fw_out = f(*args)
                return fw_out, [True if isinstance(ret, torch.Tensor) and ret.requires_grad else False for ret in fw_out]

            joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)
            _, grads = joint(list(mapped_input) + list(args),
                             [grad for grad in mapped_grads if grad is not None and grad.requires_grad])

            # In order to keep map functional for backward graph,
            # we clone outputs that are aliasing inputs
            input_storage = {StorageWeakRef(arg._typed_storage()) for arg in example_args if isinstance(arg, torch.Tensor)}

            def maybe_clone(t):
                if isinstance(t, torch.Tensor) and StorageWeakRef(t._typed_storage()) in input_storage:
                    return t.clone()
                return t
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
        raise RuntimeError(
            "Leading dimensions of mapped xs cannot be 0.")

    if any(cur_shape[0] != leading_dim_size for cur_shape in shapes):
        raise RuntimeError(
            f"Leading dimensions of mapped xs must be consistent. Got shapes {shapes}.")

    out_spec = None

    def flat_fn(*flat_args):
        xs = pytree.tree_unflatten(flat_args[:num_mapped_args], xs_spec)
        unflattened_out = f(xs, *flat_args[num_mapped_args:])
        flat_out, tmp_out_spec = pytree.tree_flatten(unflattened_out)

        nonlocal out_spec
        out_spec = tmp_out_spec
        return flat_out
    return pytree.tree_unflatten(map_impl(flat_fn, num_mapped_args, *flat_xs, *args), out_spec)

class MapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fw_graph, joint_graph, num_mapped_args, *flat_args):
        ctx.save_for_backward(*flat_args)
        ctx._joint_graph = joint_graph
        ctx._num_mapped_args = num_mapped_args
        with torch._C._AutoDispatchBelowAutograd():
            return (*map_impl(fw_graph, num_mapped_args, *flat_args), )

    @staticmethod
    def backward(ctx, *flat_grads):
        fw_args = ctx.saved_tensors
        fw_mapped_args = fw_args[:ctx._num_mapped_args]
        pos_args = fw_args[ctx._num_mapped_args:]

        grads = map_impl(ctx._joint_graph, ctx._num_mapped_args + len(flat_grads), *fw_mapped_args, *flat_grads, *pos_args)
        return None, None, None, *grads

def trace_map(proxy_mode, func_overload, f, num_mapped, *args):
    xs = list(args[:num_mapped])
    pos_args = list(args[num_mapped:])
    leading_dim_size = xs[0].shape[0]

    example_input = _unstack_pytree(xs)[0]
    body_graph = f
    if not isinstance(body_graph, torch.fx.GraphModule):
        body_graph = make_fx(body_graph)(*example_input, *pos_args)

    with disable_proxy_modes_tracing():
        example_outs = body_graph(*example_input, *pos_args)

        def expand_tensor(t):
            if isinstance(t, torch.Tensor):
                return t.expand(leading_dim_size, *t.shape)
            return t
        expanded_outs = pytree.tree_map(expand_tensor, example_outs)

    next_name = None
    i = 0
    while not next_name:
        candidate = f"body_graph_{i}"
        if hasattr(proxy_mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate

    proxy_mode.tracer.root.register_module(next_name, body_graph)
    node_args = (body_graph, num_mapped, *args)
    proxy_args = pytree.tree_map(partial(unwrap_proxy, proxy_mode), node_args)
    out_proxy = proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, {},
                                               name="map_impl")
    return track_tensor_tree(expanded_outs, out_proxy, constant=None, tracer=proxy_mode.tracer)

def _unstack_pytree(xs):
    flat_xs, inspec = pytree.tree_flatten(xs)
    if not all(isinstance(xs, torch.Tensor) for xs in flat_xs):
        raise RuntimeError(f"Leaves of xs must be Tensor {flat_xs}")

    if not all(xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs):
        raise RuntimeError(f"Leaves of xs must have same leading dimension size {[xs.shape for xs in flat_xs]}")

    a = zip(*flat_xs)
    pytrees = []
    for tuple in a:
        pytrees.append(pytree.tree_unflatten(tuple, inspec))
    return pytrees

def _stack_pytree(pytrees):
    flat_out = []
    out_spec = None
    for pt in pytrees:
        flat_pt, out_spec = pytree.tree_flatten(pt)
        flat_out.append(flat_pt)
    b = zip(*flat_out)
    stacked_out = []
    for leaves in b:
        if all(isinstance(leaf, torch.Tensor) for leaf in leaves):
            stacked_out.append(torch.stack(leaves))
        elif all(leaf is None for leaf in leaves):
            # Backward graph can return None output when forward inputs doesn't require grad.
            # When we eagerly execute backward graph, we need to call _stack_pytree on its output,
            # therefore we need to deal with None output.
            stacked_out.append(None)
        else:
            raise RuntimeError(f"Cannot stack {leaves}.")
    return pytree.tree_unflatten(stacked_out, out_spec)

@map_impl.py_impl(DispatchKey.CompositeExplicitAutograd)
def map_dense(f, num_mapped_args, *args):
    xs = args[:num_mapped_args]
    pos_args = args[num_mapped_args:]
    pytrees = []
    for inp in _unstack_pytree(xs):
        pytrees.append(f(*inp, *pos_args))
    return _stack_pytree(pytrees)


@map_impl.py_impl(DispatchKey.Autograd)
def map_autograd(f, num_mapped_args, *args):
    fw_graph, bw_graph = create_fw_bw_graph(f, num_mapped_args, *args)
    flat_out = MapAutogradOp.apply(fw_graph, bw_graph, num_mapped_args, *args)
    return flat_out


@map_impl.py_impl(ProxyTorchDispatchMode)
def map_proxy_torch_dispatch_mode(f, num_mapped, *args):
    mode = _get_current_dispatch_mode()
    assert (mode is not None), "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        if mode.enable_tracing:
            return trace_map(mode, map_impl, f, num_mapped, *args)
        else:
            return map_impl(f, num_mapped, *args)


@map_impl.py_impl(FakeTensorMode)
def map_fake_tensor_mode(f, num_mapped, *args):
    return map_dense(f, num_mapped, *args)


@map_impl.py_impl(DispatchKey.Functionalize)
def map_func(f, num_mapped, *args):
    reapply_views = torch._C._functionalization_reapply_views_tls()
    xs = args[:num_mapped]
    pos_args = args[num_mapped:]
    unwrapped_xs = _unwrap_all_tensors_from_functional(xs, reapply_views=reapply_views)
    unwrapped_args = _unwrap_all_tensors_from_functional(pos_args, reapply_views=reapply_views)
    mode = 'mutations_and_views' if reapply_views else 'mutations'

    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        functional_map_fn = functionalize(f, remove=mode)
        with disable_proxy_modes_tracing():
            example_inputs = (*_unstack_pytree(unwrapped_xs)[0], *unwrapped_args)

        if _has_potential_branch_input_mutation(f, example_inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is mutating the input!"
            )

        if _has_potential_branch_input_alias(f, example_inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is aliasing the input!"
            )

        map_return = map_impl(functional_map_fn, num_mapped, *unwrapped_xs, *unwrapped_args)
        return _wrap_all_tensors_to_functional(map_return, level=0)

@map_impl.py_impl(torch._C._functorch.TransformType.Functionalize)
def map_functionalize(interpreter, f, num_mapped, *args):
    """
    Functionalization implementation for torch.map. Currently:
      1. We don't allow any input mutation inside the map function
      2. Our check for above condition is not exhaustive
    """
    xs = args[:num_mapped]
    pos_args = args[num_mapped:]
    reapply_views = interpreter.functionalize_add_back_views()
    mode = 'mutations_and_views' if reapply_views else 'mutations'
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_xs = _unwrap_all_tensors_from_functional(xs, reapply_views=reapply_views)
    unwrapped_args = _unwrap_all_tensors_from_functional(pos_args, reapply_views=reapply_views)

    functional_map_fn = functionalize(f, remove=mode)

    with interpreter.lower():
        with disable_proxy_modes_tracing():
            example_inputs = (*_unstack_pytree(unwrapped_xs)[0], *unwrapped_args)
        if _has_potential_branch_input_mutation(f, example_inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is mutating the input!"
            )

        if _has_potential_branch_input_alias(f, example_inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is aliasing the input!"
            )

        map_return = map_impl(functional_map_fn, num_mapped, *unwrapped_xs, *unwrapped_args)
        return _wrap_all_tensors_to_functional(map_return, level=interpreter.level())

# TODO(voz) Make this automatic for keys, this is very ugly atm
map_impl.fallthrough(DispatchKey.PythonDispatcher)
map_impl.fallthrough(DispatchKey.PythonTLSSnapshot)
map_impl.fallthrough(DispatchKey.ADInplaceOrView)
map_impl.fallthrough(DispatchKey.BackendSelect)
map_impl.fallthrough(DispatchKey.AutocastCPU)
