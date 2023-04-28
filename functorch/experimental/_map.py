from functools import partial

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional, _wrap_all_tensors_to_functional, functionalize
from torch._functorch.aot_autograd import create_joint
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
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
from ._cond import _has_potential_branch_input_alias, _has_potential_branch_input_mutation, UnsupportedAliasMutationException


# TODO: We add this to prevent dymamo tracing into map_wrapper,
# remove the wrapper call when it's ready.
class MapWrapper(HigherOrderOperator):
    def __call__(self, *args, **kwargs):
        return map_wrapper(*args, **kwargs)

map = MapWrapper("map")
map_impl = HigherOrderOperator("map_impl")

def map_wrapper(f, xs, *args):
    flat_xs, xs_spec = pytree.tree_flatten(xs)
    assert all(isinstance(t, torch.Tensor) for t in flat_xs), f"mapped xs can only contain tensors got {flat_xs}"

    num_mapped_args = len(flat_xs)
    assert num_mapped_args > 0, "map must have at least one mapped argument."

    shapes = [xs.shape for xs in flat_xs]
    leading_dim_size = shapes[0][0]
    assert len(shapes) > 0 and all(cur_shape[0] == leading_dim_size for cur_shape in shapes)

    out_spec = [None]

    def flat_fn(num_mapped_args, *flat_args):
        xs = pytree.tree_unflatten(flat_args[:num_mapped_args], xs_spec)
        unflattened_out = f(xs, *flat_args[num_mapped_args:])
        flat_out, tmp_out_spec = pytree.tree_flatten(unflattened_out)
        out_spec[0] = tmp_out_spec
        return flat_out
    return pytree.tree_unflatten(map_impl(flat_fn, num_mapped_args, *flat_xs, *args), out_spec[0])

class MapAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fw_graph, joint_graph, num_mapped_args, *flat_args):
        ctx.save_for_backward(*flat_args)
        ctx._joint_graph = joint_graph
        ctx._num_mapped_args = num_mapped_args
        _ = torch._C._AutoDispatchBelowAutograd()
        return (*map_impl(fw_graph, num_mapped_args, *flat_args), )

    @staticmethod
    def backward(ctx, *flat_grads):
        fw_args = ctx.saved_tensors
        fw_mapped_args = fw_args[:ctx._num_mapped_args]
        pos_args = fw_args[ctx._num_mapped_args:]
        mapped_grads = [grad for grad in flat_grads if grad is not None]

        _ = torch._C._AutoDispatchBelowAutograd()
        grads = map_impl(ctx._joint_graph, ctx._num_mapped_args + len(mapped_grads), *fw_mapped_args, *mapped_grads, *pos_args)
        return None, None, *grads

def trace_map(proxy_mode, func_overload, f, num_mapped, *args):
    xs = list(args[:num_mapped])
    pos_args = list(args[num_mapped:])
    leading_dim_size = xs[0].shape[0]

    example_input = _unstack_pytree(xs)[0]
    body_graph = f
    if not isinstance(body_graph, torch.fx.GraphModule):
        body_graph = make_fx(body_graph)(num_mapped, *example_input, *pos_args)

    with disable_proxy_modes_tracing():
        example_outs = body_graph(num_mapped, *example_input, *pos_args)

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
    assert all(isinstance(xs, torch.Tensor) for xs in flat_xs), f"Leaves of xs must be Tensor {flat_xs}"
    assert all(xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs), \
        f"Leaves of xs must have same leading dimension size {[xs.shape for xs in flat_xs]}"
    a = list(zip(*flat_xs))
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
    b = list(zip(*flat_out))
    stacked_out = []
    for leaves in b:
        if all(leave is not None for leave in leaves):
            stacked_out.append(torch.stack(leaves))
        else:
            stacked_out.append(None)
    return pytree.tree_unflatten(stacked_out, out_spec)

@map_impl.py_impl(DispatchKey.CUDA)
@map_impl.py_impl(DispatchKey.CPU)
def map_dense(f, num_mapped_args, *args):
    mode = _get_current_dispatch_mode()
    xs = args[:num_mapped_args]
    pos_args = args[num_mapped_args:]
    assert (mode is None), "Mode should never be enabled for CPU/CUDA keyOne of the differentiated Tensors"
    pytrees = []
    for inp in _unstack_pytree(xs):
        pytrees.append(f(num_mapped_args, *inp, *pos_args))
    return _stack_pytree(pytrees)


@map_impl.py_impl(DispatchKey.Autograd)
def map_autograd(f, num_mapped_args, *args):
    mapped_xs = args[:num_mapped_args]
    pos_args = args[num_mapped_args:]

    with disable_proxy_modes_tracing():
        xs_slice = _unstack_pytree(mapped_xs)[0]
        example_args = (*xs_slice, *pos_args)
        example_xs = example_args[:num_mapped_args]
        example_pos_args = example_args[num_mapped_args:]
        example_flat_out, _ = pytree.tree_flatten(f(num_mapped_args, *example_xs, *example_pos_args))
        example_grad = [torch.ones_like(out) for out in example_flat_out if out is not None and out.requires_grad]

    fw_graph = make_fx(f)(num_mapped_args, *example_xs, *example_pos_args)

    def joint_f(num_mapped, *example_args):
        joint_mapped_args = example_args[:num_mapped]
        args = example_args[num_mapped:]

        mapped_input = joint_mapped_args[:num_mapped_args]
        mapped_grads = joint_mapped_args[num_mapped_args:]

        def fw_with_masks(num_mapped_args, *args):
            fw_out = f(num_mapped_args, *args)
            return fw_out, [True if isinstance(ret, torch.Tensor) and ret.requires_grad else False for ret in fw_out]

        joint = create_joint(fw_with_masks)
        _, grads = joint([num_mapped_args] + list(mapped_input) + list(args), list(mapped_grads))
        return grads

    joint_num_mapped = len(example_grad) + len(example_xs)
    joint_graph = make_fx(joint_f)(joint_num_mapped, *example_xs, *example_grad, *example_pos_args)
    flat_out = MapAutogradOp.apply(fw_graph, joint_graph, num_mapped_args, *args)
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
    xs = args[:num_mapped]
    pos_args = args[num_mapped:]
    leading_dims = pytree.tree_map(lambda t: t.shape[0], xs)
    xs_pytree = _unstack_pytree(xs)
    example_out = f(num_mapped, *xs_pytree[0], *pos_args)
    return pytree.tree_map(lambda t: t.expand(leading_dims[0], *t.shape), example_out)


@map_impl.py_impl(DispatchKey.Functionalize)
def map_func(f, num_mapped, *args):
    reapply_views = torch._C._functionalization_reapply_views_tls()
    xs = args[:num_mapped]
    pos_args = args[num_mapped:]
    unwrapped_xs = _unwrap_all_tensors_from_functional(xs, reapply_views=reapply_views)
    unwrapped_args = _unwrap_all_tensors_from_functional(pos_args, reapply_views=reapply_views)
    mode = 'mutations_and_views' if reapply_views else 'mutations'

    guard = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize))
    try:
        functional_map_fn = functionalize(f, remove=mode)
        inputs = (num_mapped, *unwrapped_xs, *unwrapped_args)

        if _has_potential_branch_input_mutation(f, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is mutating the input!"
            )

        if _has_potential_branch_input_alias(f, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is aliasing the input!"
            )

        map_return = map_impl(functional_map_fn, *inputs)
        return _wrap_all_tensors_to_functional(map_return, level=0)
    finally:
        del guard

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
        inputs = (num_mapped, *unwrapped_xs, *unwrapped_args)
        if _has_potential_branch_input_mutation(functional_map_fn, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is mutating the input!"
            )

        if _has_potential_branch_input_alias(functional_map_fn, inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is aliasing the input!"
            )

        map_return = map_impl(functional_map_fn, *inputs)
        return _wrap_all_tensors_to_functional(map_return, level=interpreter.level())

# TODO(voz) Make this automatic for keys, this is very ugly atm
map_impl.fallthrough(DispatchKey.PythonDispatcher)
map_impl.fallthrough(DispatchKey.PythonTLSSnapshot)
map_impl.fallthrough(DispatchKey.ADInplaceOrView)
map_impl.fallthrough(DispatchKey.BackendSelect)
map_impl.fallthrough(DispatchKey.AutocastCPU)
