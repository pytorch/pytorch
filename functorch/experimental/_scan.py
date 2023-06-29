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
class ScanWrapper(HigherOrderOperator):
    def __call__(self, f, init, xs):
        return scan_wrapper(f, init, xs)

scan = ScanWrapper("scan")
scan_impl = HigherOrderOperator("scan_impl")
scan_backward_impl = HigherOrderOperator("scan_backward_impl")

dummy_aot_config = AOTConfig(fw_compiler=None,
                             bw_compiler=None,
                             partition_fn=None,
                             decompositions={},
                             num_params_buffers=0,
                             aot_id=0,
                             keep_inference_input_mutations=False)


def create_fw_bw_graph(f, flat_init, flat_xs):
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

            example_init = [from_fun(init_element) for init_element in flat_init]
            example_xs = [from_fun(xs) for xs in _unstack_pytree(flat_xs)[0]]
            example_flat_carry_out, example_flat_out = f(example_init, example_xs)
            example_flat_carry_out, example_flat_out = pytree.tree_map(from_fun, example_flat_carry_out), pytree.tree_map(from_fun, example_flat_out)
            if any(not isinstance(out, torch.Tensor) for out in example_flat_out if out is not None):
                raise RuntimeError("Expect outputs of scan only contains tensors or None. "
                                   f"Got types {[type(out) for out in example_flat_out]}.")
            if any(not isinstance(out, torch.Tensor) for out in example_flat_carry_out if out is not None):
                raise RuntimeError("Expect output carry of scan only contains tensors or None. "
                                   f"Got types {[type(out) for out in example_flat_carry_out]}.")
            example_grad_carry_out = [from_fun(out) for out in example_flat_carry_out]
            example_grad_out = [from_fun(out) for out in example_flat_out]

            fw_graph = make_fx(f)(example_init, example_xs)

        def joint_f(example_grad_carry_out, example_xs, example_init, example_grad_out):
            scanned_input = example_xs
            init = example_init
            grad_args = list(example_grad_carry_out) + list(example_grad_out)
            all_input_args = list(example_xs) + list(example_init) + grad_args

            num_carry_args = len(init)

            def fw_with_masks(*args):
                fw_carry_out, fw_out = f(args[:num_carry_args], args[num_carry_args:])
                return list(fw_carry_out) + list(fw_out), [True if isinstance(ret, torch.Tensor) and ret.requires_grad else False for ret in list(fw_carry_out) + list(fw_out)]

            joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)
            _, grads = joint(list(init) + list(scanned_input),
                             [grad for grad in grad_args if grad is not None and grad.requires_grad])

            # In order to keep map functional for backward graph,
            # we clone outputs that are aliasing inputs
            input_storage = {StorageWeakRef(arg._typed_storage()) for arg in all_input_args if isinstance(arg, torch.Tensor)}

            def maybe_clone(t):
                if isinstance(t, torch.Tensor) and StorageWeakRef(t._typed_storage()) in input_storage:
                    return t.clone()
                return t
            # return (carry grad, output grad)
            return pytree.tree_map(maybe_clone, grads[:len(example_grad_carry_out)]), pytree.tree_map(maybe_clone, grads[len(example_grad_carry_out):])

        joint_graph = make_fx(joint_f)(example_grad_carry_out, example_xs, example_init, example_grad_out)
        return fw_graph, joint_graph


def scan_wrapper(f, init, xs):
    flat_init, init_spec = pytree.tree_flatten(init)
    if not all(isinstance(t, torch.Tensor) for t in flat_init):
        raise RuntimeError(f"Scanned init can only consist of tensors. Got init {flat_init}.")
    flat_xs, xs_spec = pytree.tree_flatten(xs)
    if not all(isinstance(t, torch.Tensor) for t in flat_xs):
        raise RuntimeError(f"Scanned xs can only consist of tensors. Got xs {flat_xs}.")
    
    num_init_args = len(flat_init)
    # Is it necessary to restrict the shape of scanned or mapped elements?
    # for scan, the only requirement should be that f spits out a carry that
    # always has the same shape
    shapes = [xs.shape for xs in flat_xs]
    leading_dim_size = shapes[0][0]
    if leading_dim_size == 0:
        raise RuntimeError(
            "Leading dimensions of scanned xs cannot be 0.")

    if any(cur_shape[0] != leading_dim_size for cur_shape in shapes):
        raise RuntimeError(
            f"Leading dimensions of scanned xs must be consistent. Got shapes {shapes}.")

    carry_out_spec = None
    out_spec = None

    def flat_fn(flat_init, flat_xs):
        # carry and init should have the same spec
        carry = pytree.tree_unflatten(flat_init, init_spec)
        xs = pytree.tree_unflatten(flat_xs, xs_spec)
        unflattened_carry_out, unflattened_out = f(carry, xs)
        flat_carry_out, tmp_carry_out_spec = pytree.tree_flatten(unflattened_carry_out)
        flat_out, tmp_out_spec = pytree.tree_flatten(unflattened_out)

        nonlocal carry_out_spec
        nonlocal out_spec
        carry_out_spec = tmp_carry_out_spec
        out_spec = tmp_out_spec
        return flat_carry_out, flat_out
    
    flat_carry_out, flat_out = scan_impl(flat_fn, flat_init, flat_xs, None, None, return_all_carries=False, reverse=False)  # (ys, carry)
    return pytree.tree_unflatten(flat_carry_out, carry_out_spec), pytree.tree_unflatten(flat_out, out_spec)

class ScanAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fw_graph, joint_graph, num_init_args, *flat_args):
        ctx._joint_graph = joint_graph
        ctx._num_init_args = num_init_args
        ctx._num_input_args = len(flat_args)
        with torch._C._AutoDispatchBelowAutograd():
            flat_carry_out, flat_out, carries = scan_impl(fw_graph, flat_args[:num_init_args], flat_args[num_init_args:], None, None, return_all_carries=True, reverse=False)
            # needs to flat the carries nested list
            # second call to save_for_backward will override whatever that is saved earlier
            # therefore, need to save everything that is needed at once
            ctx.save_for_backward(*flat_args, *(carry_entry for carry in carries for carry_entry in carry))
            return (*flat_carry_out, *flat_out)

    @staticmethod
    def backward(ctx, *flat_grads):
        flat_carries = ctx.saved_tensors[ctx._num_input_args:]
        # bring back the nested carries list
        carries = [flat_carries[i*ctx._num_init_args:(i+1)*ctx._num_init_args] for i in range(len(flat_carries)//ctx._num_init_args)]
        fwd_args = ctx.saved_tensors[:ctx._num_input_args]
        flat_xs = fwd_args[ctx._num_init_args:]
        final_carry_grad = flat_grads[:ctx._num_init_args]
        ys_grad = flat_grads[ctx._num_init_args:]
        with torch._C._AutoDispatchBelowAutograd():
            flat_carry_out_grads, flat_out_grads = scan_impl(ctx._joint_graph, final_carry_grad, flat_xs, ys_grad, carries, return_all_carries=False, reverse=True)
            return None, None, None, *flat_carry_out_grads, *flat_out_grads

# def trace_map(proxy_mode, func_overload, f, num_mapped, *args):
#     xs = list(args[:num_mapped])
#     pos_args = list(args[num_mapped:])
#     leading_dim_size = xs[0].shape[0]

#     example_input = _unstack_pytree(xs)[0]
#     body_graph = f
#     if not isinstance(body_graph, torch.fx.GraphModule):
#         body_graph = make_fx(body_graph)(*example_input, *pos_args)

#     with disable_proxy_modes_tracing():
#         example_outs = body_graph(*example_input, *pos_args)

#         def expand_tensor(t):
#             if isinstance(t, torch.Tensor):
#                 return t.expand(leading_dim_size, *t.shape)
#             return t
#         expanded_outs = pytree.tree_map(expand_tensor, example_outs)

#     next_name = None
#     i = 0
#     while not next_name:
#         candidate = f"body_graph_{i}"
#         if hasattr(proxy_mode.tracer.root, candidate):
#             i += 1
#         else:
#             next_name = candidate

#     proxy_mode.tracer.root.register_module(next_name, body_graph)
#     node_args = (body_graph, num_mapped, *args)
#     proxy_args = pytree.tree_map(partial(unwrap_proxy, proxy_mode), node_args)
#     out_proxy = proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, {},
#                                                name="map_impl")
#     return track_tensor_tree(expanded_outs, out_proxy, constant=None, tracer=proxy_mode.tracer)

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

@scan_impl.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_dense(f, flat_init, flat_xs, ys_grad=None, carries=None, return_all_carries=False, reverse=False):
    carry = flat_init
    # each carry output is needed for calculating gradients backwards
    out_carries = []
    out_pytrees = []
    direction = -1 if reverse else 1
    xs_unstacked = _unstack_pytree(flat_xs)[::direction]
    iterable_input = zip(xs_unstacked)
    if carries is not None and ys_grad is not None:
        # suppose both are not none, that means we are reusing the scan_dense for gradient computation in bwd pass
        # in this case, the flat_init is actually the gradients on final carry output from fwd pass
        ys_grad_unstacked = _unstack_pytree(ys_grad)[::direction]
        iterable_input = zip(xs_unstacked, carries[::direction], ys_grad_unstacked)
    for inp in iterable_input:
        # saves the initial carry input for each iteration
        out_carries.append(carry)
        carry, flattened_out = f(carry, *inp)
        out_pytrees.append(flattened_out)
    ys = _stack_pytree(out_pytrees[::direction])
    if return_all_carries:
        return carry, ys, out_carries
    return carry, ys

@scan_impl.py_impl(DispatchKey.Autograd)
def scan_autograd(f, flat_init, flat_xs, ys_grad=None, carries=None, return_all_carries=False, reverse=False):
    fw_graph, bw_graph = create_fw_bw_graph(f, flat_init, flat_xs)
    flat_all_out = ScanAutogradOp.apply(fw_graph, bw_graph, len(flat_init), *flat_init, *flat_xs)
    num_carry_args = len(flat_xs)
    return flat_all_out[:num_carry_args], flat_all_out[num_carry_args:]

# TODO(voz) Make this automatic for keys, this is very ugly atm
scan_impl.fallthrough(DispatchKey.PythonDispatcher)
scan_impl.fallthrough(DispatchKey.PythonTLSSnapshot)
scan_impl.fallthrough(DispatchKey.ADInplaceOrView)
scan_impl.fallthrough(DispatchKey.BackendSelect)
scan_impl.fallthrough(DispatchKey.AutocastCPU)

@scan_backward_impl.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_backward_dense(f, flat_init, flat_xs, flat_grads):
    pytrees = []
    carry = flat_init
    for inp in _unstack_pytree(flat_xs):
        carry, flattened_out = f(carry, inp)
        pytrees.append(flattened_out)
    return carry, _stack_pytree(pytrees)

scan_backward_impl.fallthrough(DispatchKey.PythonDispatcher)
scan_backward_impl.fallthrough(DispatchKey.PythonTLSSnapshot)
scan_backward_impl.fallthrough(DispatchKey.ADInplaceOrView)
scan_backward_impl.fallthrough(DispatchKey.BackendSelect)
scan_backward_impl.fallthrough(DispatchKey.AutocastCPU)