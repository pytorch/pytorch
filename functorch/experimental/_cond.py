from dataclasses import dataclass
from functools import partial
import torch
from torch.multiprocessing.reductions import StorageWeakRef

import torch.utils._pytree as pytree

from torch._C import DispatchKey, DispatchKeySet, ExcludeDispatchKeyGuard
from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional, _wrap_all_tensors_to_functional, functionalize
from torch._ops import PyOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    make_fx,
    track_tensor_tree,
    unwrap_proxy,
)
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)
from torch.utils._pytree import tree_flatten


@dataclass
class UnsupportedAliasMutationException(RuntimeError):
    reason: str


"""
We're going to define a `cond` operation.
In order to do this, we need implementations for each of the dispatch keys.
"""
cond = PyOperator("cond")


def trace_cond(proxy_mode, func_overload, pred, true_fn, false_fn, operands):
    assert isinstance(operands, (list, tuple)), "Cond operands must be a list or tuple of tensors"
    assert all(isinstance(o, torch.Tensor) for o in operands), "Cond operands must be a list of tensors"

    with disable_proxy_modes_tracing():
        true_graph = make_fx(true_fn)(*operands)
        false_graph = make_fx(false_fn)(*operands)

    true_outs = []
    false_outs = []
    for node in true_graph.graph.nodes:
        if node.op == 'output':
            true_outs.extend(node.args)

    for node in false_graph.graph.nodes:
        if node.op == 'output':
            false_outs.extend(node.args)

    flat_true_outs, _ = pytree.tree_flatten(true_outs)
    flat_false_outs, _ = pytree.tree_flatten(false_outs)
    assert(len(flat_true_outs) == len(flat_false_outs))

    for i in range(0, len(flat_true_outs)):
        true_out = flat_true_outs[i]
        false_out = flat_false_outs[i]
        assert true_out.meta['tensor_meta'] == false_out.meta['tensor_meta']

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
    assert(not hasattr(proxy_mode.tracer.root, false_name))

    proxy_mode.tracer.root.register_module(true_name, true_graph)
    proxy_mode.tracer.root.register_module(false_name, false_graph)

    args = (pred, true_graph, false_graph, operands)

    proxy_args = pytree.tree_map(partial(unwrap_proxy, proxy_mode), args)

    out_proxy = proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, {},
                                               name="conditional")

    # At this point, we're *guaranteed* that whether an output came from the
    # true or false branch is indistinguishable. So, as this is just for tracing
    # purposes, choose the true branch.

    # TODO: Uhh.... it shouldn't matter, but changing this to true_fn results in
    # a FakeTensorMode error :
    # `Current active mode <class 'torch._subclasses.fake_tensor.FakeTensorMode'> not registered`
    out = false_fn(*operands)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@cond.py_impl(DispatchKey.CUDA)
@cond.py_impl(DispatchKey.CPU)
def cond_dense(pred, true_fn, false_fn, operands):
    mode = _get_current_dispatch_mode()
    assert (mode is None), "Mode should never be enabled for CPU/CUDA key"
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)


@cond.py_impl(DispatchKey.AutogradCUDA)
@cond.py_impl(DispatchKey.AutogradCPU)
def cond_autograd(pred, true_fn, false_fn, *operands):
    # TODO: support autograd
    flat_operands, _ = tree_flatten([true_fn, false_fn] + [operands])
    assert all([not f.requires_grad for f in flat_operands
                if isinstance(f, torch.Tensor)])

    guard = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.AutogradCPU))
    return cond(pred, true_fn, false_fn, *operands)


@cond.py_impl(ProxyTorchDispatchMode)
def inner(pred, true_fn, false_fn, operands):
    mode = _get_current_dispatch_mode()
    assert (mode is not None), "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        res = trace_cond(mode, cond, pred, true_fn, false_fn, operands)
    return res


@cond.py_impl(FakeTensorMode)
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
            raise RuntimeError(
                f"Unmatched tensor metadata from cond() branches.\ntrue branch: {true_meta}, false branch: {false_meta}")
    return true_outs


# We cannot directly call fallthrough here due to issue #89037.
@cond.py_impl(DispatchKey.PythonDispatcher)
def cond_python_dispatcher(*args):
    _ = ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.PythonDispatcher))
    return cond(*args)


def _has_potential_branch_input_mutation(branch, fake_inputs):
    """
    Dispatch-trace the branch with fake inputs and check if
    producing graph has mutable op on the input. This is
    bit restrictive as the branch must be traceable.
    """
    try:
        gm = make_fx(branch)(*fake_inputs)
    except UnsupportedAliasMutationException:
        # this can happen when nested cond is
        # functionalized
        return True
    except Exception as e:
        raise e

    input_nodes = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            input_nodes.add(node)
        if node.op == "call_function":
            target = node.target
            if isinstance(target, torch._ops.OpOverload) and target._schema.is_mutable:
                for arg in node.args:
                    if arg in input_nodes:
                        return True

    return False

def _has_potential_branch_input_alias(branch, fake_inputs):
    """
    Dispatch-trace the branch with fake inputs and check if
    producing graph has output aliasing the branch input. This is
    bit restrictive as the branch must be traceable.
    """
    try:
        gm = make_fx(branch)(*fake_inputs)
    except UnsupportedAliasMutationException:
        # this can happen when nested cond is
        # functionalized
        return True
    except Exception as e:
        raise e

    input_storages = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            input_storages.add(StorageWeakRef(node.meta['val']._typed_storage()))

    outs, _ = pytree.tree_flatten(gm(*fake_inputs))
    for out in outs:
        if isinstance(out, torch.Tensor) and StorageWeakRef(out._typed_storage()) in input_storages:
            return True

    return False



@cond.py_impl(torch._C._functorch.TransformType.Functionalize)
def cond_functionalize(interpreter, pred, true_fn, false_fn, inputs):
    """
    Functionalization implementation for torch.cond. Currently:
      1. We don't allow any input mutation inside the branches
      2. Our check for above condition is not exhaustive
    """
    reapply_views = interpreter.functionalize_add_back_views()
    mode = 'mutations_and_views' if reapply_views else 'mutations'
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_inputs = _unwrap_all_tensors_from_functional(inputs, reapply_views=reapply_views)
    unwrapped_pred = _unwrap_all_tensors_from_functional(pred, reapply_views=reapply_views)

    functional_true_fn = functionalize(true_fn, remove=mode)
    functional_false_fn = functionalize(false_fn, remove=mode)

    with interpreter.lower():
        fake_tensor_mode = FakeTensorMode()
        with fake_tensor_mode as ft_mode:
            for branch in [functional_true_fn, functional_false_fn]:
                def convert(x):
                    return ft_mode.fake_tensor_converter(ft_mode, x)
                fake_inputs = pytree.tree_map_only(torch.Tensor, convert, unwrapped_inputs)
                if _has_potential_branch_input_mutation(branch, fake_inputs):
                    raise UnsupportedAliasMutationException("One of torch.cond branch "
                                                            "might be modifying the input!")
            for branch in [true_fn, false_fn]:
                def convert(x):
                    return ft_mode.fake_tensor_converter(ft_mode, x)
                fake_inputs = pytree.tree_map_only(torch.Tensor, convert, unwrapped_inputs)
                if _has_potential_branch_input_alias(branch, fake_inputs):
                    raise UnsupportedAliasMutationException("One of torch.cond branch "
                                                            "might be aliasing the input!")

        cond_return = cond(unwrapped_pred, functional_true_fn, functional_false_fn, unwrapped_inputs)
        return _wrap_all_tensors_to_functional(cond_return, level=interpreter.level())

# TODO(voz): Make this automatic for keys, this is very ugly atm
cond.fallthrough(DispatchKey.PythonTLSSnapshot)
cond.fallthrough(DispatchKey.ADInplaceOrView)
cond.fallthrough(DispatchKey.BackendSelect)
