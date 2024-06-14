# mypy: allow-untyped-defs
import os.path as _osp
import torch

from .throughput_benchmark import ThroughputBenchmark
from .cpp_backtrace import get_cpp_backtrace
from .backend_registration import rename_privateuse1_backend, generate_methods_for_privateuse1_backend
from . import deterministic
from . import collect_env
import weakref
import copyreg

def set_module(obj, mod):
    """
    Set the module attribute on a python object for a given object for nicer printing
    """
    if not isinstance(mod, str):
        raise TypeError("The mod argument should be a string")
    obj.__module__ = mod

if torch._running_with_deploy():
    # not valid inside torch_deploy interpreter, no paths exists for frozen modules
    cmake_prefix_path = None
else:
    cmake_prefix_path = _osp.join(_osp.dirname(_osp.dirname(__file__)), 'share', 'cmake')

def swap_tensors(t1, t2):
    """
    This function swaps the content of the two Tensor objects.
    At a high level, this will make t1 have the content of t2 while preserving
    its identity.

    This will not work if t1 and t2 have different slots.
    """
    # Ensure there are no weakrefs
    if weakref.getweakrefs(t1):
        raise RuntimeError("Cannot swap t1 because it has weakref associated with it")
    if weakref.getweakrefs(t2):
        raise RuntimeError("Cannot swap t2 because it has weakref associated with it")
    t1_slots = set(copyreg._slotnames(t1.__class__))  # type: ignore[attr-defined]
    t2_slots = set(copyreg._slotnames(t2.__class__))  # type: ignore[attr-defined]
    if t1_slots != t2_slots:
        raise RuntimeError("Cannot swap t1 and t2 if they have different slots")

    def swap_attr(name):
        tmp = getattr(t1, name)
        setattr(t1, name, (getattr(t2, name)))
        setattr(t2, name, tmp)

    def error_pre_hook(grad_outputs):
        raise RuntimeError("Trying to execute AccumulateGrad node that was poisoned by swap_tensors "
                           "this can happen when you try to run backward on a tensor that was swapped. "
                           "For a module m with `torch.__future__.set_swap_module_params_on_conversion(True)` "
                           "you should not change the device or dtype of the module (e.g. `m.cpu()` or `m.half()`) "
                           "between running forward and backward. To resolve this, please only change the "
                           "device/dtype before running forward (or after both forward and backward).")

    def check_use_count(t, name='t1'):
        use_count = t._use_count()
        error_str = (f"Expected use_count of {name} to be 1 or 2 with an AccumulateGrad node but got {use_count} "
                     f"make sure you are not holding references to the tensor in other places.")
        if use_count > 1:
            if use_count == 2 and t.is_leaf:
                accum_grad_node = torch.autograd.graph.get_gradient_edge(t).node
                # Make sure that the accumulate_grad node was not lazy_init-ed by get_gradient_edge
                if t._use_count() == 2:
                    accum_grad_node.register_prehook(error_pre_hook)
                else:
                    raise RuntimeError(error_str)
            else:
                raise RuntimeError(error_str)

    check_use_count(t1, 't1')
    check_use_count(t2, 't2')

    # Swap the types
    # Note that this will fail if there are mismatched slots
    swap_attr("__class__")

    # Swap the dynamic attributes
    swap_attr("__dict__")

    # Swap the slots
    for slot in t1_slots:
        if hasattr(t1, slot) and hasattr(t2, slot):
            swap_attr(slot)
        elif hasattr(t1, slot):
            setattr(t2, slot, (getattr(t1, slot)))
            delattr(t1, slot)
        elif hasattr(t2, slot):
            setattr(t1, slot, (getattr(t2, slot)))
            delattr(t2, slot)

    # Swap the at::Tensor they point to
    torch._C._swap_tensor_impl(t1, t2)
