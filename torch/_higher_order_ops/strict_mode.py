from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch._subclasses.functional_tensor
import torch.fx.traceback as fx_traceback

import torch.utils._pytree as pytree

from torch._C import DispatchKey
from torch._functorch.utils import exposed_in

from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import _get_current_dispatch_mode, _disable_current_modes


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

@exposed_in("torch")
def strict_mode(callable, operands):
    if torch._dynamo.is_compiling():
        return strict_mode_op(callable, operands)

    with _disable_current_modes():
        def func(x):
            return FunctionalTensor.from_functional(x)

        operands_not_functional = pytree.tree_map_only(FunctionalTensor, func, operands)
        with _set_compilation_env():
            gm = torch.export.export(callable, operands_not_functional).module()
    return gm(*operands)


strict_mode_op = HigherOrderOperator("strict_mode")

@strict_mode_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def strict_mode_op_dense(callable, operands):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return callable(*operands)

strict_mode_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(strict_mode_op, deferred_error=True)
)

@strict_mode_op.py_impl(FakeTensorMode)
def strict_mode_fake_tensor_mode(mode, callable, operands):
    with mode:
        true_outs = callable(*operands)
    return true_outs
