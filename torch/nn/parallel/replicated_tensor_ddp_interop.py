import torch
from torch.distributed._shard.replicated_tensor import ReplicatedTensor
from contextlib import contextmanager

_DDP_WITH_REPLICATED_TENSOR = False

@contextmanager
def _ddp_replicated_tensor(val):
    global _DDP_WITH_REPLICATED_TENSOR
    old_val = _DDP_WITH_REPLICATED_TENSOR
    _DDP_WITH_REPLICATED_TENSOR = val
    try:
        yield
    finally:
        _DDP_WITH_REPLICATED_TENSOR = old_val

def _ddp_with_replicated_tensor_enabled():
    global _DDP_WITH_REPLICATED_TENSOR
    return _DDP_WITH_REPLICATED_TENSOR

def _set_ddp_with_replicated_tensor(value):
    global _DDP_WITH_REPLICATED_TENSOR
    _DDP_WITH_REPLICATED_TENSOR = value

class ReplicatedTensorFunction(torch.autograd.Function):
    """
    Autograd function to ensure gradients are replicated between the
    parameterized tensor and the original one.
    """
    @staticmethod
    def forward(ctx, inp, process_group=None):
        # set_materialize_grads(False) will ensure that None gradients stay as
        # None and are not filled with zeros.
        ctx.set_materialize_grads(False)
        return ReplicatedTensor(inp, process_group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def _make_replicated_tensor(tensor, process_group):
    replicated_tensor = ReplicatedTensorFunction.apply(tensor, process_group)
    replicated_tensor.grad = tensor.grad
    return replicated_tensor

def _replicate_module_recurse(module, process_group):
    replica = module._replicate_for_data_parallel()
    for param_name, param in module._parameters.items():
        if param is not None:
            setattr(replica, param_name, _make_replicated_tensor(param, process_group))
        else:
            setattr(replica, param_name, param)

    for module_name, child in module._modules.items():
        setattr(replica, module_name, _replicate_module_recurse(child, process_group))
    return replica

def _replicate_module(network, process_group):
    from torch.nn.parallel.replicate import _replicatable_module
    if not _replicatable_module(network):
        raise RuntimeError("Cannot replicate network where python modules are "
                           "childrens of ScriptModule")

    return _replicate_module_recurse(network, process_group)
