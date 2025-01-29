import torch 
from torch._C import DispatchKey

from torch._ops import HigherOrderOperator
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


class AccessSubclassInnerTensor(HigherOrderOperator):
    def __init__(self):
        super().__init__("access_subclass_inner_tensor")

    def __call__(self, src_subclass_tensor: torch.Tensor, attr: str):
        return super().__call__(src_subclass_tensor, attr)


access_subclass_inner_tensor = AccessSubclassInnerTensor()

@access_subclass_inner_tensor.py_impl(DispatchKey.Autograd)
def access_subclass_inner_tensor_autograd(src_subclass_tensor: torch.Tensor, attr: str):
    assert is_traceable_wrapper_subclass(src_subclass_tensor)
    val = getattr(src_subclass_tensor, attr, None)
    if val is None or not isinstance(val, torch.Tensor):
        raise RuntimeError(f"Attribute {attr} is not a tensor or doesn't exist in {src_subclass_tensor}")
    return val


def trace_access_subclass_inner_tensor(proxy_mode, src_subclass_tensor, attr):
    proxy_subclass = proxy_mode.tracer.unwrap_proxy(src_subclass_tensor)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", access_subclass_inner_tensor, (proxy_subclass, attr), {}, name="access_subclass_inner_tensor"
    )
    out = access_subclass_inner_tensor(src_subclass_tensor, attr)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@access_subclass_inner_tensor.py_impl(ProxyTorchDispatchMode)
def inner(proxy_mode, src_subclass_tensor, attr):
    if proxy_mode.enable_tracing:
        return trace_access_subclass_inner_tensor(
            proxy_mode, src_subclass_tensor, attr
        )
    else:
        return access_subclass_inner_tensor(src_subclass_tensor, attr)
