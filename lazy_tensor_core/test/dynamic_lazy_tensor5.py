import torch
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as ltm
from collections.abc import Iterable
from enum import Enum

lazy_tensor_core._LAZYC._ltc_init_ts_backend()
lazy_tensor_core._LAZYC._ltc_set_dynamic_shapes_mode()


class LazyDynamicSize:
    def __init__(self, n):
        super().__init__()
        self._n = n

    @staticmethod
    def fromTensor(t):
        n = lazy_tensor_core._LAZYC._dynamic_size2(t)
        return LazyDynamicSize(n)

class AutogradExpand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self, size):
        ctx.save_for_backward(self)
        print("running Expand forward")
        t = lazy_tensor_core._LAZYC._dynamic_expand2(self, size._n)
        return t

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: we need sum_to for expand
        print("running Expand backward")
        (input,) = ctx.saved_tensors
        return lazy_tensor_core._LAZYC._sum_to_size(grad_output, input.sizes()._n), None

class DynamicLazyTensor4(torch.Tensor):
    """
        Non-compound ops
    """

    # Category 1: non-compound ops that don't take size or dim and don't use sizes() in backward
    # e.g. add, relu, abs, mul, div, etc
    # No work necessary as long we can rely on `torch.autograd.register_py_tensor_class_for_device("lazy", DynamicLazyTensor4)`

    # Category 2: non-compound ops that return sizes() or dim can be implemented with thin wrappers around IR
    # We would need to handwrite these ops
    def sizes(self):
        return LazyDynamicSize.fromTensor(self)

    def size(index):
        raise RuntimeError("NYI")

    def expand(self, sizes):
        return AutogradExpand.apply(self, sizes)



# register the python class
torch.autograd._register_py_tensor_class_for_device("lazy", DynamicLazyTensor4)
a = torch.ones(2, 2, requires_grad=True).to(device="lazy")
a2 = torch.ones(2, 2).to(device="lazy")
# print(f"a2={a2[0, 0]}")
# print(f"a={a[0, 0]}")

print(type(a).__name__)
sz = a.sizes()
c = a.expand(sz)
b = torch.nn.functional.gelu(c)





grad_outputs = [torch.ones(2, 2).to(device="lazy")]
grads = torch.autograd.grad((b, ), (a,), grad_outputs)

print(lazy_tensor_core._LAZYC._get_ltc_tensors_text([grads[0]]))
print(lazy_tensor_core._LAZYC._get_ltc_tensors_backend([grads[0]]))

print(grads[0][0, 0].item())
#ltm.mark_step()