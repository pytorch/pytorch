import torch
import os
import math
 
class LazyDimension():
 
    @staticmethod
    def from_tensor(input, dim):
        # TODO: not sure either _dim or _input should be exposed at all
        # ideally we only need self._node
 
        # self._dim = dim
        # self._input = input
 
        ld = LazyDimension()
        ld._node = lazy_tensor_core._LAZYC._dynamic_size(input, dim)
        return ld
 
    @staticmethod
    def from_node(node):
      ld = LazyDimension()
      ld._node = node
      return ld  
   
    def __init__(self):
        import inspect
        import warnings
        caller_name = inspect.stack()[1][3]
        if caller_name not in ["from_node", "from_tensor"]:
            warnings.warn("LazyDimension should only be called by \
            `from_node` or `from_tensor`. This warning may be spurious \
            if not using CPython")
        pass
       
    @property
    def dynamic(self):
        lazy_tensor_core._LAZYC._is_dynamic(self._node)
 
    @property
    def static_value(self):
        lazy_tensor_core._LAZYC._get_static_value(self._node)

    def __len__(self, dim):
        n = lazy_tensor_core._LAZYC._dynamic_size(self._node, dim)
        return LazyDimension.from_node(n)
 
    def __mul__(self, other):
        assert (isinstance(other, LazyDimension))
        n = lazy_tensor_core._LAZYC._dynamic_mul(self._node, other)
        return LazyDimension.from_node(n)
 
    def __div__(self, other):
        assert (isinstance(other, LazyDimension))
        n = lazy_tensor_core._LAZYC._dynamic_div(self._node, self.other)
        return LazyDimension.from_node(n)

    def __add__(self, other):
        assert (isinstance(other, LazyDimension))
        n = lazy_tensor_core._LAZYC._dynamic_add(self._node, self.other)
        return LazyDimension.from_node(n)

    # def __int__(self):
    #     if not self.dynamic:
    #         return self.static_value
    #     else:
    #         # TODO: if static_value can recursively fetch updated value
    #         # and we can detect that tensors in leaves (i.e. aten::size) were materialized
    #         # we can return the correct answer here
    #         n = lazy_tensor_core._LAZYC._dynamic_int(self._node)
    #         return LazyDimension.from_node(n)
       
    def __str__(self) -> str:
        return f"LazyDimension({self._dim})"
 
    def __repr__(self):
        return str(self)
 
class LazyTensor(torch.Tensor):
    def size(self, *args):
        if len(args) == 0:
            # LazySize TODO:
            return [LazyDimension.from_tensor(self, dim) for dim in range(len(super().size()))]
        assert (isinstance(args[0]), int)
        return LazyDimension.from_tensor(self, args[0])
 
    def numel(self):
        # assuming size can give the correct answer
        # i.e. there are either no dynamic dimensions
        # or we materialize to get the precise answer
        # numel should also work
        return math.prod(self.size(), 1)
 
def view(self, *view_args):
    num_dims = len(view_args)
    class AutogradView(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, *view_args):
            ctx.save_for_backward(self)
            # view_args is List[LazyDimension]
            return lazy_tensor_core._LAZYC._dynamic_view(self, view_args)
 
        @staticmethod
        def backward(ctx, grad):
            (self,) = ctx.saved_tensors
     
            out = lazy_tensor_core._LAZYC._dynamic_view(grad, self.size())
            nonlocal num_dims
            nones = [None] * num_dims
            return out, *nones
 
if os.environ["LTC_ENABLE_DYNAMIC_SHAPES"]:
    torch._C._register_py_class_for_device("lazy", LazyTensor)