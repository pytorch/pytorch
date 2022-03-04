import torch
from functools import wraps


@wraps(torch._nested_tensor)
def nested_tensor(*args, **kwargs):
    return NestedTensor(torch._nested_tensor(*args, **kwargs))


# TODO: This entire class is not really necessary now that NestedTensor lives
# in tree; before it lived out of tree and there was no way to conveniently
# override the string printing behavior.  Now that we are in tree, we can
# directly override _tensor_str to capture this behavior, and the wrapper subclass
# is not necessary. See also https://github.com/pytorch/pytorch/issues/73506
class NestedTensor(torch.Tensor):
    # data is a torch.Tensor backed by a NestedTensorImpl

    @staticmethod
    def __new__(cls, impl):
        # Use a Tensor that of the give size for the wrapper.
        kwargs = {}
        kwargs["device"] = impl.device
        kwargs["dtype"] = impl.dtype
        kwargs["layout"] = impl.layout
        kwargs["requires_grad"] = False
        tensors = impl.unbind()
        if len(tensors) == 0:
            size = (1,)
        else:
            size = (1,) * int(tensors[0].dim() + 1)
        return torch.Tensor._make_wrapper_subclass(cls, size, **kwargs)

    def __init__(self, impl):
        self._t_impl = impl

    @property
    def dtype(self):
        """
        The data type of ```self``` NestedTensor.
        """
        return self._t_impl.dtype

    @property
    def layout(self):
        """
        The layout of ```self``` NestedTensor.
        """
        return self._t_impl.layout

    @property
    def device(self):
        """
        The device of ```self``` NestedTensor.
        """
        return self._t_impl.device

    @property
    def requires_grad(self):
        """
        Is ```True``` if gradients need to be computed for this Tensor.
        """
        return self._t_impl.requires_grad

    def stride(self):
        """
        NestedTensor currently does not have a stride. This will throw.
        """
        return self._t_impl.stride()

    def size(self):
        """
        NestedTensor currently does not have a size. This will throw.
        """
        return self._t_impl.size()

    def dim(self):
        """
        The dimension of ```self``` NestedTensor.
        """
        tensors = self.unbind()
        if len(tensors) == 0:
            return 1
        return int(tensors[0].dim() + 1)

    def numel(self):
        """
        The number of elements of ```self``` NestedTensor.
        """
        return self._t_impl.numel()

    def is_contiguous(self):
        """
        Returns true if ```self``` NestedTensor is contiguous.
        """
        return self._t_impl.is_contiguous()

    def __str__(self):
        def _str(x, indent=0, tab="  "):
            s = indent * tab + "[\n"
            strs = list(map(str, x.unbind()))
            strs = list(
                map(
                    lambda xi: "\n".join(
                        map(lambda xij: (indent + 1) * tab + xij, xi.split("\n"))
                    ),
                    strs,
                )
            )
            s += ",\n".join(strs)
            s += "\n" + indent * tab + "]"
            return s

        return "nested_tensor(" + _str(self) + ")"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if func is torch.ops.aten.unbind:
            if len(args) == 1:
                self, dim = args[0], None
            else:
                self, dim = args
            assert len(kwargs) == 0
            if dim is None:
                unbound = torch.ops.aten.unbind.int(self._t_impl, 0)
                if len(unbound) == 0:
                    return ()
                return unbound
            return torch.ops.aten.unbind.int(self._t_impl, dim)
        return NotImplemented
