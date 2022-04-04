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
class NestedTensor:
    # data is a torch.Tensor backed by a NestedTensorImpl

    def __init__(self, impl):
        self._impl = impl

    @property
    def dtype(self):
        """
        The data type of ```self``` NestedTensor.
        """
        return self._impl.dtype

    @property
    def layout(self):
        """
        The layout of ```self``` NestedTensor.
        """
        return self._impl.layout

    @property
    def device(self):
        """
        The device of ```self``` NestedTensor.
        """
        return self._impl.device

    @property
    def requires_grad(self):
        """
        Is ```True``` if gradients need to be computed for this Tensor.
        """
        return self._impl.requires_grad

    def stride(self):
        """
        NestedTensor currently does not have a stride. This will throw.
        """
        return self._impl.stride()

    def size(self):
        """
        NestedTensor currently does not have a size. This will throw.
        """
        return self._impl.size()

    def dim(self):
        """
        The dimension of ```self``` NestedTensor.
        """
        return self._impl.dim()

    def numel(self):
        """
        The number of elements of ```self``` NestedTensor.
        """
        return self._impl.numel()

    def is_contiguous(self):
        """
        Returns true if ```self``` NestedTensor is contiguous.
        """
        return self._impl.is_contiguous()

    def __str__(self):
        return str(self._impl)

    def __repr__(self):
        return self.__str__()

    def unbind(self, dim=None):
        if dim is None:
            unbound = self._impl.unbind(0)
            if len(unbound) == 0:
                return ()
            return unbound
        return self._impl.unbind(dim)
