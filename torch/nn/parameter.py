import torch
from torch._C import _disabled_torch_function_impl
from collections import OrderedDict


class Parameter(torch.Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`locally-disable-grad-doc` for more details. Default: `True`
    """
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.requires_grad)
            memo[id(self)] = result
            return result

    def __repr__(self):
        return 'Parameter containing:\n' + super(Parameter, self).__repr__()

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (
            torch._utils._rebuild_parameter,
            (self.data, self.requires_grad, OrderedDict())
        )

    __torch_function__ = _disabled_torch_function_impl


class UninitializedTensorMixin:
    _allowed_methods = [
        torch.Tensor.__hash__,
        torch.Tensor.size,
        torch.Tensor.copy_,
        torch.Tensor.is_floating_point,
        torch.Tensor.half,
        torch.Tensor.float,
        torch.Tensor.double,
        torch.Tensor.char,
        torch.Tensor.short,
        torch.Tensor.int,
        torch.Tensor.long,
        torch.Tensor.cuda,
        torch.Tensor.cpu,
        torch.Tensor.to,
        torch.Tensor.get_device,
        torch._has_compatible_shallow_copy_type,
    ]

    def materialize(self, shape, device=None, dtype=None):
        r"""Create a Parameter or Tensor with the same properties of the uninitialized one.
        Given a shape, it materializes a parameter in the same device
        and with the same `dtype` as the current one or the specified ones in the
        arguments.

        Args:
            shape : (tuple): the shape for the materialized tensor.
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module. Optional.
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module. Optional.
        """
        if device is None:
            device = self.data.device
        if dtype is None:
            dtype = self.data.dtype
        self.data = torch.empty(shape, device=device, dtype=dtype)
        self.__class__ = self.cls_to_become

    @property
    def shape(self):
        raise RuntimeError(
            'Can\'t access the shape of an uninitialized parameter or buffer. '
            'This error usually happens in `load_state_dict` when trying to load '
            'an uninitialized parameter into an initialized one. '
            'Call `forward` to initialize the parameters before accessing their attributes.')

    def share_memory_(self):
        raise RuntimeError(
            'Can\'t share memory on an uninitialized parameter or buffer. '
            'Call `forward` to initialize the parameters before calling '
            '`module.share_memory()`.')

    def __repr__(self):
        return f'<{self.__class__.__name__}>'

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (
            self.__class__,
            (self.requires_grad,)
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # method-wrapper is to detect access to Tensor properties that are
        # wrapped in descriptors
        if func in cls._allowed_methods or func.__class__.__name__ == 'method-wrapper':
            if kwargs is None:
                kwargs = {}
            return super().__torch_function__(func, types, args, kwargs)
        raise ValueError(
            'Attempted to use an uninitialized parameter in {}. '
            'This error happens when you are using a `LazyModule` or '
            'explicitly manipulating `torch.nn.parameter.{}` '
            'objects. When using LazyModules Call `forward` with a dummy batch '
            'to initialize the parameters before calling torch functions'.format(func, cls.__name__))


def is_lazy(param):
    return isinstance(param, UninitializedTensorMixin)


class UninitializedParameter(UninitializedTensorMixin, Parameter):
    r"""A parameter that is not initialized.

    Unitialized Parameters are a a special case of :class:`torch.nn.Parameter`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.nn.Parameter`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.nn.Parameter`.

    The default device or dtype to use when the parameter is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """

    cls_to_become = Parameter

    def __new__(cls, requires_grad=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        data = torch.empty(0, **factory_kwargs)
        return torch.Tensor._make_subclass(cls, data, requires_grad)


class UninitializedBuffer(UninitializedTensorMixin, torch.Tensor):
    r"""A buffer that is not initialized.

    Unitialized Buffer is a a special case of :class:`torch.Tensor`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.Tensor`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.Tensor`.

    The default device or dtype to use when the buffer is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """

    cls_to_become = torch.Tensor

    def __new__(cls, requires_grad=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        data = torch.empty(0, **factory_kwargs)
        return torch.Tensor._make_subclass(cls, data, requires_grad)
