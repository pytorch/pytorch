# mypy: ignore-errors

import torch
from copy import deepcopy
from torch.utils._pytree import tree_map
import torch.utils._pytree as pytree


# TODO: Move LoggingTensor here.
from torch.testing._internal.logging_tensor import LoggingTensor


# Base class for wrapper-style tensors.
class WrapperTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        t, kwargs = cls.get_wrapper_properties(*args, **kwargs)
        if "size" not in kwargs:
            size = t.size()
        else:
            size = kwargs["size"]
            del kwargs["size"]
        if "dtype" not in kwargs:
            kwargs["dtype"] = t.dtype
        if "layout" not in kwargs:
            kwargs["layout"] = t.layout
        if "device" not in kwargs:
            kwargs["device"] = t.device
        if "requires_grad" not in kwargs:
            kwargs["requires_grad"] = False
        # Ignore memory_format and pin memory for now as I don't know how to
        # safely access them on a Tensor (if possible??)

        wrapper = torch.Tensor._make_wrapper_subclass(cls, size, **kwargs)
        wrapper._validate_methods()
        return wrapper

    @classmethod
    def get_wrapper_properties(cls, *args, **kwargs):
        # Should return both an example Tensor and a dictionary of kwargs
        # to override any of that example Tensor's properly.
        # This is very similar to the `t.new_*(args)` API
        raise NotImplementedError("You need to implement get_wrapper_properties")

    def _validate_methods(self):
        # Skip this if not in debug mode?
        # Changing these on the python side is wrong as it would not be properly reflected
        # on the c++ side
        # This doesn't catch attributes set in the __init__
        forbidden_overrides = ["size", "stride", "dtype", "layout", "device", "requires_grad"]
        for el in forbidden_overrides:
            if getattr(self.__class__, el) is not getattr(torch.Tensor, el):
                raise RuntimeError(f"Subclass {self.__class__.__name__} is overwriting the "
                                   f"property {el} but this is not allowed as such change would "
                                   "not be reflected to c++ callers.")


class DiagTensorBelow(WrapperTensor):
    @classmethod
    def get_wrapper_properties(cls, diag, requires_grad=False):
        assert diag.ndim == 1
        return diag, {"size": diag.size() + diag.size(), "requires_grad": requires_grad}

    def __init__(self, diag, requires_grad=False):
        self.diag = diag

    handled_ops = {}

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        # For everything else, call the handler:
        fn = cls.handled_ops.get(func.__name__, None)
        if fn:
            return fn(*args, **(kwargs or {}))
        else:
            # Note that here, because we don't need to provide the autograd formulas
            # we can have a default "fallback" that creates a plain Tensor based
            # on the diag elements and calls the func again.

            def unwrap(e):
                return e.diag.diag() if isinstance(e, DiagTensorBelow) else e

            def wrap(e):
                if isinstance(e, torch.Tensor) and e.ndim == 1:
                    return DiagTensorBelow(e)
                if isinstance(e, torch.Tensor) and e.ndim == 2 and e.count_nonzero() == e.diag().count_nonzero():
                    return DiagTensorBelow(e.diag())
                return e

            rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
            return rs

    def __repr__(self):
        return super().__repr__(tensor_contents=f"diag={self.diag}")


class SparseTensor(WrapperTensor):
    @classmethod
    def get_wrapper_properties(cls, size, values, indices, requires_grad=False):
        assert values.device == indices.device
        return values, {"size": size, "requires_grad": requires_grad}

    def __init__(self, size, values, indices, requires_grad=False):
        self.values = values
        self.indices = indices

    def __repr__(self):
        return super().__repr__(tensor_contents=f"values={self.values}, indices={self.indices}")

    def sparse_to_dense(self):
        res = torch.zeros(self.size(), dtype=self.values.dtype)
        res[self.indices.unbind(1)] = self.values
        return res

    @staticmethod
    def from_dense(t):
        indices = t.nonzero()
        values = t[indices.unbind(1)]
        return SparseTensor(t.size(), values, indices)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        func_name = f"{func.__module__}.{func.__name__}"

        res = cls._try_call_special_impl(func_name, args, kwargs)
        if res is not NotImplemented:
            return res

        # Otherwise, use a default implementation that construct dense
        # tensors and use that to compute values
        def unwrap(e):
            return e.sparse_to_dense() if isinstance(e, SparseTensor) else e

        # Wrap back all Tensors into our custom class
        def wrap(e):
            # Check for zeros and use that to get indices
            return SparseTensor.from_dense(e) if isinstance(e, torch.Tensor) else e

        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
        return rs

    # To show how things happen later
    def __rmul__(self, other):
        return super().__rmul__(other)

    _SPECIAL_IMPLS = {}

    @classmethod
    def _try_call_special_impl(cls, func, args, kwargs):
        if func not in cls._SPECIAL_IMPLS:
            return NotImplemented
        return cls._SPECIAL_IMPLS[func](args, kwargs)


# Example non-wrapper subclass that stores extra state.
class NonWrapperTensor(torch.Tensor):
    def __new__(cls, data):
        t = torch.Tensor._make_subclass(cls, data)
        t.extra_state = {
            'last_func_called': None
        }
        return t

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        result = super().__torch_function__(func, types, args, kwargs)

        if isinstance(result, cls):
            # Do something with the extra state. For the example here, just store the name of the
            # last function called (skip for deepcopy so the copy has the same extra state).
            if func is torch.Tensor.__deepcopy__:
                result.extra_state = deepcopy(args[0].extra_state)
            else:
                result.extra_state = {
                    'last_func_called': func.__name__,
                }

        return result

    # new_empty() must be defined for deepcopy to work
    def new_empty(self, shape):
        return type(self)(torch.empty(shape))


# Class used to store info about subclass tensors used in testing.
class SubclassInfo:

    __slots__ = ['name', 'create_fn', 'closed_under_ops']

    def __init__(self, name, create_fn, closed_under_ops=True):
        self.name = name
        self.create_fn = create_fn  # create_fn(shape) -> tensor instance
        self.closed_under_ops = closed_under_ops


subclass_db = {
    torch.Tensor: SubclassInfo(
        'base_tensor', create_fn=torch.randn
    ),
    NonWrapperTensor: SubclassInfo(
        'non_wrapper_tensor',
        create_fn=lambda shape: NonWrapperTensor(torch.randn(shape))
    ),
    LoggingTensor: SubclassInfo(
        'logging_tensor',
        create_fn=lambda shape: LoggingTensor(torch.randn(shape))
    ),
    SparseTensor: SubclassInfo(
        'sparse_tensor',
        create_fn=lambda shape: SparseTensor.from_dense(torch.randn(shape).relu())
    ),
    DiagTensorBelow: SubclassInfo(
        'diag_tensor_below',
        create_fn=lambda shape: DiagTensorBelow(torch.randn(shape)),
        closed_under_ops=False  # sparse semantics
    ),
}

class SubclassWithTensorFactory(torch.Tensor):
    @staticmethod
    def __new__(cls, src):
        shape = src.shape
        kwargs = {}
        kwargs["strides"] = src.stride()
        kwargs["storage_offset"] = src.storage_offset()
        kwargs["device"] = src.device
        kwargs["layout"] = src.layout
        kwargs["requires_grad"] = src.requires_grad
        kwargs["dtype"] = src.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __init__(self, src):
        self.src = src

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __tensor_flatten__(self):
        return ["src"], None

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, meta, outer_size, outer_stride):
        src = inner_tensors["src"]
        return cls(src)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}

        def _fn(x):
            return x.src * torch.ones(x.src.shape) if x.src.dtype == torch.float32 else x.src

        _args = pytree.tree_map_only(cls, _fn, args)
        _kwargs = pytree.tree_map_only(cls, _fn, kwargs)

        _out = func(*_args, **_kwargs)

        _out_flat, _out_spec = pytree.tree_flatten(_out)

        out_flat = [cls(o) if isinstance(o, torch.Tensor) else o for o in _out_flat]
        return pytree.tree_unflatten(out_flat, _out_spec)
