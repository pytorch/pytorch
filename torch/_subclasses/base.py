from typing import Any, Optional

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


class BaseTensorSubclassMeta(type):
    def __new__(meta, name, bases, attrs):
        breakpoint()
        return type.__new__(meta, name, bases, attrs)


class BaseTensorSubclass(torch.Tensor, metaclass=BaseTensorSubclassMeta):

    # __metaclass__ = BaseTensorSubclassMeta

    @classmethod
    def _args(cls, args, get_arg_fn):
        return pytree.tree_map_only(cls, get_arg_fn, args)

    @classmethod
    def args_attr(cls, args, attr_name):
        return cls._args(args, lambda x: getattr(x, attr_name))

    @classmethod
    def func_args_kwargs(cls, func, args, kwargs, get_arg_fn):
        args_ = cls.args(args, _get_arg_fn)
        kwargs_ = cls.args(kwargs, _get_arg_fn)
        return func(*args_, **kwargs_)

    @classmethod
    def func_args_kwargs_attr(cls, func, args, kwargs, attr_name):
        args_ = cls.args_attr(args, attr_name)
        kwargs_ = cls.args_attr(kwargs, attr_name)
        return func(*args_, **kwargs_)

    @classmethod
    def _return(cls, func, args, kwargs, out):
        from torch._higher_order_ops.cond import cond_op

        if func is cond_op:
            return out
        else:
            return return_and_correct_aliasing(func, args, kwargs, out)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        raise NotImplementedError

    @classmethod
    def torch_function_prologue(cls, func, types, args, kwargs) -> Optional[Any]:
        # User can override this method to do something before the func is called
        # If return value is not None, it will be returned to the caller of __torch_function__
        # E.g.:
        # if func is torch.nn.functional.linear:
        #     print("linear")
        return None

    @classmethod
    def torch_function_epilogue(cls, func, types, args, kwargs, out):
        # User can override this method to do something after the func is called
        # The return value will be returned to the caller of __torch_function__
        # E.g.:
        # if func is torch.nn.functional.linear:
        #     print("linear")
        return out

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if ret := cls.torch_function_prologue(func, types, args, kwargs):
            return ret
        with torch._C.DisableTorchFunctionSubclass():
            return cls.torch_function_epilogue(
                func, types, args, kwargs, func(*args, **kwargs)
            )

    # TODO: classmethod.__get_inner_tensors__ ?
    # to produce __tensor_flatten__ and __tensor_unflatten__ ?

    def __tensor_flatten__(self):
        return ["a", "b"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert meta is None
        a, b = inner_tensors["a"], inner_tensors["b"]
        if type(a) is torch.Tensor:
            assert outer_size is not None
            assert outer_stride is not None
        return TwoTensor(a, b, outer_size, outer_stride)
