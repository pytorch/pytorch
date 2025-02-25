# mypy: ignore-errors

import importlib
from typing import Callable, List, Optional

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


TORCH_FN_OVERRIDE_OPS_ATTR = "tsc_torch_function_ops"
TORCH_DISPATCH_OVERRIDE_OPS_ATTR = "tsc_torch_dispatch_ops"


def gen_tensor_flatten_fn(
    inner_tensors: List[str],
    meta_attrs: Optional[List[str]],
    get_meta_fn: Optional[Callable],
):
    assert meta_attrs is None or get_meta_fn is None
    if meta_attrs is not None:

        def __tensor_flatten__(self):
            return inner_tensors, tuple(getattr(self, a) for a in meta_attrs)

        return __tensor_flatten__

    elif get_meta_fn is not None:

        def __tensor_flatten__(self):
            return inner_tensors, get_meta_fn(self)

        return __tensor_flatten__

    def __tensor_flatten__(self):
        return inner_tensors, None

    return __tensor_flatten__


def gen_tensor_unflatten_fn(
    module_name: str,
    class_name: str,
    inner_tensors: List[str],
    meta_attrs: Optional[List[str]],
    meta_init_kwargs_fn: Optional[Callable],
):
    assert meta_attrs is None or meta_init_kwargs_fn is None
    if meta_attrs is not None:

        def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
            module = importlib.import_module(module_name)
            clz = getattr(module, class_name)
            kwargs = inner_tensors
            for a, v in zip(meta_attrs, meta):
                kwargs[a] = v
            return clz(outer_size=outer_size, outer_stride=outer_stride, **kwargs)

        return __tensor_unflatten__

    elif meta_init_kwargs_fn:

        def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
            module = importlib.import_module(module_name)
            clz = getattr(module, class_name)
            kwargs = inner_tensors
            kwargs.update(meta_init_kwargs_fn(meta))

            return clz(outer_size=outer_size, outer_stride=outer_stride, **kwargs)

        return __tensor_unflatten__

    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        module = importlib.import_module(module_name)
        clz = getattr(module, class_name)
        kwargs = inner_tensors

        return clz(outer_size=outer_size, outer_stride=outer_stride, **kwargs)

    return __tensor_unflatten__


def gen_torch_function(
    pro_fn: Optional[Callable], epi_fn: Optional[Callable], override_fn_table
):
    # TODO: Shortcuts to gen minimal control flow if smth undef

    def fn(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if override_fn_table and func in override_fn_table:
            return override_fn_table[func](cls, func, types, args, kwargs)

        if pro_fn is not None:
            if ret := pro_fn.__wrapped__(cls, func, types, args, kwargs):
                return ret
        with torch._C.DisableTorchFunctionSubclass():
            if epi_fn:
                return epi_fn.__wrapped__(
                    cls, func, types, args, kwargs, func(*args, **kwargs)
                )

            return func(*args, **kwargs)

    return fn


def gen_torch_dispatch(override_fn_table):
    # TODO: Shortcuts to gen minimal control flow if smth undef

    def fn(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if override_fn_table and func in override_fn_table:
            return override_fn_table[func](cls, func, types, args, kwargs)

        return func(*args, **kwargs)

    return fn


def gen_repr(qualname, inner_tensors):
    def fn(self):
        r = ", ".join([f"{a}:{repr(getattr(self, a))}" for a in inner_tensors])
        return f"{qualname}({r})"

    return fn


def _parse_override_table(attrs, fn_attr):
    table = {}
    for k, v in attrs.items():
        if ops := getattr(v, fn_attr, None):
            for op in ops:
                assert (
                    op not in table
                ), f"More than one override of {fn_attr} op {op} func:{v}"
                table[op] = v
    return table or None


class BaseTensorSubclassMeta(torch._C._TensorMeta):
    def __new__(meta, name, bases, attrs):  # noqa: B902
        if "_INNER_TENSORS" in attrs:
            inner_tensors = attrs["_INNER_TENSORS"]
            meta_attrs = attrs.get("_META", None)
            if "__tensor_flatten__" not in attrs:
                attrs["__tensor_flatten__"] = gen_tensor_flatten_fn(
                    inner_tensors, meta_attrs, attrs.get("get_meta", None)
                )

            if "__tensor_unflatten__" not in attrs:
                module_name = attrs["__module__"]
                qualname = attrs["__qualname__"]
                if "<locals>" in qualname:
                    raise RuntimeError(
                        "Local subclasses of BaseTensorSubclass are not supported yet"
                    )
                attrs["__tensor_unflatten__"] = staticmethod(
                    gen_tensor_unflatten_fn(
                        module_name,
                        qualname,
                        inner_tensors,
                        meta_attrs,
                        attrs.get("meta_init_kwargs", None),
                    )
                )

            if "__slots__" not in attrs:
                attrs["__slots__"] = inner_tensors

            if "__repr__" not in attrs:
                attrs["__repr__"] = gen_repr(attrs["__qualname__"], inner_tensors)

        if "__torch_function__" not in attrs:
            torch_fn_pro = attrs.get("torch_function_prologue", None)
            torch_fn_epi = attrs.get("torch_function_epilogue", None)
            torch_fn_override_table = _parse_override_table(
                attrs, TORCH_FN_OVERRIDE_OPS_ATTR
            )
            attrs["__torch_function__"] = classmethod(
                gen_torch_function(torch_fn_pro, torch_fn_epi, torch_fn_override_table)
            )

        if "__torch_dispatch__" not in attrs:
            torch_dispatch_override_table = _parse_override_table(
                attrs, TORCH_DISPATCH_OVERRIDE_OPS_ATTR
            )
            attrs["__torch_dispatch__"] = classmethod(
                gen_torch_dispatch(torch_dispatch_override_table)
            )

        return super().__new__(meta, name, bases, attrs)


def tensor_kwargs_from(t: torch.Tensor, outer_stride=None):
    kwargs = {}
    kwargs["strides"] = outer_stride or t.stride()
    kwargs["storage_offset"] = t.storage_offset()
    kwargs["device"] = t.device
    kwargs["layout"] = t.layout
    kwargs["requires_grad"] = t.requires_grad
    kwargs["dtype"] = t.dtype
    return kwargs


class BaseTensorSubclass(torch.Tensor, metaclass=BaseTensorSubclassMeta):
    # TODO:
    # - Local subclasses support
    # - Dynamic number of inner tensors
    # - Comments
    # - Examples

    # User can override methods `get_meta` and `meta_init_kwargs` to be used by default generated
    # __tensor_flatten__, __tensor_unflatten__.
    #
    # Called by generated __tensor_flatten__ as subclass meta provider.
    # def get_meta(self):
    #     return None
    #
    # Called by __tensor_unflatten__ and passed to constructor of subclass.
    # def meta_init_kwargs(meta):
    #     return {}

    @classmethod
    def _args(cls, args, get_arg_fn):
        return pytree.tree_map_only(cls, get_arg_fn, args)

    @classmethod
    def args_attr(cls, args, attr_name):
        return cls._args(args, lambda x: getattr(x, attr_name))

    @classmethod
    def func_args_kwargs(cls, func, args, kwargs, get_arg_fn):
        args_ = cls.args(args, get_arg_fn)
        kwargs_ = cls.args(kwargs, get_arg_fn)
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

        return return_and_correct_aliasing(func, args, kwargs, out)

    # User can override this method to do something before the func is called
    # If return value is not None, it will be returned to the caller of __torch_function__
    # E.g.:
    # @classmethod
    # def torch_function_prologue(cls, func, types, args, kwargs) -> Optional[Any]:
    #     if func is torch.nn.functional.linear:
    #         print("linear")
    #     return None
    #
    # User can override this method to do something after the func is called
    # The return value will be returned to the caller of __torch_function__
    # E.g.:
    # @classmethod
    # def torch_function_epilogue(cls, func, types, args, kwargs, out):
    #     if func is torch.nn.functional.linear:
    #         print("linear")
    #     return out


def torch_function_override(ops):
    def wrapped_func(fn):
        setattr(fn, TORCH_FN_OVERRIDE_OPS_ATTR, ops)
        return fn

    return wrapped_func


def torch_dispatch_override(ops):
    def wrapped_func(fn):
        setattr(fn, TORCH_DISPATCH_OVERRIDE_OPS_ATTR, ops)
        return fn

    return wrapped_func
