import inspect
import operator
import types
from typing import Dict, List

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


import sympy

import torch._numpy as tnp

import torch.fx
import torch.random

from torch.fx.experimental.symbolic_shapes import free_symbols, guard_scalar, SymTypes

from .. import config, variables

from ..exc import unimplemented
from ..guards import GuardBuilder
from ..source import AttrSource
from ..utils import (
    fqn,
    get_custom_getattr,
    get_fake_value,
    get_real_value,
    guard_if_dyn,
    object_has_getattribute,
    product,
    proxy_args_kwargs,
    tensortype_to_dtype,
)
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import SizeVariable

supported_tensor_comparison_ops = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}
supported_const_comparison_ops = {
    "is": operator.is_,
    "is not": operator.is_not,
    "==": operator.eq,
    "!=": operator.ne,
}


class TensorVariable(VariableTracker):
    """A torch.Tensor input or an intermediate value in the FX graph"""

    _nonvar_fields = [
        "proxy",
        "dtype",
        "device",
        "layout",
        "ndim",
        "size",
        "stride",
        "requires_grad",
        "is_quantized",
        "is_contiguous",
    ]

    def get_real_value(self):
        """
        Get the actual value represented by this variable if computation is run
        using the user-provided inputs.
        NOTE: this runs actual tensor computation and may be
        slow and memory-intensive.
        """
        return get_real_value(self.proxy.node, self.proxy.tracer)

    def __init__(
        self,
        proxy: torch.fx.Proxy,
        *,
        dtype,
        device,
        layout,
        ndim,
        requires_grad,
        is_quantized,
        is_sparse,
        class_type,
        size=None,
        stride=None,
        is_contiguous=None,
        specialized_value=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proxy = proxy
        self.dtype = dtype
        self.device = device
        self.layout = layout
        self.ndim = ndim
        self.size = size
        self.stride = stride
        self.requires_grad = requires_grad
        self.is_quantized = is_quantized
        self.is_contiguous = is_contiguous
        self.is_sparse = is_sparse
        self.class_type = class_type
        self.specialized_value = specialized_value

    def as_proxy(self):
        return self.proxy

    def python_type(self):
        return self.class_type

    def call_isinstance(self, tensor_type):
        def check_type(ty):
            if ty not in tensortype_to_dtype:
                return issubclass(self.python_type(), ty)

            dtypes = tensortype_to_dtype[ty]
            return self.dtype in dtypes

        if type(tensor_type) is tuple:
            return any(check_type(ty) for ty in tensor_type)
        else:
            return check_type(tensor_type)

    @staticmethod
    def specialize(value: torch.Tensor):
        props = {
            "dtype": value.dtype,
            "device": value.device,
            "layout": value.layout,
            "ndim": int(value.ndim),
            "requires_grad": value.requires_grad,
            "is_quantized": value.is_quantized,
            "is_sparse": value.is_sparse,
            "class_type": type(value),
        }
        if not free_symbols(value):
            # this is a fully static shape, and the keys on props here inform specialization.
            # We have to cast to int here, because these might get accessed as ConstantVariable, which has
            # a strict no-symint policy. If we got here due to not having free symbols, this is a known constant
            # already. We could remove the discrepancy here, by having ConstantVariable be more permissive for
            # constant backed SymInts, but that assert being strict has led to some good signal in hunting bugs, and
            # I'd like to keep it around for now.
            props["size"] = tuple([int(s) for s in value.size()])
            props["stride"] = tuple(value.stride())
            props["is_contiguous"] = tuple(
                [
                    x
                    for x in torch._prims_common._memory_formats
                    if value.is_contiguous(memory_format=x)
                ]
            )
        return props

    def dynamic_getattr(self, tx, name):
        if not self.source:
            raise NotImplementedError()

        # For local source, we associate the real value. We use this real value
        # for implementing getattr fallthrough on the variable tracker base class.

        # Note - this scope construction is mirrored in guards
        # A subsequent PR will introduce a util.
        scope = {"L": tx.output.local_scope, "G": tx.output.global_scope}
        try:
            # We raise in case we get a typerror bug w/ SuperSource.
            # SuperSource has bugs in it atm, and can produce code like
            # eval("super(L['mod'].model.model.encoder.embed_positions.forward__class__,
            # L['mod'].model.model.encoder.embed_positions)", scope)
            # Which is incorrect, and violates the invariant that all sources should be eval()-able against the scope.
            _input_associated_real_value = eval(self.source.name(), scope)
        except Exception:
            raise NotImplementedError()

        if _input_associated_real_value is None:
            raise NotImplementedError()

        if object_has_getattribute(_input_associated_real_value):
            raise NotImplementedError()

        if get_custom_getattr(_input_associated_real_value):
            raise NotImplementedError()

        real_value = getattr(_input_associated_real_value, name)
        if callable(real_value):
            # Callables have more nuanced handling, and we should let the existing system delegate here.
            # Raising was past behavior and so should always be sound to fall back.
            # Note - at a certain point we may want to handle
            raise NotImplementedError()

        from ..guards import GuardBuilder
        from .builder import VariableBuilder

        attr_source = AttrSource(self.source, name)
        has_attr_guard = attr_source.make_guard(GuardBuilder.HASATTR)
        return (
            VariableBuilder(tx, attr_source)(real_value)
            .add_options(self)
            .add_guard(has_attr_guard)
        )

    def var_getattr(self, tx, name):
        from . import ConstantVariable, TorchVariable

        if tx.strict_checks_enabled:
            if name in self._strict_mode_banned_ops():
                unimplemented(f"Illegal getattr invocation {name} in strict mode")

        result = None
        options = VariableTracker.propagate(self)
        if name == "ndim" and self.ndim is not None:
            result = ConstantVariable.create(self.ndim, **options)
        elif name == "dtype" and self.dtype is not None:
            result = TorchVariable(self.dtype, **options)
        elif name == "device" and self.device is not None:
            result = TorchVariable(self.device, **options)
        elif name == "layout" and self.layout is not None:
            result = TorchVariable(self.layout, **options)
        elif name == "is_cuda" and self.device is not None:
            result = ConstantVariable.create(self.device.type == "cuda", **options)
        elif name == "shape" and self.size is not None:
            sizes = [variables.ConstantVariable.create(x) for x in self.size]
            result = SizeVariable(sizes, **options)
        elif name == "requires_grad" and self.requires_grad is not None:
            result = ConstantVariable.create(self.requires_grad, **options)
        elif name == "is_quantized" and self.is_quantized is not None:
            result = ConstantVariable.create(self.is_quantized, **options)
        elif name == "is_sparse" and self.is_sparse is not None:
            result = ConstantVariable.create(self.is_sparse, **options)
        elif name == "shape" and self.size is None:
            result = self.call_method(tx, "size", [], {})
        elif name == "ndim" and self.ndim is None:
            result = self.call_method(tx, "dim", [], {})
        elif name == "data":
            result = self.call_method(tx, "detach", [], {})
        if name == "__class__":
            return TorchVariable(self.python_type(), **options)

        # Add a guard for type matching, these guards are checked before tensor guards
        # In some cases, a <tensor>.<attr> guard can be evaluated first, and break if
        # <tensor> is later changed to another type
        if result is not None and self.source is not None:
            result = result.add_guard(self.make_guard(GuardBuilder.TYPE_MATCH))

        # It's hard to get inplace view (metadata mutation) on graph input work properly across
        # dynamo/aot/inductor, just fall back.
        if self.source is not None and hasattr(torch.ops.aten, name):
            fn = getattr(torch.ops.aten, name)
            if (
                hasattr(fn, "overloads")
                and hasattr(fn, fn.overloads()[0])
                and torch.Tag.inplace_view in getattr(fn, fn.overloads()[0]).tags
            ):
                # Delay the graph break to the actual call of unsqueeze_/resize_/resize_as_ etc.
                return variables.misc.DelayGraphBreakVariable()

        # For attributes (not methods) that were not caught in the special handling above,
        # (e.g. tensor.real), we handle these generically, assuming that the output type is
        # a tensor.
        if result is None:

            def try_generic_attr_handling():
                from .builder import wrap_fx_proxy
                from .misc import GetAttrVariable

                try:
                    static_attr = inspect.getattr_static(torch.Tensor, name)
                except AttributeError:
                    return None

                # Make sure this is an attribute, not a method.
                # type(torch.Tensor.H) should be "getset_descriptor"
                # This is a because of CPython implementation, see THPVariableType:
                # these attributes are implemented under tp_getset, which appear
                # as `getset_descriptor`s, (compared to, say, methods which appear
                # as `method_descriptor`s)
                if type(static_attr) != types.GetSetDescriptorType:
                    return None

                return wrap_fx_proxy(
                    tx=tx,
                    proxy=GetAttrVariable.create_getattr_proxy(self.as_proxy(), name),
                    **options,
                )

            result = try_generic_attr_handling()

        if result is None:
            result = self.dynamic_getattr(tx, name)

        if result is None:
            raise NotImplementedError()
        return result

    def has_unpack_var_sequence(self, tx):
        return self.ndim > 0

    def unpack_var_sequence(self, tx, idxes=None):
        from .builder import wrap_fx_proxy

        options = VariableTracker.propagate(self)
        if idxes is None:
            if self.size:
                length = self.size[0]
            else:
                dyn_length = self.call_method(
                    tx, "size", [ConstantVariable.create(0)], {}
                )
                # SymNodeVariable for symbolic sizes, ConstantVariable for constants OR values prouced through
                # symbolic_shapes, but that end up as int/sympy.Integer
                assert isinstance(dyn_length, (SymNodeVariable, ConstantVariable))
                if isinstance(dyn_length, SymNodeVariable):
                    length = dyn_length.evaluate_expr(tx.output)
                else:
                    length = dyn_length.value
            idxes = range(length)
        return [wrap_fx_proxy(tx, self.as_proxy()[i], **options) for i in idxes]

    def _strict_mode_banned_ops(self):
        return torch._dynamo.config._autograd_backward_strict_mode_banned_ops

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if tx.strict_checks_enabled:
            if name in self._strict_mode_banned_ops():
                unimplemented(f"Illegal method invocation {name} in strict mode")
        from . import ConstantVariable, TorchVariable, TupleVariable
        from .builder import wrap_fx_proxy

        kwargs = dict(kwargs)
        options = VariableTracker.propagate(self, args, kwargs.values())

        if name in ("stride", "size"):
            dim_var = None
            if len(args) == 1:
                dim_var = args[0]
            elif "dim" in kwargs:
                dim_var = kwargs["dim"]
            else:
                assert not args and not kwargs, f"Tensor.{name}() unhandled args/kwargs"

            dim = guard_if_dyn(dim_var)

            def make_const_size_variable(x, **options):
                return SizeVariable(
                    [ConstantVariable.create(y, **options) for y in x], **options
                )

            RetVariable = (
                make_const_size_variable if name == "size" else ConstantVariable.create
            )

            # Technically, this should not be necessary, but I'm including it
            # for enhanced BC, in case example_value is sometimes not set
            # (it really should always be set though!)
            if (r := getattr(self, name)) is not None:
                if dim is None:
                    return RetVariable(r, **options)
                else:
                    return ConstantVariable.create(r[dim], **options)

            # It might still be constant!  Consult the fake tensor and see
            if (fake := self.proxy.node.meta.get("example_value")) is not None:
                if dim is None:
                    fake_r = getattr(fake, name)()
                    if not free_symbols(fake_r):
                        # int conversion for safety, in case a SymInt refined
                        # to constant
                        return RetVariable(tuple(int(r) for r in fake_r), **options)
                else:
                    fake_r = getattr(fake, name)(dim)
                    if not free_symbols(fake_r):
                        return ConstantVariable.create(int(fake_r), **options)

            # Oops, it's not constant.  Do the dynamic shapes path.
            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_method",
                    name,
                    *proxy_args_kwargs([self] + list(args), kwargs),
                ),
                **options,
            )

        elif name in ("numel", "nelement"):
            if self.size is not None:
                return ConstantVariable.create(product(self.size), **options)

            # It might still be constant!  Consult the fake tensor and see
            if (fake := self.proxy.node.meta.get("example_value")) is not None:
                fake_r = fake.numel()
                if not free_symbols(fake_r):
                    return ConstantVariable.create(int(fake_r), **options)

            assert not kwargs, f"Tensor.{name}() unhandled kwargs"

            # Oops, it's not constant.  Do the dynamic shapes path.
            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_method",
                    "numel",
                    *proxy_args_kwargs([self] + list(args), kwargs),
                ),
                **options,
            )

        elif name in ("ndimension", "dim") and self.ndim is not None:
            constant_result = ConstantVariable.create(self.ndim, **options)
        elif name == "is_floating_point" and self.dtype is not None:
            constant_result = ConstantVariable.create(
                self.dtype.is_floating_point, **options
            )
        elif name == "is_contiguous" and self.is_contiguous is not None:
            if "memory_format" in kwargs:
                memory_format = kwargs.pop("memory_format").as_python_constant()
            else:
                memory_format = torch.contiguous_format
            constant_result = ConstantVariable.create(
                memory_format in self.is_contiguous, **options
            )
        elif (
            name == "type"
            and self.dtype is not None
            and len(args) == 0
            and isinstance(self.device, torch.device)
        ):
            tensortype = [k for k, v in tensortype_to_dtype.items() if self.dtype in v][
                0
            ]
            if self.device.type == "cuda":
                constant_result = ConstantVariable.create(
                    f"torch.cuda.{tensortype.__name__}", **options
                )
            else:
                constant_result = ConstantVariable.create(
                    f"torch.{tensortype.__name__}", **options
                )
        elif (
            name == "type"
            and len(args) == 1
            and fqn(type(args[0].as_python_constant())) == "torch.tensortype"
        ):
            # torch.FloatTensor, etc. are all of type "torch.tensortype".
            # torch.fx's tracer fails on these types, because it doesn't support arguments of torch.tensortype type.
            # So, we pass it in as a string (which is also supported, see above implementation for .type() with 0 args)
            tensor_type = args[0].as_python_constant()
            tensor_type_const = ConstantVariable.create(fqn(tensor_type), **options)
            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_method",
                    name,
                    *proxy_args_kwargs([self, tensor_type_const], kwargs),
                ),
                **options,
            )
        elif name == "get_device" and isinstance(self.device, torch.device):
            index = self.device.index if self.device.type != "cpu" else -1
            constant_result = ConstantVariable.create(index, **options)
        else:
            constant_result = None

        if constant_result:
            assert not kwargs, f"Tensor.{name}() unhandled kwargs"
            # TODO: I think this branch is dead
            if len(args) == 1:
                return constant_result.getitem_const(args[0])
            elif args:
                return TupleVariable(
                    [constant_result.getitem_const(a) for a in args], **options
                )
            return constant_result
        elif name == "numpy":
            if not config.trace_numpy:
                unimplemented("Tensor.numpy(). config.trace_numpy is False")
            if not np:
                unimplemented("Tensor.numpy(). NumPy is not available")
            assert not args, "Tensor.numpy() doesn't take args."
            if self.layout != torch.strided:
                raise TypeError(
                    f"can't convert {self.layout} layout tensor to numpy. Use Tensor.dense() first"
                )
            # We don't check that the tensor is on CPU when force is False, as this
            # allows us to execute NumPy code on CUDA.
            # We don't check that requires_grad=False as we are curently doing an
            # unconditional detach.
            # TODO: We may want to avoid detching if `requires_grad=True`
            #       and `force=False` to allow computing gradients.
            force = "force" in kwargs and kwargs["force"].as_python_constant()
            proxy = tx.output.create_proxy(
                "call_method", "detach", *proxy_args_kwargs([self], {})
            )
            if force:
                # TODO Add resolve_conj and resolve_neg once we support complex tensors
                proxy = tx.output.create_proxy(
                    "call_method", "cpu", *proxy_args_kwargs([self], {})
                )
            return NumpyNdarrayVariable.create(tx, proxy, **options)
        elif name == "tolist":
            from .builder import SourcelessBuilder

            def tolist(tensor, sub_proxy):
                def wrap(i, sub_proxy):
                    return SymNodeVariable.create(
                        tx,
                        sub_proxy.item(),
                        sym_num=tx.output.shape_env.create_unbacked_symint(),
                    )

                if tensor.dtype not in [
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                ]:
                    unimplemented("Input tensor for tolist must be an integer tensor")

                if tensor.dim() == 0:
                    return wrap(tensor, sub_proxy)

                if tensor.dim() == 1:
                    return [wrap(val, sub_proxy[i]) for i, val in enumerate(tensor)]

                return [
                    tolist(sub_tensor, sub_proxy=sub_proxy[i])
                    for i, sub_tensor in enumerate(tensor)
                ]

            tensor = self.as_proxy().node.meta["example_value"]
            out = tolist(tensor, self.as_proxy())
            return SourcelessBuilder()(tx, out).add_options(options)
        elif name in ("backward", "data_ptr"):
            unimplemented(f"Tensor.{name}")
        elif name == "item" and not config.capture_scalar_outputs:
            unimplemented(f"Tensor.{name}")
        elif name == "__len__":
            return self.call_method(
                tx, "size", [ConstantVariable.create(0, **options)], {}
            )
        elif name == "__setitem__":
            key, value = args

            def has_bool_key(v):
                if isinstance(v, TensorVariable):
                    return v.dtype in (torch.bool, torch.int8)
                elif isinstance(v, TupleVariable):
                    return any(has_bool_key(item) for item in v.items)
                else:
                    return False

            if (
                not config.capture_dynamic_output_shape_ops
                and has_bool_key(key)
                and isinstance(value, TensorVariable)
                and value.requires_grad
            ):
                unimplemented(
                    "boolean masking setitem backwards requires dynamic shapes"
                )
            tx.output.guards.update(options["guards"])
            tx.output.create_proxy(
                "call_function",
                operator.setitem,
                *proxy_args_kwargs([self] + list(args), kwargs),
            )
            return ConstantVariable.create(None, **options)
        elif name in ("resize_", "resize_as_"):
            if "memory_format" in kwargs:
                memory_format = kwargs["memory_format"].as_python_constant()
            else:
                memory_format = torch.contiguous_format

            if name == "resize_":
                self.size = args[0].as_python_constant()
                self.is_contiguous = (memory_format,)
            else:
                assert isinstance(args[0], TensorVariable)
                if self.size and args[0].size:
                    if (
                        self.size == args[0].size
                        or memory_format is torch.preserve_format
                    ):
                        self.is_contiguous = args[0].is_contiguous
                    else:
                        self.size = args[0].size
                        self.stride = args[0].stride
                        self.ndim = args[0].ndim
                        self.is_contiguous = (memory_format,)

            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_method",
                    name,
                    *proxy_args_kwargs([self] + list(args), kwargs),
                ),
                **options,
            )
        elif (
            name == "add_" and len(args) == 1 and len(kwargs) == 1 and "alpha" in kwargs
        ):
            result = TorchVariable(torch.mul, **options).call_function(
                tx, args + [kwargs["alpha"]], {}
            )
            return self.call_method(tx, "add_", [result], {})
        elif (
            name == "addcdiv_"
            and len(args) == 2
            and len(kwargs) == 1
            and "value" in kwargs
        ):
            result = TorchVariable(torch.div, **options).call_function(tx, args, {})
            result = TorchVariable(torch.mul, **options).call_function(
                tx, [result, kwargs["value"]], {}
            )
            return self.call_method(tx, "add_", [result], {})
        elif name == "__contains__":
            # Rewrite __contains__ here so that downstream passes can trace through
            # without dealing with unbacked symbool. Roughly the code we translate is:
            # def __contains__(self, x):
            #     return (x == self).any().item()
            result = TorchVariable(torch.eq, **options).call_function(
                tx, [self, args[0]], {}
            )
            result = TorchVariable(torch.any, **options).call_function(tx, [result], {})
            return result.call_method(tx, "item", [], {})
        elif name == "redistribute":
            # rewrite non-primitive args/kwargs to be included in the on-the-fly prim function
            # and rewrite args to have only proxyable args, then insert call_function
            args_as_value = [x.as_python_constant() for x in args]
            kwargs_as_value = {k: v.as_python_constant() for k, v in kwargs.items()}

            def redistribute_fn_with_prim_types(x):
                return x.redistribute(*args_as_value, **kwargs_as_value)

            # attach the same function name for better debugging
            redistribute_fn_with_prim_types.__name__ = f"prim_{name}"

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    redistribute_fn_with_prim_types,
                    *proxy_args_kwargs([self], {}),
                ),
                **options,
            )
        elif name == "register_hook":
            # see [On tensor.register_hook]
            assert len(args) == 1
            fn_var = args[0]

            if isinstance(fn_var, variables.NestedUserFunctionVariable):
                # NestedUserFunctionVariable don't carry their fn, but reconstruction builds it
                # This should not be onerous to support when needed.
                unimplemented("NYI - lambda variables as hooks")
            elif isinstance(fn_var, variables.functions.FunctoolsPartialVariable):
                fn = fn_var.as_python_constant()
                name = fn_var.func.fn.__name__
            else:
                fn = fn_var.fn
                name = fn_var.fn.__name__

            handle_variable = variables.user_defined.RemovableHandleVariable(
                mutable_local=variables.base.MutableLocal(),
                **options,
            )
            if not self.source:
                # Intermediary
                unimplemented("Intermediary tensors with registered hooks - NYI")
            else:
                assert (
                    fn_var.source
                ), "Unreachable - See unimplemented for lambdas above"
            tx.output.side_effects.register_hook(self, fn_var, handle_variable)
            return handle_variable

        else:
            # Convert x.new(torch.Size) into x.new_empty(torch.Size),
            # as Tensor.new acts differently with a Size input versus a tuple input.
            if name == "new" and len(args) == 1 and isinstance(args[0], SizeVariable):
                name = "new_empty"
            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_method",
                    name,
                    *proxy_args_kwargs([self] + list(args), kwargs),
                ),
                **options,
            )

    def rename(self, tx, name):
        self.proxy.node._rename(name)
        return super().rename(tx, name)


class SymNodeVariable(VariableTracker):
    """
    Represents a symbolic size, e.g., as returned by tensor.size(0)
    """

    @classmethod
    def create(cls, tx, proxy, sym_num, **options):
        if "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == sym_num
        if sym_num is None:
            sym_num = get_fake_value(proxy.node, tx)
        proxy.node.meta["example_value"] = sym_num

        if isinstance(sym_num, (sympy.Integer, int)):
            return ConstantVariable.create(int(sym_num))

        return SymNodeVariable(proxy, sym_num, **options)

    def __init__(self, proxy, sym_num, **kwargs):
        super().__init__(**kwargs)
        self.proxy = proxy
        # TODO: Should we allow non SymTypes here?  Today it is allowed
        self.sym_num = sym_num

    def python_type(self):
        if isinstance(self.sym_num, SymTypes):
            return self.sym_num.node.pytype
        else:
            return type(self.sym_num)

    def unpack_var_sequence(self, tx):
        super().unpack_var_sequence(tx)

    def as_proxy(self):
        return self.proxy

    def evaluate_expr(self, output_graph=None):
        return guard_scalar(self.sym_num)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        options = VariableTracker.propagate(self, args, kwargs.values())

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_method",
                name,
                *proxy_args_kwargs([self] + list(args), kwargs),
            ),
            **options,
        )


class TensorWithTFOverrideVariable(VariableTracker):
    """
    Represents a tensor subclass instance with a __torch_function__ override.
    """

    @staticmethod
    def create(
        tx,
        tensor_variable,
        orig_tensor_variable_source,
        torch_function_fn,
        subclass_type,
        **kwargs,
    ):
        var = TensorWithTFOverrideVariable(
            tensor_variable,
            orig_tensor_variable_source,
            torch_function_fn,
            subclass_type,
            **kwargs,
        )
        # stash the subclass type to rewrap an output tensor if needed
        tx.output.install_global(var.global_class_name(), subclass_type)
        return var

    def __init__(
        self,
        tensor_variable,
        orig_tensor_variable_source,
        subclass_torch_function__func,
        subclass_type,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tensor_variable = tensor_variable
        self.orig_tensor_variable_source = orig_tensor_variable_source
        self.subclass_torch_function__func = subclass_torch_function__func
        self.subclass_type = subclass_type

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # This code block implements inlining the __torch_function__ override
        # of `call_method`.
        from . import GetAttrVariable

        options = VariableTracker.propagate(self, args, kwargs.values())
        # insert unwrapped version of self as the first argument
        # TODO: This is wrong!  When you call the internal __torch_function__,
        # you still get the wrapped version of self, and if you call functions
        # inside __torch_function__, they should come back here.  If we unwrap
        # the tensor immediately, that will not happen.
        # See https://github.com/pytorch/torchdynamo/issues/1951
        args = list(args)
        args.insert(0, self.tensor_variable)
        func_var = GetAttrVariable(self.tensor_variable, name)

        unwrapped = TensorWithTFOverrideVariable.inline_torch_function_unwrapped(
            tx,
            func_var,
            self.orig_tensor_variable_source,
            self.subclass_torch_function__func,
            self.subclass_type,
            options,
            args,
            kwargs,
        )

        # TODO(future PR): implement rewrapping conditional on method presence
        # in `torch.overrides.get_default_nowrap_function()`. It's unclear how
        # to do this easily in the current codebase since the resolution of
        # `GetAttrVariable` depends on the type of the underlying object.

        return TensorWithTFOverrideVariable(
            unwrapped,
            self.orig_tensor_variable_source,
            self.subclass_torch_function__func,
            self.subclass_type,
        )

    def global_class_name(self):
        return f"__subclass_{self.subclass_type.__name__}"

    @staticmethod
    def inline_torch_function_unwrapped(
        tx,
        original_func_var,
        tensor_with_tf_override_source,
        tf_func,
        subclass_type,
        options,
        args,
        kwargs,
    ):
        """
        This function inlines the `__torch_function__` override for `original_func_var`.
        For example, if the user code is

           x1 = torch.sigmoid(x0)

        And `x0` has an override, then:
        * `original_func_var` will be a `VariableTracker` object wrapping `torch.sigmoid`
        * `tensor_with_tf_override_source` will be the `Source` object from
          the original tensor override instance in the beginning of the program
        * `tf_func` will be the custom `__torch_function__` function
        * `subclass_type` will be `type(x0)`

        The caller is expected to properly massage args and kwargs before
        passing them into this function.

        The caller is responsible for wrapping the return value, if needed.
        """
        from . import UserDefinedClassVariable
        from .builder import TupleVariable, VariableBuilder

        source = AttrSource(
            AttrSource(tensor_with_tf_override_source, "__torch_function__"),
            "__func__",
        )
        tf_func_var = VariableBuilder(tx, source)(tf_func)
        type_var = UserDefinedClassVariable(subclass_type, **options)

        # signature:
        # def __torch_function__(cls, func, types, args=(), kwargs=None):
        tf_args = (
            type_var,  # cls
            original_func_var,  # func
            (type_var,),  # types
            TupleVariable(args),  # args
            kwargs,  # kwargs
        )

        # Disable __torch_function__ here to prevent the clone of the
        # example tensor from going into the override.
        with torch._C.DisableTorchFunctionSubclass():
            return tx.inline_user_function_return(tf_func_var, tf_args, {})


class NumpyNdarrayVariable(TensorVariable):
    """
    Represents an np.ndarray, but backed by torch Tensor via torch._numpy.ndarray.
    Use this for Tensor.numpy() call.
    """

    @staticmethod
    def create(tx, proxy, **options):
        from .builder import wrap_fx_proxy_cls

        return wrap_fx_proxy_cls(
            target_cls=NumpyNdarrayVariable,
            tx=tx,
            proxy=proxy,
            **options,
        )

    def var_getattr(self, tx, name):
        # NB: This INTENTIONALLY does not call super(), because there is
        # no intrinsic reason ndarray properties are related to Tensor
        # properties.  The inheritance here is for implementation sharing.

        from ..utils import numpy_attr_wrapper
        from .builder import wrap_fx_proxy

        result = None
        options = VariableTracker.propagate(self)

        example_value = self.as_proxy().node.meta["example_value"]
        example_ndarray = tnp.ndarray(example_value)

        def insert_into_graph():
            return wrap_fx_proxy(
                tx,
                tx.output.create_proxy(
                    "call_function", numpy_attr_wrapper, (self.as_proxy(), name), {}
                ),
                **options,
            )

        if name in ["T", "real", "imag"]:
            proxy = tx.output.create_proxy(
                "call_function",
                numpy_attr_wrapper,
                (self.as_proxy(), name),
                {},
            )
            result = NumpyNdarrayVariable.create(tx, proxy, **options)

        # These are awkward to implement.  The standard playbook for torch._numpy
        # interop is to trace a call into the torch._numpy wrapper which works for
        # Tensor operations.  However, we don't want to do this for calls
        # that don't return Tensors, because in those cases we may not want
        # to trace the attribute access into the graph at all (it is sort
        # of harmless to do so, because AOTAutograd will eliminate them,
        # but it's best not to trace them in to begin with.)  But in any
        # case, tracing these into the graph is like trying to fit a square
        # peg into a round hole; best not to do it.  So instead we
        # painstakingly implement these by hand
        #
        # NB: only ALWAYS specialized attributes can go here; notably,
        # size/shape not allowed!
        elif name in ("ndim", "itemsize"):
            return ConstantVariable.create(getattr(example_ndarray, name), **options)
        elif name in ("shape", "stride"):
            if not free_symbols(r := getattr(example_ndarray, name)):
                return ConstantVariable.create(tuple(int(r) for r in r), **options)
            return insert_into_graph()
        elif name == "size":
            if not free_symbols(r := example_ndarray.size):
                return ConstantVariable.create(int(r), **options)
            return insert_into_graph()
        elif name in ["base", "flags", "dtype"]:
            unimplemented(f"TODO: add support for ndarray.{name}")
        if result is None:
            raise NotImplementedError()
        return result

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate([[self]], [args], [list(kwargs.values())])
        from ..utils import numpy_method_wrapper

        if name in ["__len__", "size"]:
            # delegate back to TensorVariable
            return super().call_method(tx, name, args, kwargs)
        proxy = tx.output.create_proxy(
            "call_function",
            numpy_method_wrapper(name),
            *proxy_args_kwargs([self] + list(args), kwargs),
        )
        return NumpyNdarrayVariable.create(tx, proxy, **options)

    def python_type(self):
        return np.ndarray


class UnspecializedPythonVariable(TensorVariable):
    """
    This is a 1-element tensor represents unspecialized python float/int.
    """

    def __init__(self, proxy: torch.fx.Proxy, **kwargs):
        raw_value = kwargs.pop("raw_value", None)
        need_unwrap = kwargs.pop("need_unwrap", True)
        super().__init__(proxy, **kwargs)
        self.raw_value = raw_value
        self.need_unwrap = need_unwrap

    @classmethod
    def from_tensor_variable(cls, tensor_variable, raw_value, need_unwrap=True):
        # Convert a `TensorVariable` instance into an `UnspecializedPythonVariable` instance.
        return UnspecializedPythonVariable(
            **dict(tensor_variable.__dict__),
            raw_value=raw_value,
            need_unwrap=need_unwrap,
        )

    def as_specialized(self, tx):
        for graph_arg in tx.output.graphargs:
            if graph_arg.source is self.source:
                graph_arg.erase()

        for g in self.guards:
            if g.is_volatile:
                g.create_fn = GuardBuilder.CONSTANT_MATCH

        return ConstantVariable.create(value=self.raw_value, guards=self.guards)


class FakeItemVariable(TensorVariable):
    """An unspecialized python variable which prevents access to the underlying raw value.
    This is needed if item is called on a FakeTensor."""

    def __init__(self, proxy: torch.fx.Proxy, **kwargs):
        need_unwrap = kwargs.pop("need_unwrap", False)
        super().__init__(proxy, **kwargs)
        self.need_unwrap = need_unwrap

    @classmethod
    def from_tensor_variable(cls, tensor_variable):
        return FakeItemVariable(**dict(tensor_variable.__dict__))


class TensorSubclassVariable(VariableTracker):
    def __init__(self, value, *args, **kwargs):
        self.value = value
        super().__init__(*args, **kwargs)

    def call_function(
        self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]
    ) -> VariableTracker:
        if len(args) == 1 and isinstance(args[0], TensorVariable):
            return TensorWithTFOverrideVariable.create(
                tx,
                args[0],
                args[0].source,
                self.value.__torch_function__.__func__,
                self.value,
            )

        return super().call_function(tx, args, kwargs)
