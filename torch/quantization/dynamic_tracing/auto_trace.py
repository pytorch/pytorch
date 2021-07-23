import torch
import copy
import operator
import torch.fx
from torch.fx.node import map_aggregate

from .quantization_state import (
    AutoQuantizationState,
)


def add_auto_observation(model : torch.nn.Module) -> torch.nn.Module:
    def convert_to_interception_proxy(x):
        if isinstance(x, torch.Tensor):
            return x.as_subclass(QuantizationInterceptionProxy)  # type: ignore[arg-type]
        else:
            return x

    cur_module = None
    modules_to_introspect = set()
    first_call = True

    class QuantizationInterceptionProxy(torch.Tensor):
        """
        An override of `torch.Tensor` to enable dynamic tracing for
        quantization, used to dynamically create observers and feed data
        through them.

        If the observers have not been created yet (on the first pass
        of tensors through the current module), an observer is created
        and recorded in a map along with the current observer index and
        the current op type.

        The correct observer is looked up by observer index and op type,
        and the tensor is observed before being returned to the caller.
        """

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs if kwargs else {}
            output = super().__torch_function__(func, types, args, kwargs)

            # TODO: is this right? Don't really understand this
            if output is NotImplemented:
                with torch._C.DisableTorchFunction():
                    output = func(*args, **kwargs).as_subclass(
                        QuantizationInterceptionProxy)

            if cur_module and cur_module in modules_to_introspect:
                output = \
                    cur_module._auto_quantization_state.after_observed_function_hook(
                        func, output, first_call)
            return output

        def __repr__(self):
            return f'QuantizationInterceptionProxy({super().__repr__()})'

        def __add__(self, other):
            return self.__torch_function__(
                torch.add, [type(self), type(other)], (self, other), {})

        def __mul__(self, other):
            return self.__torch_function__(
                torch.mul, [type(self), type(other)], (self, other), {})

    class QuantizationInterceptionModule(type(model)):  # type: ignore[misc]
        """
        An override of user defined subclass of `nn.Module` to enable
        dynamic tracing for quantization.

        `cur_module` keeps track of the current module in the stack.

        During the fist call, an `AutoQuantizationState` object is created and
        attached to the current module.

        Tensor arguments are converted to `QuantizationInterceptionProxy`
        hooked up to the instance of `ArithmeticObserver` from `cur_module`.
        This ensures that observers are created on the first call, and needed
        values are observed on all calls.
        """

        __interception_module__ = True

        def __call__(self, *input, **kwargs):
            new_input = map_aggregate(input, convert_to_interception_proxy)
            new_kwargs = map_aggregate(kwargs, convert_to_interception_proxy)
            orig_module_call = torch.nn.Module.__call__

            def record_module(self, *input, **kwargs):
                nonlocal cur_module
                old_module = cur_module
                cur_module = self
                try:
                    return orig_module_call(self, *input, **kwargs)
                finally:
                    cur_module = old_module
            torch.nn.Module.__call__ = record_module
            nonlocal first_call
            try:
                named_modules = list(self.named_modules())
                for k, v in named_modules:
                    # TODO(future PR): verify correctness of this for all
                    # quantizeable modules
                    is_leaf = (
                        # allowlist everything in torch.nn except nn.Sequential
                        (v.__module__.startswith('torch.nn') and (
                            not isinstance(v, torch.nn.Sequential)
                        )) or
                        # allowlist nni modules, as they inherit from nn.Sequential
                        v.__module__.startswith('torch.nn.intrinsic')
                    )
                    if hasattr(v, 'qconfig') and not is_leaf:
                        if first_call:
                            v._auto_quantization_state = AutoQuantizationState(v.qconfig)
                            modules_to_introspect.add(v)
                        else:
                            if not isinstance(v, AutoQuantizationState):
                                assert hasattr(v, '_auto_quantization_state')
                                v._auto_quantization_state.reset_to_new_call()

                return super().__call__(*new_input, **new_kwargs)
            finally:
                torch.nn.Module.__call__ = orig_module_call
                first_call = False


    model.__class__ = QuantizationInterceptionModule

    return model


class AllModuleTracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name) -> bool:
        return True

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        if target == operator.add:
            target = torch.add
        if target == operator.mul:
            target = torch.mul
        target, args, kwargs = \
            self.root._auto_quantization_state.maybe_update_func_args_kwargs_for_quantized_inference(
                target, args, kwargs, unwrap_scale_zp=True)
        return super().create_node(kind, target, args, kwargs, name, type_expr)


# TODO(future PR): add serialization support
def add_auto_convert(module : torch.nn.Module) -> torch.nn.Module:
    def convert_to_dispatch_proxy(x):
        if isinstance(x, torch.Tensor):
            return x.as_subclass(QuantizationDispatchProxy)  # type: ignore[arg-type]
        else:
            return x

    class QuantizationDispatchProxy(torch.Tensor):
        """
        An override of `torch.Tensor` to enable dynamic tracing for
        quantization, used for dynamically replacing fp32 ops with
        quantized ops, and looking up the necessary scale and zero_point.
        """

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs if kwargs else {}
            quantized_arg_present = False

            def check(a):
                if isinstance(a, QuantizationDispatchProxy):
                    a.__class__ = torch.Tensor  # type: ignore[assignment]
                    try:
                        if a.is_quantized:
                            nonlocal quantized_arg_present
                            quantized_arg_present = True
                    finally:
                        a.__class__ = QuantizationDispatchProxy
            torch.fx.node.map_aggregate(args, check)
            torch.fx.node.map_aggregate(kwargs, check)
            if quantized_arg_present and cur_module and hasattr(cur_module, '_auto_quantization_state'):
                func, args, kwargs = \
                    cur_module._auto_quantization_state.maybe_update_func_args_kwargs_for_quantized_inference(func, args, kwargs)

            return super().__torch_function__(func, types, args, kwargs)

        def __repr__(self):
            return f'QuantizationDispatchProxy({super().__repr__()})'

        def __add__(self, other):
            return self.__torch_function__(
                torch.add, [type(self), type(other)], (self, other), {})

        def __mul__(self, other):
            return self.__torch_function__(
                torch.mul, [type(self), type(other)], (self, other), {})

    cur_module = None

    assert len(module.__class__.__bases__) == 1

    class QuantizationDispatchModule(module.__class__.__bases__[0]):  # type: ignore[name-defined]
        """
        An override of user defined subclass of `nn.Module` to enable
        dynamic tracing for quantization, after model conversion
        to quantized domain.

        `cur_module` keeps track of the current module in the stack.

        Tensor arguments are converted to `QuantizationDispatchProxy`, which
        knows how to dynamically call quantized ops from fp32 equivalents, and
        look up scale and zero_point when necessary from the `AutoQuantizationState`
        object attached to the current module.
        """

        def __call__(self, *input, **kwargs):
            new_input = map_aggregate(input, convert_to_dispatch_proxy)
            new_kwargs = map_aggregate(kwargs, convert_to_dispatch_proxy)
            orig_module_call = torch.nn.Module.__call__

            def record_module(self, *input, **kwargs):
                nonlocal cur_module
                old_module = cur_module
                cur_module = self
                try:
                    return orig_module_call(self, *input, **kwargs)
                finally:
                    cur_module = old_module
            torch.nn.Module.__call__ = record_module

            try:
                for k, v in self.named_modules():
                    if hasattr(v, '_auto_quantization_state'):
                        v._auto_quantization_state.reset_to_new_call()
                rv = super().__call__(*new_input, **new_kwargs)

                def unwrap_proxy(a):
                    if isinstance(a, QuantizationDispatchProxy):
                        a.__class__ = torch.Tensor  # type: ignore[assignment]
                    return a
                rv = torch.fx.node.map_aggregate(rv, unwrap_proxy)
                return rv
            finally:
                torch.nn.Module.__call__ = orig_module_call

        # TODO(future PR): handle cases where the module is not symbolically
        # traceable
        def rewrite(self):
            def rewrite_helper(mod : torch.nn.Module):
                copied = copy.copy(mod)
                for name, child in mod.named_children():
                    setattr(copied, name, rewrite_helper(child))

                if hasattr(mod, '_auto_quantization_state') and \
                        len(mod._auto_quantization_state.idx_to_observer) != 0:  # type: ignore[union-attr, arg-type]
                    copied._auto_quantization_state.reset_to_new_call()  # type: ignore[union-attr]

                    graph = AllModuleTracer().trace(copied)
                    return torch.fx.GraphModule(copied, graph, copied.__class__.__name__)
                else:
                    return copied

            return rewrite_helper(self)

    module.__class__ = QuantizationDispatchModule

    return module
