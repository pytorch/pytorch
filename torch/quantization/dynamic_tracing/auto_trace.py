import torch
from torch.fx.node import map_aggregate
from typing import Tuple, Any

from .quantization_state import (
    AutoQuantizationState,
)
from . import auto_trace_rewrite


# TODO(future PR): verify correctness of this for all
# quantizeable modules
def is_leaf(m: torch.nn.Module) -> bool:
    return (
        # allowlist everything in torch.nn except nn.Sequential
        (m.__module__.startswith('torch.nn') and (
            not isinstance(m, torch.nn.Sequential)
        )) or
        # allowlist nni modules, as they inherit from nn.Sequential
        m.__module__.startswith('torch.nn.intrinsic')
    )


def add_auto_observation(
    model : torch.nn.Module,
    example_inputs: Tuple[Any],
) -> torch.nn.Module:
    def convert_to_interception_proxy(x):
        if isinstance(x, torch.Tensor):
            return x.as_subclass(QuantizationInterceptionProxy)  # type: ignore[arg-type]
        else:
            return x

    cur_module = None
    modules_to_introspect = set()
    first_call = True
    module_stack : List[torch.nn.Module] = []
    # Counter for tensor IDs, will be modified inplace by quant state.
    # This is used to track tensors from output ops to input ops. For example,
    # if op_n had a tensor output with id=1, and op_n+2 had a tensor input with
    # id=1, we know that the output of op_n is the input to op_n+2.
    qtensor_id = [0]

    class QuantizationInterceptionProxy(torch.Tensor):
        """
        An override of `torch.Tensor` to enable dynamic tracing for
        quantization.

        For each function with a `__torch_fuction__` override and a parent
        module with auto quantization enabled, this proxy does the following:

        1. calls `_auto_quant_state.func_or_mod_before_hook`.
        2. executes the original function
        3. calls `cur_module._auto_quant_state.func_or_mod_after_hook`
        """

        def __torch_function__(self, func, types, args=(), kwargs=None):
            nonlocal qtensor_id
            kwargs = kwargs if kwargs else {}

            # run "before" hook
            if cur_module and cur_module in modules_to_introspect:
                cur_module._auto_quant_state.func_or_mod_before_hook(
                    func, args, kwargs, first_call, qtensor_id)

            output = super().__torch_function__(func, types, args, kwargs)
            # TODO: is this right? Don't really understand this
            if output is NotImplemented:
                with torch._C.DisableTorchFunction():
                    output = func(*args, **kwargs).as_subclass(
                        QuantizationInterceptionProxy)

            # run "after" hook
            if cur_module and cur_module in modules_to_introspect:
                output = \
                    cur_module._auto_quant_state.func_or_mod_after_hook(
                        func, output, first_call, qtensor_id)

            # mark completion
            if cur_module and cur_module in modules_to_introspect:
                cur_module._auto_quant_state.validate_and_increment(func)

            return output

        def __repr__(self):
            return f'QuantizationInterceptionProxy({super().__repr__()})'

        def __add__(self, other):
            return self.__torch_function__(
                torch.add, [type(self), type(other)], (self, other), {})

        def __mul__(self, other):
            return self.__torch_function__(
                torch.mul, [type(self), type(other)], (self, other), {})

        # TODO(future PR): add other math overrides

    class QuantizationInterceptionModule(type(model)):  # type: ignore[misc]
        """
        An override of user defined subclass of `nn.Module` to enable
        dynamic tracing for quantization.

        `cur_module` keeps track of the current module in the stack.

        During the fist call, an `AutoQuantizationState` object is created and
        attached to each non-leaf modules which we need to check for
        quantizeable operations.

        We override the `__call__` function to do the following for
        any `cur_module` whose parent module has quantization state:

        1. calls parent module's `._auto_quant_state.func_or_mod_before_hook`
        2. executes the original module forward
        3. calls parent module's `_auto_quant_state.module_after_hook`
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
                    parent_module = module_stack[-1] if len(module_stack) else None
                    module_stack.append(self)
                    needs_hooks = parent_module is not None and \
                        hasattr(parent_module, '_auto_quant_state')

                    # "before" hook
                    if needs_hooks:
                        parent_module._auto_quant_state.func_or_mod_before_hook(
                            cur_module, input, kwargs, first_call, qtensor_id)


                    output = orig_module_call(self, *input, **kwargs)

                    # "after" hook
                    if needs_hooks:
                        output = parent_module._auto_quant_state.func_or_mod_after_hook(
                            cur_module, output, first_call, qtensor_id)

                        # mark completion
                        parent_module._auto_quant_state.validate_and_increment(cur_module)

                    return output
                except Exception as e:
                    # import pdb
                    # pdb.set_trace()
                    raise e
                finally:
                    module_stack.pop()
                    cur_module = old_module
            torch.nn.Module.__call__ = record_module
            nonlocal first_call
            try:
                named_modules = list(self.named_modules())
                for k, v in named_modules:
                    if hasattr(v, 'qconfig') and not is_leaf(v):
                        if first_call:
                            v._auto_quant_state = AutoQuantizationState(v.qconfig)
                            modules_to_introspect.add(v)
                        else:
                            if not isinstance(v, AutoQuantizationState):
                                assert hasattr(v, '_auto_quant_state')
                                v._auto_quant_state.reset_to_new_call()
                return super().__call__(*new_input, **new_kwargs)
            finally:
                torch.nn.Module.__call__ = orig_module_call
                first_call = False


    model.__class__ = QuantizationInterceptionModule

    # create the graph
    with torch.no_grad():
        old_training = model.training
        model.eval()
        model(*example_inputs)
        if old_training:
            model.train()

    return model


# TODO(future PR): add serialization support
def add_auto_convert(module : torch.nn.Module) -> torch.nn.Module:
    def convert_to_dispatch_proxy(x):
        if isinstance(x, torch.Tensor):
            return x.as_subclass(QuantizationDispatchProxy)  # type: ignore[arg-type]
        else:
            return x

    class QuantizationDispatchProxy(torch.Tensor):
        """
        An override of `torch.Tensor` to enable dynamic dispatch for
        quantization inference.

        For each function with a `__torch_fuction__` override and a parent
        module with auto quantization enabled, this proxy does the following:

        1. calls `_auto_quant_state.inference_func_or_mod_before_hook`.
        2. calls `_auto_quant_state.get_inference_func_args_kwargs`
        3. executes the function, with target, args and kwargs possibly modified
           by (2)
        4. calls `_auto_quant_state.inference_function_after_hook`.
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
            map_aggregate(args, check)
            map_aggregate(kwargs, check)
            needs_hooks = quantized_arg_present and cur_module and \
                hasattr(cur_module, '_auto_quant_state')

            if needs_hooks:
                args = cur_module._auto_quant_state.inference_func_or_mod_before_hook(
                    func, args, kwargs)
                old_func = func
                func, args, kwargs = \
                    cur_module._auto_quant_state.get_inference_func_or_mod_args_kwargs(
                        func, args, kwargs)

            output = super().__torch_function__(func, types, args, kwargs)

            if needs_hooks:
                output = cur_module._auto_quant_state.inference_func_or_mod_after_hook(
                    func, output)

                cur_module._auto_quant_state.validate_and_increment(old_func)

            return output

        def __repr__(self):
            return f'QuantizationDispatchProxy({super().__repr__()})'

        def __add__(self, other):
            return self.__torch_function__(
                torch.add, [type(self), type(other)], (self, other), {})

        def __mul__(self, other):
            return self.__torch_function__(
                torch.mul, [type(self), type(other)], (self, other), {})

    cur_module = None
    module_stack : List[torch.nn.Module] = []

    assert len(module.__class__.__bases__) == 1

    class QuantizationDispatchModule(module.__class__.__bases__[0]):  # type: ignore[name-defined]
        """
        An override of user defined subclass of `nn.Module` to enable
        dynamic tracing for quantization, after model conversion
        to quantized domain.

        `cur_module` keeps track of the current module in the stack.

        Tensor arguments are converted to `QuantizationDispatchProxy`.

        We override the `__call__` function to do the following for
        any `cur_module` whose parent module has quantization state:

        1. (TODO) calls parent module's `._auto_quant_state.func_or_mod_before_hook`
        2. executes the original module forward
        3. calls parent module's `_auto_quant_state.module_after_hook`
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
                    parent_module = module_stack[-1] if len(module_stack) else None
                    module_stack.append(self)
                    needs_hooks = parent_module is not None and \
                        hasattr(parent_module, '_auto_quant_state')

                    if needs_hooks:
                        first_call = False
                        qtensor_id = []
                        # before hook
                        input = parent_module._auto_quant_state.inference_func_or_mod_before_hook(
                            cur_module, input, kwargs)
                        parent_module._auto_quant_state.get_inference_func_or_mod_args_kwargs(
                            cur_module, input, kwargs)

                    # execute original module forward
                    output = orig_module_call(self, *input, **kwargs)

                    # after hook
                    if needs_hooks:
                        first_call = False
                        qtensor_id = []
                        output = parent_module._auto_quant_state.inference_func_or_mod_after_hook(
                            cur_module, output)

                        parent_module._auto_quant_state.validate_and_increment(cur_module)

                    return output
                finally:
                    module_stack.pop()
                    cur_module = old_module
            torch.nn.Module.__call__ = record_module

            try:
                for k, v in self.named_modules():
                    if hasattr(v, '_auto_quant_state'):
                        v._auto_quant_state.reset_to_new_call()
                rv = super().__call__(*new_input, **new_kwargs)

                def unwrap_proxy(a):
                    if isinstance(a, QuantizationDispatchProxy):
                        a.__class__ = torch.Tensor  # type: ignore[assignment]
                    return a
                rv = map_aggregate(rv, unwrap_proxy)
                return rv
            finally:
                torch.nn.Module.__call__ = orig_module_call

        def rewrite_for_scripting(self):
            return auto_trace_rewrite.rewrite_for_scripting(self)

    module.__class__ = QuantizationDispatchModule

    return module
