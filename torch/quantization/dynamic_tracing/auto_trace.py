import torch
import copy
import operator
import torch.fx
from torch.fx.node import map_aggregate
from typing import Tuple, Any

from .quantization_state import (
    AutoQuantizationState,
)


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

        1. calls `_auto_quant_state.function_before_hook`.
        2. executes the original function
        3. calls `cur_module._auto_quant_state.function_after_hook`
        """

        def __torch_function__(self, func, types, args=(), kwargs=None):
            nonlocal qtensor_id
            kwargs = kwargs if kwargs else {}

            # run "before" hook
            if cur_module and cur_module in modules_to_introspect:
                cur_module._auto_quant_state.function_before_hook(
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
                    cur_module._auto_quant_state.function_after_hook(
                        func, output, first_call, qtensor_id)

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
        attached to each non-leaf modules which we need to check for
        quantizeable operations.

        We override the `__call__` function to do the following for
        any `cur_module` whose parent module has quantization state:

        1. calls parent module's `._auto_quant_state.module_before_hook`
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
                    # print(type(self), [type(x) for x in module_stack])

                    # "before" hook
                    if parent_module is not None and hasattr(parent_module, '_auto_quant_state'):
                        parent_module._auto_quant_state.module_before_hook(
                            cur_module, input, kwargs, first_call, qtensor_id)


                    output = orig_module_call(self, *input, **kwargs)

                    # "after" hook
                    if parent_module is not None and hasattr(parent_module, '_auto_quant_state'):
                        output = parent_module._auto_quant_state.module_after_hook(
                            cur_module, output, first_call, qtensor_id)
                    return output
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


# TODO(future PR): need ability to create N nodes from 1 node, for
# reference patterns
class AllModuleTracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name) -> bool:
        return True

    def _maybe_update_args_with_quants(self, args):
        # insert quants for inputs, if needed
        input_args_quant_info = self.root._auto_quant_state.get_input_args_quant_info()
        if len(input_args_quant_info):
            new_args = []
            for idx, input_arg_quant_info in enumerate(input_args_quant_info):
                if input_arg_quant_info is None:
                    new_args.append(args[idx])
                else:
                    # create a quant node
                    scale, zp = input_arg_quant_info
                    quant = super().create_node(
                        'call_function', torch.quantize_per_tensor,
                        (args[idx], scale.item(), zp.item(), torch.quint8), {}, None, None)
                    new_args.append(quant)
            args = tuple(new_args)
        return args

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        if target == operator.add:
            target = torch.add
        if target == operator.mul:
            target = torch.mul

        if kind == 'call_function':
            # insert quants for inputs, if needed
            args = self._maybe_update_args_with_quants(args)
            target, args, kwargs = \
                self.root._auto_quant_state.get_inference_func_args_kwargs(
                    target, args, kwargs, unwrap_scale_zp=True)

        elif kind == 'call_module':
            # TODO: handle fqn

            # insert quants for inputs, if needed
            args = self._maybe_update_args_with_quants(args)
            module_instance = getattr(self.root, target)
            self.root._auto_quant_state.get_inference_mod_args_kwargs(
                module_instance)

        out = super().create_node(kind, target, args, kwargs, name, type_expr)
        return out


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

        1. (future PR) calls `_auto_quant_state.inference_function_before_hook`.
        2. calls `_auto_quant_state.get_inference_func_args_kwargs`
        3. executes the function, with target, args and kwargs possibly modified
           by (2)
        4. (future PR) calls `_auto_quant_state.inference_function_after_hook`.

        TODO: add the before and after hooks (for future quant/dequant insertion)
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
            if quantized_arg_present and cur_module and \
                    hasattr(cur_module, '_auto_quant_state'):

                # TODO(future PR): before hook
                args = cur_module._auto_quant_state.inference_function_before_hook(
                    func, args, kwargs)

                func, args, kwargs = \
                    cur_module._auto_quant_state.get_inference_func_args_kwargs(
                        func, args, kwargs)

            output = super().__torch_function__(func, types, args, kwargs)

            # TODO(future PR): after hook

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

        1. (TODO) calls parent module's `._auto_quant_state.module_before_hook`
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

                    # before hook
                    if parent_module is not None and hasattr(parent_module, '_auto_quant_state'):
                        first_call = False
                        qtensor_id = []
                        input = parent_module._auto_quant_state.inference_module_before_hook(
                            cur_module, input)

                    # execute original module forward
                    output = orig_module_call(self, *input, **kwargs)

                    # after hook
                    if parent_module is not None and hasattr(parent_module, '_auto_quant_state'):
                        first_call = False
                        qtensor_id = []
                        output = parent_module._auto_quant_state.module_after_hook(
                            cur_module, output, first_call, qtensor_id)
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

                if hasattr(mod, '_auto_quant_state') and \
                        len(mod._auto_quant_state.tensor_id_to_observer) != 0:  # type: ignore[union-attr, arg-type]
                    copied._auto_quant_state.reset_to_new_call()  # type: ignore[union-attr]

                    graph = AllModuleTracer().trace(copied)
                    return torch.fx.GraphModule(copied, graph, copied.__class__.__name__)
                else:
                    return copied

            return rewrite_helper(self)

    module.__class__ = QuantizationDispatchModule

    return module
