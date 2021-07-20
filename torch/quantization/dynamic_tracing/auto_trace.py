import torch
from typing import Callable, List, Tuple
from ..observer import ObserverBase
import copy
import operator
import torch.fx
from torch.fx.node import map_aggregate


fp32_to_int8_fun_mapping = {
    torch.Tensor.add: torch.ops.quantized.add,
    torch.add: torch.ops.quantized.add,
    operator.add: torch.ops.quantized.add,
    torch.Tensor.mul: torch.ops.quantized.mul,
    torch.mul: torch.ops.quantized.mul,
    operator.mul: torch.ops.quantized.mul,
    torch.cat: torch.ops.quantized.cat,
}


def _raise_obs_not_found_error(func):
    raise RuntimeError(
        f'Encountered arithmetic operation {torch.typename(func)} but we have '
        f'encountered fewer arithmetic operations in previous calibration runs. '
        f'This likely indicates that the program contains dynamic control flow. '
        f' Quantization is not defined over dynamic control flow!')

def _raise_obs_op_mismatch(func, prev_op):
    raise RuntimeError(
        f'Encountered arithmetic operation {torch.typename(func)} but previously '
        f'recorded operation was {torch.typename(prev_op)}!. This likely indicates '
        f'that the program contains dynamic control flow. Quantization is not '
        f'defined over dynamic control flow!')


class ArithmeticObservers(object):
    idx : int
    op_observers : List[Tuple[ObserverBase, Callable]]

    def __init__(self):
        self.idx = 0
        self.op_observers = []

    def insert_observer(self, op, activation_ctr):
        self.op_observers.insert(self.idx, (activation_ctr(), op))

    def get_next(self, op):
        observer, func = self.op_observers[self.idx]
        self.idx += 1
        return observer, func


class AllModuleTracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name) -> bool:
        return True

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        if target == operator.add:
            target = torch.add
        if target == operator.mul:
            target = torch.mul
        if target in fp32_to_int8_fun_mapping:
            try:
                first_call = False
                observer, prev_op = self.root._arithmetic_observers.get_next(  # type: ignore[union-attr, operator]
                    target)
            except IndexError:
                _raise_obs_not_found_error(target)
            if prev_op != target:
                _raise_obs_op_mismatch(target, prev_op)
            scale, zp = observer.calculate_qparams()
            kwargs.update({'scale': scale.item(), 'zero_point': int(zp.item())})
            target = fp32_to_int8_fun_mapping[target]
        return super().create_node(kind, target, args, kwargs, name, type_expr)


# TODO(future PR): add serialization support
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
                    output = func(*args, **kwargs).as_subclass(QuantizationInterceptionProxy)

            if cur_module and cur_module in modules_to_introspect:
                # TODO(future PR): abstract the quantization related logic away
                # from this file
                if func in {torch.Tensor.add, torch.Tensor.mul, torch.add, torch.mul,
                            torch.cat}:
                    try:
                        if first_call:
                            cur_module._arithmetic_observers.insert_observer(
                                func, cur_module.qconfig.activation)
                        observer, prev_op = cur_module._arithmetic_observers.get_next(func)
                    except IndexError:
                        _raise_obs_not_found_error(func)
                    if prev_op != func:
                        _raise_obs_op_mismatch(func, prev_op)
                    assert isinstance(output, torch.Tensor)
                    observer(output)

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

        During the fist call, an `ArithmeticObservers` object is created and
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
                for k, v in self.named_modules():
                    # TODO: is this valid?
                    if hasattr(v, 'qconfig'):
                        if first_call:
                            v._arithmetic_observers = ArithmeticObservers()
                            modules_to_introspect.add(v)
                        else:
                            assert hasattr(v, '_arithmetic_observers')
                            v._arithmetic_observers.idx = 0

                return super().__call__(*new_input, **new_kwargs)
            finally:
                torch.nn.Module.__call__ = orig_module_call
                first_call = False


    model.__class__ = QuantizationInterceptionModule

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
            if quantized_arg_present and cur_module and hasattr(cur_module, '_arithmetic_observers'):
                # TODO(future PR): abstract quantization related logic away
                # from this file
                if func in fp32_to_int8_fun_mapping:
                    try:
                        observer, prev_op = cur_module._arithmetic_observers.get_next(func)
                    except IndexError:
                        _raise_obs_not_found_error(func)
                    if prev_op != func:
                        _raise_obs_op_mismatch(func, prev_op)

                    scale, zp = observer.calculate_qparams()
                    kwargs.update({'scale': scale, 'zero_point': zp})
                    func = fp32_to_int8_fun_mapping[func]

            return super().__torch_function__(func, types, args, kwargs)

        def __repr__(self):
            return f'QuantizationDispatchProxy({super().__repr__()})'

        def __add__(self, other):
            return self.__torch_function__(torch.add, [type(self), type(other)], (self, other), {})

        def __mul__(self, other):
            return self.__torch_function__(torch.mul, [type(self), type(other)], (self, other), {})

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
        look up scale and zero_point when necessary from the `ArithmeticObservers`
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
                    if hasattr(v, '_arithmetic_observers'):
                        v._arithmetic_observers.idx = 0
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

                if hasattr(mod, '_arithmetic_observers') and \
                        len(mod._arithmetic_observers.op_observers) != 0:  # type: ignore[union-attr, arg-type]
                    copied._arithmetic_observers.idx = 0  # type: ignore[union-attr]

                    graph = AllModuleTracer().trace(copied)
                    return torch.fx.GraphModule(copied, graph, copied.__class__.__name__)
                else:
                    return copied

            return rewrite_helper(self)

    module.__class__ = QuantizationDispatchModule

    return module
