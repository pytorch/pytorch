import torch
from typing import Callable, List, Tuple
from .observer import ObserverBase
import copy
import operator
import torch.fx
from torch.fx.node import map_aggregate

def add_auto_observation(model : torch.nn.Module) -> torch.nn.Module:
    def convert_to_proxy(x):
        if isinstance(x, torch.Tensor):
            x.__class__ = QuantizationInterceptionProxy
            return x
        else:
            return x

    cur_module = None
    modules_to_introspect = set()
    first_call = True

    class ArithmeticObservers(object):
        idx : int
        op_observers : List[Tuple[Callable, ObserverBase]]

        def __init__(self):
            self.idx = 0
            self.op_observers = []

        def get_next(self, op):
            if first_call:
                self.op_observers.insert(self.idx, (cur_module.qconfig.activation(), op))
            rv = self.op_observers[self.idx]
            self.idx += 1
            return rv

    class QuantizationInterceptionProxy(torch.Tensor):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs if kwargs else {}
            output = super().__torch_function__(func, types, args, kwargs)

            if cur_module and cur_module in modules_to_introspect:
                if func in {torch.Tensor.add, torch.Tensor.mul, torch.add, torch.mul,
                            torch.cat}:
                    try:
                        observer, prev_op = cur_module._arithmetic_observers.get_next(func)
                    except IndexError:
                        raise RuntimeError(f'Encountered arithmetic operation {torch.typename(func)} but we have '
                                           f'encountered fewer arithmetic operations in previous calibration runs. '
                                           f'This likely indicates that the program contains dynamic control flow. '
                                           f' Quantization is not defined over dynamic control flow!')
                    if prev_op != func:
                        raise RuntimeError(f'Encountered arithmetic operation {torch.typename(func)} but previously '
                                           f'recorded operation was {torch.typename(prev_op)}!. This likely indicates '
                                           f'that the program contains dynamic control flow. Quantization is not '
                                           f'defined over dynamic control flow!')
                    assert isinstance(output, torch.Tensor)
                    observer(output)

            return output

        def __repr__(self):
            return f'QuantizationInterceptionProxy({super().__repr__()})'

        def __add__(self, other):
            return self.__torch_function__(torch.add, [type(self), type(other)], (self, other), {})

        def __mul__(self, other):
            return self.__torch_function__(torch.mul, [type(self), type(other)], (self, other), {})

    class QuantizationInterceptionModule(type(model)):
        __interception_module__ = True

        def __call__(self, *input, **kwargs):
            new_input = map_aggregate(input, convert_to_proxy)
            new_kwargs = map_aggregate(kwargs, convert_to_proxy)
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
                            setattr(v, '_arithmetic_observers', ArithmeticObservers())
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

def add_auto_convert(module : torch.nn.Module) -> torch.nn.Module:
    replacement_mapping = {
        torch.Tensor.add: torch.ops.quantized.add,
        torch.add: torch.ops.quantized.add,
        operator.add: torch.ops.quantized.add,
        torch.Tensor.mul: torch.ops.quantized.mul,
        torch.mul: torch.ops.quantized.mul,
        operator.mul: torch.ops.quantized.mul,
        torch.cat: torch.ops.quantized.cat,
    }

    def convert_to_proxy(x):
        if isinstance(x, torch.Tensor):
            x.__class__ = QuantizationDispatchProxy
            return x
        else:
            return x

    class QuantizationDispatchProxy(torch.Tensor):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs if kwargs else {}
            quantized_arg_present = False

            def check(a):
                if isinstance(a, QuantizationDispatchProxy):
                    a.__class__ = torch.Tensor
                    try:
                        if a.is_quantized:
                            nonlocal quantized_arg_present
                            quantized_arg_present = True
                    finally:
                        a.__class__ = QuantizationDispatchProxy
            torch.fx.node.map_aggregate(args, check)
            torch.fx.node.map_aggregate(kwargs, check)
            if quantized_arg_present and cur_module and hasattr(cur_module, '_arithmetic_observers'):
                if func in replacement_mapping:
                    try:
                        observer, prev_op = cur_module._arithmetic_observers.get_next(func)
                    except IndexError:
                        raise RuntimeError(f'Encountered arithmetic operation {torch.typename(func)} but we have '
                                           f'encountered fewer arithmetic operations in previous calibration runs. '
                                           f'This likely indicates that the program contains dynamic control flow. '
                                           f' Quantization is not defined over dynamic control flow!')
                    if prev_op != func:
                        raise RuntimeError(f'Encountered arithmetic operation {torch.typename(func)} but previously '
                                           f'recorded operation was {torch.typename(prev_op)}!. This likely indicates '
                                           f'that the program contains dynamic control flow. Quantization is not '
                                           f'defined over dynamic control flow!')

                    scale, zp = observer.calculate_qparams()
                    kwargs.update({'scale': scale, 'zero_point': zp})
                    func = replacement_mapping[func]

            # print('QuantizationDispatchProxy.__torch_function__', func, args, kwargs, flush=True)
            return super().__torch_function__(func, types, args, kwargs)

        def __repr__(self):
            return f'QuantizationDispatchProxy({super().__repr__()})'

        def __add__(self, other):
            return self.__torch_function__(torch.add, [type(self), type(other)], (self, other), {})

        def __mul__(self, other):
            return self.__torch_function__(torch.mul, [type(self), type(other)], (self, other), {})

    cur_module = None

    assert len(module.__class__.__bases__) == 1

    class QuantizationDispatchModule(module.__class__.__bases__[0]):
        def __call__(self, *input, **kwargs):
            new_input = map_aggregate(input, convert_to_proxy)
            new_kwargs = map_aggregate(kwargs, convert_to_proxy)
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
                        a.__class__ = torch.Tensor
                    return a
                rv = torch.fx.node.map_aggregate(rv, unwrap_proxy)
                return rv
            finally:
                torch.nn.Module.__call__ = orig_module_call

        def rewrite(self):
            def rewrite_helper(mod : torch.nn.Module):
                copied = copy.copy(mod)
                for name, child in mod.named_children():
                    setattr(copied, name, rewrite_helper(child))

                if hasattr(mod, '_arithmetic_observers') and len(mod._arithmetic_observers.op_observers) != 0:
                    copied._arithmetic_observers.idx = 0

                    class AllModuleTracer(torch.fx.Tracer):
                        def is_leaf_module(self, m, module_qualified_name) -> bool:
                            return True

                        def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
                            if target == operator.add:
                                target = torch.add
                            if target == operator.mul:
                                target = torch.mul
                            if target in replacement_mapping:
                                try:
                                    observer, prev_op = self.root._arithmetic_observers.get_next(target)
                                except IndexError:
                                    raise RuntimeError(f'Encountered arithmetic operation {torch.typename(target)} but we have '
                                                       f'encountered fewer arithmetic operations in previous calibration runs. '
                                                       f'This likely indicates that the program contains dynamic control flow. '
                                                       f' Quantization is not defined over dynamic control flow!')
                                if prev_op != target:
                                    raise RuntimeError(f'Encountered arithmetic operation {torch.typename(target)} but previously '
                                                       f'recorded operation was {torch.typename(prev_op)}!. This likely indicates '
                                                       f'that the program contains dynamic control flow. Quantization is not '
                                                       f'defined over dynamic control flow!')

                                scale, zp = observer.calculate_qparams()
                                kwargs.update({'scale': scale.item(), 'zero_point': int(zp.item())})
                                target = replacement_mapping[target]
                            return super().create_node(kind, target, args, kwargs, name, type_expr)

                    graph = AllModuleTracer().trace(copied)
                    return torch.fx.GraphModule(copied, graph, copied.__class__.__name__)
                else:
                    return copied

            return rewrite_helper(self)

    module.__class__ = QuantizationDispatchModule

    return module
