import torch
import operator
from typing import Callable, List, Tuple
from ..observer import ObserverBase

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


# TODO(future PR): maybe better name
# TODO(future PR): add serialization support
class AutoQuantizationState(object):
    idx : int
    op_observers : List[Tuple[ObserverBase, Callable]]

    def __init__(self, qconfig):
        self.idx = 0
        self.op_observers = []
        # TODO(future PR): change this to the subset of qconfig_dict
        # relevant to the parent module
        self.qconfig = qconfig

    def insert_observer(self, op):
        self.op_observers.insert(self.idx, (self.qconfig.activation(), op))

    def get_next(self, prev_op):
        try:
            observer, func = self.op_observers[self.idx]
        except IndexError:
            _raise_obs_not_found_error(func)
        if prev_op != func:
            _raise_obs_op_mismatch(func, prev_op)
        self.idx += 1
        return observer, func

    def observe(self, tensor_to_observe, func):
        observer, cur_func = self.get_next(func)
        return observer(tensor_to_observe)

    def maybe_update_func_args_kwargs_for_quantized_inference(
            self, func, args, kwargs, unwrap_scale_zp=False):
        if func in fp32_to_int8_fun_mapping:
            observer, prev_op = self.get_next(func)
            scale, zp = observer.calculate_qparams()
            # TODO(future PR): remove this boolean flag
            if not unwrap_scale_zp:
                kwargs.update({'scale': scale, 'zero_point': zp})
            else:
                kwargs.update({'scale': scale.item(), 'zero_point': zp.item()})
            func = fp32_to_int8_fun_mapping[func]
        return func, args, kwargs

    def after_observed_function_hook(self, func, output, first_call):
        """
        This function is called after a function call
        """
        if func in {torch.Tensor.add, torch.Tensor.mul, torch.add, torch.mul,
                    torch.cat}:
            if first_call:
                self.insert_observer(func)
            output = self.observe(output, func)
        return output

    def reset_to_new_call(self):
        """
        Resets the internal op counter to start a new inference call
        """
        self.idx = 0
