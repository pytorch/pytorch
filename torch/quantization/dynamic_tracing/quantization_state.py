import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import operator
import collections
from typing import Callable, List, Tuple, Any, Optional
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

functions_supported_by_quantization = set([
    torch.Tensor.add, torch.Tensor.mul, torch.add, torch.mul, torch.cat,
])

module_types_supported_by_quantization = set([
    nn.Conv2d,
    nnq.Conv2d,
    nn.intrinsic.modules.fused.ConvReLU2d,
    nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d,
])

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


# TODO(future PR): figure out if there is a better option than namedtuple
SeenOp = collections.namedtuple(
    'SeenOp',
    [
        'idx',
        'type',
        'input_tensor_ids',
        'output_tensor_ids',
    ],
)
def seen_op_repr(self) -> str:
    s = f"(type): {self.type}\n"
    s += f"     (input_tensor_ids): {self.input_tensor_ids}\n"
    s += f"     (output_tensor_ids): {self.output_tensor_ids}"
    return s

SeenOp.__repr__ = seen_op_repr

QTensorInfo = collections.namedtuple(
    'QTensorInfo',
    [
        'id',  # tensor ID
        'inf_dtype',  # dtype at inference
    ],
)


# TODO(future PR): maybe better name
# TODO(future PR): add serialization support
class AutoQuantizationState(torch.nn.Module):
    """
    Contains state necessary to perform auto quantization on the parent
    `nn.Module` instance.
    """

    idx : int

    def __init__(self, qconfig):
        super().__init__()
        self.idx = 0
        # TODO(future PR): change this to the subset of qconfig_dict
        # relevant to the parent module
        self.qconfig = qconfig
        # this is a ModuleDict in order to properly register observers
        # to be within the module hierarchy.
        self.tensor_id_to_observer = torch.nn.ModuleDict()

        # TODO(future PR): include kwargs
        self.idx_to_seen_ops = {}

    def extra_repr(self) -> str:
        s = ""
        s += "(seen_ops): {\n"
        for k, v in self.idx_to_seen_ops.items():
            s += f"  {k}: {v}\n"
        s += "}"
        return s

    def _insert_observer(self, tensor_id: int) -> None:
        self.tensor_id_to_observer[str(tensor_id)] = self.qconfig.activation()

    def _validate_and_increment(self, prev_op: Callable) -> None:
        try:
            seen_op = self.idx_to_seen_ops[str(self.idx)]
            func = seen_op.type
        except IndexError:
            _raise_obs_not_found_error(func)
        if prev_op != func:
            _raise_obs_op_mismatch(func, prev_op)
        self.idx += 1

    def function_before_hook(
        self,
        func: Callable,
        args, kwargs,
        first_call: bool,
        qtensor_id: List[int],
    ) -> None:
        if func not in functions_supported_by_quantization:
            return

        if first_call:
            arg_tensor_ids = []
            for arg in args:
                # If a tensor does not have an ID, add it. This allows
                # us to track inputs shared by multiple quantizeable modules.
                if not hasattr(arg, '_qtensor_info'):
                    arg._qtensor_info = QTensorInfo(qtensor_id[0], torch.float)
                    qtensor_id[0] += 1
                arg_tensor_ids.append(arg._qtensor_info)

                # if the existing inf_dtype is not torch.quint8, add an observer
                # which will be converted to a quant later
                # TODO(future PR): share these observers if multiple ops need
                # this quant
                if arg._qtensor_info.inf_dtype != torch.quint8:
                    tensor_id = arg._qtensor_info.id
                    self.tensor_id_to_observer[str(tensor_id)] = \
                        self.qconfig.activation()

            key = str(self.idx)
            if key not in self.idx_to_seen_ops:
                self.idx_to_seen_ops[key] = SeenOp(self.idx, func, arg_tensor_ids, [])

        else:
            seen_op = self.idx_to_seen_ops[str(self.idx)]
            for input_arg_idx, input_arg in enumerate(seen_op.input_tensor_ids):
                tensor_id = input_arg.id
                # TODO: do not run this twice on input and output
                if str(tensor_id) in self.tensor_id_to_observer:
                    observer = self.tensor_id_to_observer[str(tensor_id)]
                    # TODO: return this to the caller
                    observer(args[input_arg_idx])

    def function_after_hook(
        self,
        func: Callable,
        output: Any,
        first_call: bool,
        qtensor_id: List[int],
    ) -> Any:
        """
        This function is called after a function call
        """
        if func in functions_supported_by_quantization:
            if first_call:
                self._insert_observer(str(qtensor_id[0]))

                # TODO(future PR): check if _qtensor_id needs to become an actual
                # attribute of Tensor
                output._qtensor_info = QTensorInfo(qtensor_id[0], torch.quint8)
                self.idx_to_seen_ops[str(self.idx)].output_tensor_ids.append(
                    output._qtensor_info)
                qtensor_id[0] += 1
                self.idx += 1
            else:
                self._validate_and_increment(func)
                # TODO(future PR): other output types
                seen_op = self.idx_to_seen_ops[str(self.idx - 1)]
                tensor_id = seen_op.output_tensor_ids[0].id
                obs = self.tensor_id_to_observer[str(tensor_id)]
                output = obs(output)
        return output

    def module_before_hook(
        self,
        mod: torch.nn.Module,
        input: Any,
        kwargs,
        first_call: bool,
        qtensor_id: List[int],
    ) -> None:
        for module_type in module_types_supported_by_quantization:
            if not isinstance(mod, module_type):
                continue
            if first_call:
                arg_tensor_ids = []
                for arg in input:
                    # If a tensor does not have an ID, add it. This allows
                    # us to track inputs shared by multiple quantizeable modules.
                    if not hasattr(arg, '_qtensor_info'):
                        arg._qtensor_info = QTensorInfo(qtensor_id[0], torch.float)
                        qtensor_id[0] += 1
                    arg_tensor_ids.append(arg._qtensor_info)

                    # if the existing inf_dtype is not torch.quint8, add an observer
                    # which will be converted to a quant later
                    # TODO(future PR): share these observers if multiple ops need
                    # this quant
                    if arg._qtensor_info.inf_dtype != torch.quint8:
                        tensor_id = arg._qtensor_info.id
                        self.tensor_id_to_observer[str(tensor_id)] = \
                            self.qconfig.activation()
                key = str(self.idx)
                if key not in self.idx_to_seen_ops:
                    self.idx_to_seen_ops[key] = SeenOp(self.idx, type(mod), arg_tensor_ids, [])
            else:
                seen_op = self.idx_to_seen_ops[str(self.idx)]
                for input_arg_idx, input_arg in enumerate(seen_op.input_tensor_ids):
                    tensor_id = input_arg.id
                    if str(tensor_id) in self.tensor_id_to_observer:
                        observer = self.tensor_id_to_observer[str(tensor_id)]
                        # TODO: return this to the caller
                        observer(input[input_arg_idx])

    def module_after_hook(
        self,
        mod: torch.nn.Module,
        output: Any,
        first_call: bool,
        qtensor_id: List[int],
    ) -> Any:
        """
        This function is called after a module call
        """
        for module_type in module_types_supported_by_quantization:
            if isinstance(mod, module_type):
                if first_call:
                    # self.tensor_id_to_observer[str(self.idx)] = None
                    output._qtensor_info = QTensorInfo(qtensor_id[0], torch.quint8)
                    self.idx_to_seen_ops[str(self.idx)].output_tensor_ids.append(
                        output._qtensor_info)
                    qtensor_id[0] += 1
                self.idx += 1
                break

        return output

    def inference_function_before_hook(
        self,
        func: Callable,
        args,
        kwargs,
    ) -> Tuple[Any]:
        if func in functions_supported_by_quantization:
            seen_op = self.idx_to_seen_ops[str(self.idx)]

            new_inputs = []
            for input_arg_idx, input_arg in enumerate(seen_op.input_tensor_ids):
                tensor_id = input_arg.id
                if str(tensor_id) in self.tensor_id_to_observer:
                    observer = self.tensor_id_to_observer[str(tensor_id)]
                    # TODO: return this to the caller
                    scale, zp = observer.calculate_qparams()
                    quantized_input = \
                        torch.quantize_per_tensor(args[input_arg_idx], scale, zp, torch.quint8)
                    new_inputs.append(quantized_input)
                else:
                    new_inputs.append(args[input_arg_idx])
            return tuple(new_inputs)

        return args

    def inference_module_before_hook(
        self,
        mod: torch.nn.Module,
        input: Any,
    ) -> Tuple[Any]:
        for module_type in module_types_supported_by_quantization:
            if not isinstance(mod, module_type):
                continue
            seen_op = self.idx_to_seen_ops[str(self.idx)]
            new_inputs = []
            for input_arg_idx, input_arg in enumerate(seen_op.input_tensor_ids):
                tensor_id = input_arg.id
                if str(tensor_id) in self.tensor_id_to_observer:
                    observer = self.tensor_id_to_observer[str(tensor_id)]
                    # TODO: return this to the caller
                    scale, zp = observer.calculate_qparams()
                    quantized_input = \
                        torch.quantize_per_tensor(input[input_arg_idx], scale, zp, torch.quint8)
                    new_inputs.append(quantized_input)
                else:
                    new_inputs.append(input[input_args_idx])
            return tuple(new_inputs)

        return input

    def get_inference_func_args_kwargs(
        self,
        func: Callable,
        args: Any,
        kwargs: Any,
        unwrap_scale_zp: bool = False
    ) -> Tuple[Callable, Any, Any]:
        if func in fp32_to_int8_fun_mapping:
            seen_op = self.idx_to_seen_ops[str(self.idx)]
            output_tensor_ids = seen_op.output_tensor_ids
            tensor_id = output_tensor_ids[0].id
            observer = self.tensor_id_to_observer[str(tensor_id)]
            self.idx += 1

            scale, zp = observer.calculate_qparams()
            # TODO(future PR): remove this boolean flag
            if not unwrap_scale_zp:
                kwargs.update({'scale': scale, 'zero_point': zp})
            else:
                kwargs.update({'scale': scale.item(), 'zero_point': zp.item()})
            func = fp32_to_int8_fun_mapping[func]

        return func, args, kwargs

    def get_inference_mod_args_kwargs(
        self,
        mod: torch.nn.Module,
    ):
        for module_type in module_types_supported_by_quantization:
            if isinstance(mod, module_type):
                self.idx += 1
                break

    def get_input_args_quant_info(self) -> Tuple[Optional[Tuple[float, int]]]:
        """
        For each input arg of the current op,
        * if the input arg needs a quant, returns scale + zp
        * else, returns None
        """
        seen_op = self.idx_to_seen_ops[str(self.idx)]
        new_inputs = []
        if seen_op.type in functions_supported_by_quantization or \
                seen_op.type in module_types_supported_by_quantization:
            for input_arg_idx, input_arg in enumerate(seen_op.input_tensor_ids):
                if input_arg.inf_dtype != torch.quint8:
                    tensor_id = input_arg.id
                    observer = self.tensor_id_to_observer[str(tensor_id)]
                    # TODO: return this to the caller
                    scale, zp = observer.calculate_qparams()
                    new_inputs.append((scale, zp,))
                else:
                    new_inputs.append(None)
        return new_inputs

    def reset_to_new_call(self):
        """
        Resets the internal op counter to start a new inference call
        """
        self.idx = 0
