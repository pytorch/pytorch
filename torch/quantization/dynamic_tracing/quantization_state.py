import torch
from typing import Callable, List, Tuple, Any, Optional

from .mappings import (
    fp32_to_int8_fun_mapping,
    functions_supported_by_quantization,
    module_types_supported_by_quantization,
    q_mod_to_float_mod_mapping,
)

from .utils import (
    _raise_obs_not_found_error,
    _raise_obs_op_mismatch,
    func_or_mod_needs_quantization,
    SeenOp,
    QTensorInfo,
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

    def has_at_least_one_seen_op(self) -> bool:
        return len(self.idx_to_seen_ops) > 0

    def extra_repr(self) -> str:
        s = ""
        s += "(seen_ops): {\n"
        for k, v in self.idx_to_seen_ops.items():
            s += f"  {k}: {v}\n"
        s += "},\n"
        s += "(idx): " + str(self.idx) + ","
        return s

    def _get_cur_seen_op(self):
        return self.idx_to_seen_ops[str(self.idx)]

    def validate_and_increment(self, cur_func_or_mod: Callable) -> None:
        if not func_or_mod_needs_quantization(cur_func_or_mod):
            return
        try:
            seen_op = self._get_cur_seen_op()
            expected_func_or_mod = seen_op.type
        except IndexError:
            _raise_obs_not_found_error(cur_func_or_mod)
        if isinstance(cur_func_or_mod, torch.nn.Module):
            cur_type = type(cur_func_or_mod)
            is_related = (
                (cur_type == expected_func_or_mod) or
                (
                    cur_type in q_mod_to_float_mod_mapping and
                    q_mod_to_float_mod_mapping[cur_type] == expected_func_or_mod
                )
            )
            if not is_related:
                _raise_obs_op_mismatch(cur_func_or_mod, expected_func_or_mod)
        else:
            if cur_func_or_mod != expected_func_or_mod:
                _raise_obs_op_mismatch(cur_func_or_mod, expected_func_or_mod)
        self.idx += 1

    def func_or_mod_before_hook(
        self,
        func_or_mod: Callable,
        args, kwargs,
        first_call: bool,
        qtensor_id: List[int],
    ) -> None:
        if not func_or_mod_needs_quantization(func_or_mod):
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
                # this quant.
                # TODO(future PR): create from qconfig of op instead of global
                # qconfig.
                if arg._qtensor_info.inf_dtype != torch.quint8:
                    tensor_id = arg._qtensor_info.id
                    self.tensor_id_to_observer[str(tensor_id)] = \
                        self.qconfig.activation()

            key = str(self.idx)
            if key not in self.idx_to_seen_ops:
                if not isinstance(func_or_mod, torch.nn.Module):
                    self.idx_to_seen_ops[key] = SeenOp(
                        self.idx, func_or_mod, arg_tensor_ids, [])
                else:
                    self.idx_to_seen_ops[key] = SeenOp(
                        self.idx, type(func_or_mod), arg_tensor_ids, [])

        else:
            seen_op = self._get_cur_seen_op()
            for input_arg_idx, input_arg in enumerate(seen_op.input_tensor_ids):
                tensor_id = input_arg.id
                # TODO: do not run this twice on input and output
                if str(tensor_id) in self.tensor_id_to_observer:
                    observer = self.tensor_id_to_observer[str(tensor_id)]
                    # TODO: return this to the caller
                    observer(args[input_arg_idx])

    def func_or_mod_after_hook(
        self,
        func_or_mod: Callable,
        output: Any,
        first_call: bool,
        qtensor_id: List[int],
    ) -> Any:
        """
        This function is called after a function call
        """
        if not func_or_mod_needs_quantization(func_or_mod):
            return

        if first_call:
            self.tensor_id_to_observer[str(qtensor_id[0])] = \
                self.qconfig.activation()

            # TODO(future PR): check if _qtensor_id needs to become an actual
            # attribute of Tensor
            output._qtensor_info = QTensorInfo(qtensor_id[0], torch.quint8)
            self.idx_to_seen_ops[str(self.idx)].output_tensor_ids.append(
                output._qtensor_info)
            qtensor_id[0] += 1
        else:
            # TODO(future PR): other output types
            seen_op = self._get_cur_seen_op()
            tensor_id = seen_op.output_tensor_ids[0].id
            obs = self.tensor_id_to_observer[str(tensor_id)]
            output = obs(output)
        return output

    def inference_func_or_mod_before_hook(
        self,
        func_or_mod: Callable,
        args,
        kwargs,
    ) -> Tuple[Any]:
        if not func_or_mod_needs_quantization(func_or_mod):
            return

        seen_op = self._get_cur_seen_op()

        new_inputs = []
        for input_arg_idx, input_arg in enumerate(seen_op.input_tensor_ids):
            tensor_id = input_arg.id
            # only quantize if the dtype is float and we are going to a quantized
            # op.
            if str(tensor_id) in self.tensor_id_to_observer and \
                    input_arg.inf_dtype == torch.float:
                observer = self.tensor_id_to_observer[str(tensor_id)]
                # TODO: return this to the caller
                scale, zp = observer.calculate_qparams()
                quantized_input = \
                    torch.quantize_per_tensor(args[input_arg_idx], scale, zp, torch.quint8)
                new_inputs.append(quantized_input)
            else:
                new_inputs.append(args[input_arg_idx])
        return tuple(new_inputs)

    def inference_func_or_mod_after_hook(
        self,
        func: Callable,
        output,
    ) -> Any:
        return output

    def get_inference_func_or_mod_args_kwargs(
        self,
        func_or_mod: Callable,
        args: Any,
        kwargs: Any,
        unwrap_scale_zp: bool = False
    ) -> Tuple[Callable, Any, Any]:
        if func_or_mod_needs_quantization(func_or_mod):
            if not isinstance(func_or_mod, torch.nn.Module):
                if func_or_mod in fp32_to_int8_fun_mapping:
                    seen_op = self._get_cur_seen_op()
                    output_tensor_ids = seen_op.output_tensor_ids
                    tensor_id = output_tensor_ids[0].id
                    observer = self.tensor_id_to_observer[str(tensor_id)]

                    scale, zp = observer.calculate_qparams()
                    # TODO(future PR): remove this boolean flag
                    if not unwrap_scale_zp:
                        kwargs.update({'scale': scale, 'zero_point': zp})
                    else:
                        kwargs.update({'scale': scale.item(), 'zero_point': zp.item()})
                    func_or_mod = fp32_to_int8_fun_mapping[func_or_mod]

        return func_or_mod, args, kwargs

    def get_input_args_quant_info(self) -> Tuple[Optional[Tuple[float, int]]]:
        """
        For each input arg of the current op,
        * if the input arg needs a quant, returns scale + zp
        * else, returns None
        """
        seen_op = self._get_cur_seen_op()
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
