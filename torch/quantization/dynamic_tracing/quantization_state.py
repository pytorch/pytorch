from typing import Callable, List, Tuple, Any, Optional, Dict

import torch

from .mappings import (
    fp32_to_int8_fun_mapping,
    q_mod_to_float_mod_mapping,
)

from .utils import (
    _raise_obs_not_found_error,
    _raise_obs_op_mismatch,
    op_needs_quantization,
    SeenOp,
    QTensorInfo,
    FuncOutputObsType,
    get_func_output_obs_type,
    converted_func_needs_scale_zp,
    FuncOutputDTypeType,
    get_func_output_dtype_type,
)

# TODO(future PR): maybe better name
# TODO(future PR): add serialization support
class AutoQuantizationState(torch.nn.Module):
    """
    Contains state necessary to perform auto quantization on the parent
    `nn.Module` instance.
    """

    idx : int

    def __init__(
        self,
        qconfig,
        input_dtypes: Any = None,
        output_dtypes: Any = None,
    ):
        super().__init__()
        self.idx = 0
        # TODO(future PR): change this to the subset of qconfig_dict
        # relevant to the parent module
        self.qconfig = qconfig
        # this is a ModuleDict in order to properly register observers
        # to be within the module hierarchy.
        self.tensor_id_to_observer = torch.nn.ModuleDict()
        # TODO(future PR): include kwargs
        self.idx_to_seen_ops: Dict[str, SeenOp] = {}
        # qtensor_info objects of tensor outputs of the module, specified
        # in order of iteration through the output type. Non-tensor outputs
        # are represented with `None`.
        self.output_qtensor_infos: List[Optional[QTensorInfo]] = []
        self.input_dtypes = input_dtypes
        self.output_dtypes = output_dtypes

    def has_at_least_one_seen_op(self) -> bool:
        return len(self.idx_to_seen_ops) > 0

    def validate_is_at_last_seen_idx(self) -> None:
        is_at_last_seen_idx = (
            len(self.idx_to_seen_ops) == 0 or
            self.idx == len(self.idx_to_seen_ops)
        )
        assert is_at_last_seen_idx, \
            f"Cur idx: {self.idx}, expected idx: {len(self.idx_to_seen_ops)}"

    def extra_repr(self) -> str:
        s = ""
        if len(self.idx_to_seen_ops):
            s += "(seen_ops): {\n"
            for k, v in self.idx_to_seen_ops.items():
                s += f"  {k}: {v}\n"
            s += "}\n"
        else:
            s += "(seen_ops): {}\n"
        s += "(output_qtensor_infos): ["
        for i in self.output_qtensor_infos:
            s += f"{i} "
        s += "]"
        return s

    def _get_cur_seen_op(self):
        return self.idx_to_seen_ops[str(self.idx)]

    def reset_to_new_call(self):
        """
        Resets the internal op counter to start a new top level module call
        """
        self.idx = 0

    def cur_op_needs_hooks(self, cur_op: Callable) -> bool:
        return op_needs_quantization(cur_op)

    def validate_cur_op(self, cur_op: Callable) -> None:
        """
        This function is expected to be called before any new function or
        module call which needs hooks. It validates that the new function or
        module is of the expected type based on the order of execution.
        """
        assert self.cur_op_needs_hooks(cur_op)
        try:
            seen_op = self._get_cur_seen_op()
            expected_op = seen_op.type
        except IndexError:
            _raise_obs_not_found_error(cur_op)
        if isinstance(cur_op, torch.nn.Module):
            cur_type = type(cur_op)
            is_related = (
                (cur_type == expected_op) or
                (
                    cur_type in q_mod_to_float_mod_mapping and
                    q_mod_to_float_mod_mapping[cur_type] == expected_op
                )
            )
            if not is_related:
                _raise_obs_op_mismatch(cur_op, expected_op)
        else:
            # TODO: move to mapping
            is_related = (
              (cur_op == expected_op) or
              (cur_op is torch.add and expected_op is torch.Tensor.add) or
              (cur_op is torch.Tensor.add and expected_op is torch.add) or
              # TODO: add other flavors of add to complete this
              (cur_op is torch.add and expected_op is torch.Tensor.add_) or
              (cur_op is torch.mul and expected_op is torch.Tensor.mul) or
              (cur_op is torch.Tensor.mul and expected_op is torch.mul)
            )
            if not is_related:
                _raise_obs_op_mismatch(cur_op, expected_op)

    def mark_cur_op_complete(self, cur_op: Callable) -> None:
        """
        This function is expected to be called after a function or module
        processing is complete.
        """
        if op_needs_quantization(cur_op):
            self.idx += 1

    def _outputs_prepare_hook_tensor(
        self,
        output: torch.Tensor,
        first_call: bool,
        qtensor_id: List[int],
    ) -> torch.Tensor:
        if first_call:
            if not hasattr(output, '_qtensor_info'):
                # TODO: use actual dtype instead of defaulting to float
                output._qtensor_info = QTensorInfo(qtensor_id[0], torch.float)  # type: ignore[attr-defined]
                qtensor_id[0] += 1
            self.output_qtensor_infos.append(output._qtensor_info)  # type: ignore[attr-defined]
            # TODO(future PR): add an observer if needed
        return output

    def outputs_prepare_hook(
        self,
        outputs: Any,
        first_call: bool,
        qtensor_id: List[int],
    ) -> Any:
        """
        This function is expected to be called on the outputs of a prepared
        module right before they are returned to the parent.
        """
        if first_call:
            if isinstance(outputs, torch.Tensor):
                self._outputs_prepare_hook_tensor(outputs, first_call, qtensor_id)
            elif isinstance(outputs, tuple):
                new_outputs = []
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_outputs.append(self._outputs_prepare_hook_tensor(
                            output, first_call, qtensor_id))
                    else:
                        new_outputs.append(output)
                return tuple(new_outputs)
            else:
                if False:
                    print('unhandled outputs', outputs)
                    raise AssertionError(
                        f'module outputs with type {type(outputs)} are not handled yet')
                pass

        return outputs

    def outputs_convert_hook(
        self,
        outputs: Any,
    ) -> Any:
        """
        This function is expected to be called on the outputs of a converted
        module right before they are returned to the parent.
        """

        if isinstance(outputs, torch.Tensor):
            qtensor_info = self.output_qtensor_infos[0]
            if self.output_dtypes is not None:
                assert qtensor_info is not None
                # check the output dtype, and do the conversion if needed
                output_dtype = self.output_dtypes[0]
                if qtensor_info.inf_dtype != output_dtype:
                    assert output_dtype is torch.float, \
                        'non-float output dtypes not handled yet'
                    outputs = outputs.dequantize()
            else:
                # if no output dtype was specified, do nothing
                pass

        return outputs

    def get_output_qtensor_infos(self) -> List[Optional[QTensorInfo]]:
        """
        Used by the conversion to torch.jit.script.
        """
        return self.output_qtensor_infos

    def get_output_dtypes(self) -> Any:
        """
        Used by the conversion to torch.jit.script.
        """
        return self.output_dtypes

    def _op_prepare_before_hook_tensor_first_call(
        self,
        arg: Any,
        arg_tensor_infos: List[QTensorInfo],
        qtensor_id: List[int],
    ) -> None:
        # TODO(next): fix this for torch.cat
        if not isinstance(arg, torch.Tensor):
            arg_tensor_infos.append(None)
            return

        # If a tensor does not have an ID, add it. This allows
        # us to track inputs shared by multiple quantizeable modules.
        if not hasattr(arg, '_qtensor_info'):
            arg._qtensor_info = QTensorInfo(qtensor_id[0], arg.dtype)  # type: ignore[attr-defined]
            qtensor_id[0] += 1
        arg_tensor_infos.append(arg._qtensor_info)  # type: ignore[attr-defined]

        # if the existing inf_dtype is not torch.quint8, add an observer
        # which will be converted to a quant later
        # TODO(future PR): share these observers if multiple ops need
        # this quant.
        # TODO(future PR): create from qconfig of op instead of global
        # qconfig.
        if arg._qtensor_info.inf_dtype != torch.quint8:  # type: ignore[attr-defined]
            tensor_id = arg._qtensor_info.id  # type: ignore[attr-defined]
            self.tensor_id_to_observer[str(tensor_id)] = \
                self.qconfig.activation()

    def op_prepare_before_hook(
        self,
        op: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        first_call: bool,
        qtensor_id: List[int],
        fqn: str,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        This function is expected to be called on args and kwargs of
        `op` directly before `op` is executed.

        If `first_call` is True, we record the type of `op`
        and the IDs of its tensor inputs. Note: we add a placeholder for IDs
        of tensor outputs, the placeholder will be filled out during the
        `op_prepare_after_hook`.

        If `first_call` is False, we do the following:
        * pass the inputs through observers, if needed

        The function returns modified `args` and `kwargs`.
        """
        assert self.cur_op_needs_hooks(op)
        if first_call:
            arg_tensor_infos: List[Optional[QTensorInfo]] = []
            for arg in args:
                if isinstance(arg, (list, tuple)):
                    for inner_arg in arg:
                        self._op_prepare_before_hook_tensor_first_call(
                            inner_arg, arg_tensor_infos, qtensor_id)
                else:
                    self._op_prepare_before_hook_tensor_first_call(
                        arg, arg_tensor_infos, qtensor_id)

            key = str(self.idx)
            if key not in self.idx_to_seen_ops:
                if not isinstance(op, torch.nn.Module):
                    self.idx_to_seen_ops[key] = SeenOp(
                        self.idx, op, fqn, arg_tensor_infos, [])
                else:
                    self.idx_to_seen_ops[key] = SeenOp(
                        self.idx, type(op), fqn, arg_tensor_infos, [])

            return args, kwargs

        else:
            seen_op = self._get_cur_seen_op()
            for input_arg_idx, input_arg in enumerate(seen_op.input_tensor_infos):
                if input_arg is not None:
                    tensor_id = input_arg.id
                    # TODO: do not run this twice on input and output
                    if str(tensor_id) in self.tensor_id_to_observer:
                        observer = self.tensor_id_to_observer[str(tensor_id)]
                        # TODO: return this to the caller

                        # special case torch.cat because its first argument
                        # is a list of tensors
                        if seen_op.type == torch.cat:
                            observer(args[0][input_arg_idx])
                        else:
                            observer(args[input_arg_idx])
            return args, kwargs

    def op_prepare_after_hook(
        self,
        op: Callable,
        output: Any,
        args: Tuple[Any, ...],
        first_call: bool,
        qtensor_id: List[int],
        root_module: torch.nn.Module,
    ) -> Any:
        """
        This function is called after an op call on a prepared model.

        If `first_call` is True, we
        * create an observer for the output, if needed, and record it in
          `tensor_id_to_observer`
        * amend the current seen op with the tensor ID of the output

        If `first_call` is False, we
        * observe the output, if needed
        """
        assert self.cur_op_needs_hooks(op)
        func_output_obs_type = get_func_output_obs_type(op, args)
        if first_call:
            if func_output_obs_type == FuncOutputObsType.NEW_OBS:
                self.tensor_id_to_observer[str(qtensor_id[0])] = \
                    self.qconfig.activation()
            elif func_output_obs_type == FuncOutputObsType.REUSES_FIRST_INPUT_OBS:
                seen_op = self._get_cur_seen_op()
                first_input_tensor_id = seen_op.input_tensor_infos[0].id

                first_input_obs = None
                if str(first_input_tensor_id) in self.tensor_id_to_observer:
                    first_input_obs = \
                        self.tensor_id_to_observer[str(first_input_tensor_id)]
                else:
                    # This observer may be in a module (handled by eager
                    # convert), in which case it's not in our map. For now,
                    # copy it from the module. In the future, we could look
                    # into having a soft link.
                    # TODO: make this handle more cases
                    # TODO: handle module -> add_scalar -> add_scalar
                    # TODO: this needs to look up by tensor_id, not idx,
                    # current code may be wrong for some subgraphs
                    prev_op = self.idx_to_seen_ops[str(self.idx - 1)]
                    # TODO: the following line needs to only check fqn
                    # for modules, not for functions
                    fqn_last_part = prev_op.fqn.split('.')[-1]
                    if hasattr(root_module, fqn_last_part):
                        first_input_mod = getattr(root_module, fqn_last_part)
                    else:
                        first_input_mod = None
                    # Currently, both tracing for module fusion and tracing for
                    # quantization go through this code path. When tracing
                    # for module fusion, quantizeable modules do not have
                    # observers yet. For this path to not crash, we create one.
                    # When tracing for quantization, this will be ignored.
                    # TODO(future PR): refactor to avoid this.
                    if first_input_mod and hasattr(first_input_mod, 'activation_post_process'):
                        first_input_obs = getattr(
                            first_input_mod, 'activation_post_process')
                    else:
                        first_input_obs = self.qconfig.activation()

                self.tensor_id_to_observer[str(qtensor_id[0])] = first_input_obs

            # TODO(future PR): check if _qtensor_id needs to become an actual
            # attribute of Tensor
            # TODO(future PR): handle non-tensor outputs

            func_output_dtype_type = get_func_output_dtype_type(op)
            if func_output_dtype_type == FuncOutputDTypeType.DTYPE_DEPENDS_ON_QCONFIG:
                dtype_to_use = torch.quint8
            else:
                first_arg_dtype = args[0]._qtensor_info.inf_dtype
                dtype_to_use = first_arg_dtype
            output._qtensor_info = QTensorInfo(qtensor_id[0], dtype_to_use)
            self.idx_to_seen_ops[str(self.idx)].output_tensor_infos.append(
                output._qtensor_info)
            qtensor_id[0] += 1
        else:
            # TODO(future PR): other output types
            if func_output_obs_type != FuncOutputObsType.NONE:
                seen_op = self._get_cur_seen_op()
                tensor_id = seen_op.output_tensor_infos[0].id
                obs = self.tensor_id_to_observer[str(tensor_id)]
                output = obs(output)
        return output

    def op_convert_before_hook(
        self,
        op: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]:
        """
        This function is called before an op call in a converted model.

        For each arg in `args`, quantizes it if necessary.

        Returns potentially modified `op`, potentially modified `args`,
        potentially modified `kwargs`.
        """
        assert self.cur_op_needs_hooks(op)

        op, arg_quant_infos, additional_kwargs = \
            self.get_op_convert_info(op)

        # potentially quantize args, based on arg_quant_infos
        new_args = []
        tensor_arg_idx = 0
        if op == torch.ops.quantized.cat:
            # input tensors
            new_first_arg = []
            for arg in args[0]:
                # TODO: handle non-tensor inputs
                quant_info = arg_quant_infos[tensor_arg_idx]
                if quant_info is not None:
                    scale, zp = quant_info
                    arg = torch.quantize_per_tensor(arg, scale, zp, torch.quint8)
                new_first_arg.append(arg)
                tensor_arg_idx += 1
            new_args = [new_first_arg, *args[1:]]
        else:
            for arg in args:
                if not isinstance(arg, torch.Tensor):
                    new_args.append(arg)
                    continue
                # TODO: handle non-tensor inputs
                quant_info = arg_quant_infos[tensor_arg_idx]
                if quant_info is not None:
                    scale, zp = quant_info
                    arg = torch.quantize_per_tensor(arg, scale, zp, torch.quint8)
                new_args.append(arg)
                tensor_arg_idx += 1

        # potentially extend kwargs with scale and zero_point
        kwargs.update(**additional_kwargs)

        return op, tuple(new_args), kwargs

    def op_convert_after_hook(
        self,
        op: Callable,
        output,
    ) -> Any:
        """
        This function is called aftern an op call in a converted model.

        TODO: add dequant, if needed
        """
        return output

    def get_op_convert_info(
        self,
        op: Callable,
        unwrap_scale_zp: bool = False,
    ) -> Tuple[Callable, Any, Any]:
        """
        Returns the information needed for convert time modifications to `op`.
        Has no side effects.

        For `op`, returns either the original callable unchanged,
        or the corresponding quantized target. Note: always returns the original
        callable for modules, because they are quantized with module swaps.

        For `args`, returns information needed to quantize each arg, if
        applicable.

        For `kwargs`, returns additional kwargs for scale and zero_point, if
        applicable.
        """
        assert self.cur_op_needs_hooks(op)

        # calculate new op
        new_op = op
        if not isinstance(op, torch.nn.Module):
            if op in fp32_to_int8_fun_mapping:
                new_op = fp32_to_int8_fun_mapping[op]

        # calculate quant infos
        arg_quant_infos = self._get_input_args_quant_info(op)

        # calculate scale and zp for output
        # TODO: instead of always doing this if there is an observer,
        # calculate whether this is needed based on the op and dtypes
        additional_kwargs = {}
        seen_op = self._get_cur_seen_op()
        needs_scale_zp = converted_func_needs_scale_zp(op, seen_op)
        if needs_scale_zp:
            seen_op = self._get_cur_seen_op()
            output_tensor_infos = seen_op.output_tensor_infos
            tensor_id = output_tensor_infos[0].id
            observer = self.tensor_id_to_observer[str(tensor_id)]

            scale, zp = observer.calculate_qparams()
            # TODO(future PR): remove this boolean flag
            if not unwrap_scale_zp:
                additional_kwargs.update({'scale': scale, 'zero_point': zp})
            else:
                additional_kwargs.update(
                    {'scale': scale.item(), 'zero_point': zp.item()})

        return new_op, arg_quant_infos, additional_kwargs

    def _get_input_args_quant_info(
        self,
        cur_op: Callable,
    ) -> List[Optional[Tuple[float, int]]]:
        """
        Returns a list of information about the tensor inputs to the current op.
        For each tensor input:
        * if the tensor input needs a quant, the list will contain
          (scale, zero_point)
        * if the tensor input does not need a quant, the list will contain None

        For example, if there are two tensor inputs to the current op, and the
        first input needs a quant, this function will return

          [(scale0, zero_point0), None]
        """
        assert self.cur_op_needs_hooks(cur_op)
        seen_op = self._get_cur_seen_op()
        quant_infos: List[Optional[Tuple[float, int]]] = []

        # determine the expected output dtype
        output_dtype = seen_op.output_tensor_infos[0].inf_dtype

        for input_arg_idx, input_arg in enumerate(seen_op.input_tensor_infos):
            if input_arg is not None:
                tensor_id = input_arg.id
                if str(tensor_id) in self.tensor_id_to_observer and \
                        input_arg.inf_dtype != output_dtype:
                    observer = self.tensor_id_to_observer[str(tensor_id)]
                    # TODO: return this to the caller
                    scale, zp = observer.calculate_qparams()
                    quant_infos.append((scale, zp,))
                else:
                    quant_infos.append(None)
            else:
                quant_infos.append(None)
        return quant_infos

    # This is a hack to enable nn.Sequential to properly work with
    # this class.
    # TODO(future): remove the hack
    def forward(self, x):
        return x
