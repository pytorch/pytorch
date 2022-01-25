from typing import Callable, List, Tuple, Any, Optional, Dict, Set

import torch
import torch.nn.functional as F

from .mappings import (
    ops_are_related,
)

from .utils import (
    _raise_obs_not_found_error,
    _raise_obs_op_mismatch,
    op_needs_quantization,
    SeenOpInfo,
    QTensorInfo,
    FuncOutputObsType,
    get_func_output_obs_type,
    converted_func_needs_scale_zp,
    FuncOutputDTypeType,
    get_func_output_dtype_type,
    get_quantized_op,
    get_input_observed_arg_idxs,
    get_packable_tensor_arg_idxs,
    get_param_name,
    get_packable_nontensor_arg_idxs,
    get_packable_arg_idxs,
    get_weight_arg_idx,
    iterate_and_apply,
    get_op_packing_only_uses_module_attributes,
    get_packable_tensor_kwarg_names,
    get_producer_of_seen_op_info,
    clone_detach_tensor_without_dispatch,
    get_input_args_quant_dequant_info,
    get_cur_qconfig,
)

OpConvertInfo = Tuple[
    # quantized equivalent of original op (None means keep original)
    Optional[Callable],
    # arg_quant_infos, each element is (scale, zp) for quantized and None otherwise
    List[Optional[Tuple[float, int]]],
    # arg_dequant_infos, each element is True if this arg needs a dequant
    List[bool],
    # packed param name, if the op has a packed param
    Optional[str],
    # additional kwargs, such as output scale and zero_point
    Dict[str, Any],
    # any_arg_quant_or_dequant_needed, if False then we can skip looking at
    # arg_quant_infos and arg_dequant_infos, for performance
    bool,
    # any_arg_kwarg_modification_needed, if False then we can return original
    # args and kwargs, for performance
    bool,
]

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
        qconfig_dict: Dict[str, Any],
        fqn: str,
        input_dtypes: Any = None,
        output_dtypes: Any = None,
    ):
        super().__init__()
        self.idx = 0
        self.qconfig_dict = qconfig_dict
        self.fqn = fqn
        # this is a ModuleDict in order to properly register observers
        # to be within the module hierarchy.
        self.tensor_id_to_observer = torch.nn.ModuleDict()
        # TODO(future PR): include kwargs
        self.idx_to_seen_op_infos: Dict[int, SeenOpInfo] = {}
        # qtensor_info objects of tensor outputs of the module, specified
        # in order of iteration through the output type. Non-tensor outputs
        # are represented with `None`.
        self.output_qtensor_infos: List[Optional[QTensorInfo]] = []
        self.input_dtypes = input_dtypes
        self.output_dtypes = output_dtypes
        # key: idx of seen op
        # value: name of packed weight
        # note: this is filled out right before convert
        self.idx_to_packed_weight_name: Dict[int, str] = {}
        self.tensor_id_to_scale_zp: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Numeric Suite add_loggers functionality
        # if this flag is True, op outputs will be saved for debugging
        self.log_op_outputs = False
        # data structure to save op outputs for debugging
        # * outer list represents the different model forward call instances
        # * inner list represents the different op forward call instances in a
        #   model forward
        # TODO(future PR): handle types which are not torch.Tensor
        # TODO(future PR): use the Logger class and allow user overrides of it
        self.op_outputs: List[List[Tuple[
            int,  # global op idx
            Optional[str],  # fqn
            Callable,  # fp32 op type (TODO future PR: add quantized op type)
            torch.Tensor,  # value
        ]]] = []
        # model name to use in logging results
        self.logging_model_name: Optional[str]

        self.idx_to_op_convert_info: Dict[int, OpConvertInfo] = {}

        # If this is True, module outputs will be checked and converted
        # to the dtype specified by the user. If this is False, module outputs
        # will be returned as is. This value can be precalculated and it is set
        # to its final value after tracing.
        self.needs_dtype_transform_on_outputs = True

        # For debugging only, stores the types of ops seen by the parent which
        # did not require op hooks.
        self.seen_op_types_without_op_hooks: Set[Callable] = set()

    def get_extra_state(self):
        return {"tensor_id_to_scale_zp": self.tensor_id_to_scale_zp}

    def set_extra_state(self, state):
        self.tensor_id_to_scale_zp = state["tensor_id_to_scale_zp"]
        for _, seen_op_info in self.idx_to_seen_op_infos.items():
            self.idx_to_op_convert_info[seen_op_info.idx] = \
                self.calculate_op_convert_info(seen_op_info)

    def has_at_least_one_seen_op_info(self) -> bool:
        return len(self.idx_to_seen_op_infos) > 0

    def validate_is_at_last_seen_idx(self) -> None:
        is_at_last_seen_idx = (
            len(self.idx_to_seen_op_infos) == 0 or
            self.idx == len(self.idx_to_seen_op_infos)
        )
        if not is_at_last_seen_idx:
            raise AssertionError(
                f"Cur idx: {self.idx}, expected idx: {len(self.idx_to_seen_op_infos)}")

    def extra_repr(self) -> str:
        s = ""
        # idx_to_seen_op_infos
        if len(self.idx_to_seen_op_infos):
            s += "(seen_op_infos): {\n"
            for k, v in self.idx_to_seen_op_infos.items():
                s += f"  {k}: {v}\n"
            s += "}\n"
        else:
            s += "(seen_op_infos): {}\n"
        # output_qtensor_infos
        s += "(output_qtensor_infos): ["
        for i in self.output_qtensor_infos:
            s += f"{i} "
        s += "]\n"
        # seen_op_types_without_op_hooks
        s += f"(seen_op_types_without_op_hooks): {self.seen_op_types_without_op_hooks}\n"
        # idx_to_packed_weight_name
        if len(self.idx_to_packed_weight_name):
            s += "(idx_to_packed_weight_name): {\n"
            for k, v in self.idx_to_packed_weight_name.items():  # type: ignore[assignment]
                s += f"  {k}: {v}\n"
            s += "}\n"
        else:
            s += "(idx_to_packed_weight_name): {}"
        if len(self.tensor_id_to_scale_zp):
            s += "(tensor_id_to_scale_zp): {\n"
            for k, v in self.tensor_id_to_scale_zp.items():  # type: ignore[assignment]
                s += f"  {k}: {v}\n"
            s += "}"
        return s

    def _get_cur_seen_op_info(self):
        return self.idx_to_seen_op_infos[self.idx]

    def get_cur_output_inf_dtype(self):
        return self._get_cur_seen_op_info().output_tensor_infos[0].inf_dtype

    def reset_to_new_call(self):
        """
        Resets the internal op counter to start a new top level module call
        """
        # torch.nn.Module __setattr__ has overhead,
        # this code is the explicit fast path for `self.idx = 0`
        object.__setattr__(self, 'idx', 0)

        if self.log_op_outputs:
            self.op_outputs.append([])

    def cur_op_needs_hooks(self, cur_op: Callable) -> bool:
        return op_needs_quantization(cur_op)

    def validate_cur_op(self, cur_op: Callable) -> None:
        """
        This function is expected to be called before any new function or
        module call which needs hooks. It validates that the new function or
        module is of the expected type based on the order of execution.
        """
        try:
            seen_op_info = self._get_cur_seen_op_info()
            expected_op = seen_op_info.type
        except IndexError:
            _raise_obs_not_found_error(cur_op)
        if not ops_are_related(cur_op, expected_op, seen_op_info.type_is_module):
            _raise_obs_op_mismatch(cur_op, expected_op)

    def mark_cur_op_complete(self, cur_op: Callable) -> None:
        """
        This function is expected to be called after a function or module
        processing is complete.
        """
        # torch.nn.Module __setattr__ has overhead,
        # this code is the explicit fast path for `self.idx += 1`
        object.__setattr__(self, 'idx', self.idx + 1)

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
            outputs = self._first_call_assign_qtensor_infos_to_mod_outputs(
                outputs, qtensor_id)
        return outputs

    def outputs_convert_hook(
        self,
        outputs: Any,
    ) -> Any:
        """
        This function is expected to be called on the outputs of a converted
        module right before they are returned to the parent.
        """
        outputs = self._maybe_mod_outputs_dtype_transform(outputs)
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

    def op_prepare_before_hook(
        self,
        op: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        first_call: bool,
        qtensor_id: List[int],
        fqn: str,
        root_module: torch.nn.Module,
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
        if first_call:
            return self._first_call_op_prepare_before_hook_create_subgraphs(
                op, args, kwargs, first_call, qtensor_id, fqn, root_module)
        else:
            seen_op_info = self._get_cur_seen_op_info()

            def _maybe_observe(arg, tensor_info):
                tensor_id = tensor_info.id
                # TODO: do not run this twice on input and output
                if str(tensor_id) in self.tensor_id_to_observer:
                    observer = self.tensor_id_to_observer[str(tensor_id)]
                    return observer(arg)
                else:
                    return arg

            args = iterate_and_apply(
                args, seen_op_info.input_tensor_infos, _maybe_observe)

            return args, kwargs

    def op_prepare_after_hook(
        self,
        op: Callable,
        output: Any,
        args: Tuple[Any, ...],
        first_call: bool,
        qtensor_id: List[int],
        global_op_idx: List[int],
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
        seen_op_info = self._get_cur_seen_op_info()
        if first_call:
            self._first_call_op_prepare_after_hook_adjust_subgraphs(
                op, output, args, first_call, qtensor_id,
                seen_op_info)
        else:
            func_output_obs_type = get_func_output_obs_type(seen_op_info)
            # TODO(future PR): other output types
            if func_output_obs_type != FuncOutputObsType.NONE:
                seen_op_info = self._get_cur_seen_op_info()
                tensor_id = seen_op_info.output_tensor_infos[0].id
                obs = self.tensor_id_to_observer[str(tensor_id)]
                output = obs(output)

        if self.log_op_outputs:
            output_clone = clone_detach_tensor_without_dispatch(output)
            self.op_outputs[-1].append(
                (global_op_idx[0], seen_op_info.fqn, seen_op_info.type, output_clone))
            global_op_idx[0] += 1

        return output

    def op_convert_before_hook(
        self,
        op: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        root_module: torch.nn.Module,
    ) -> Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]:
        """
        This function is called before an op call in a converted model.

        For each arg in `args`, quantizes it if necessary.

        Returns potentially modified `op`, potentially modified `args`,
        potentially modified `kwargs`.
        """
        # TODO generalize this for more things
        # currently:
        # * can quantize args (via arg_quant_infos)
        # * can add scale and zp (via additional kwargs)

        # needed for F.conv2d
        # F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        #   to
        # q.conv2d(input, packed_params, scale, zero_point)
        orig_op = op
        maybe_new_op, arg_quant_infos, arg_dequant_infos, packed_param_name, \
            additional_kwargs, any_arg_quant_or_dequant_needed, \
            any_arg_kwarg_modification_needed = self.get_op_convert_info(op)
        if maybe_new_op is not None:
            op = maybe_new_op
        if not any_arg_kwarg_modification_needed:
            return op, args, kwargs
        # print(op, arg_quant_infos, packed_param_name, additional_kwargs)

        # potentially quantize args, based on arg_quant_infos
        new_args = []
        if any_arg_quant_or_dequant_needed:
            tensor_arg_idx = 0
            # TODO: refactor this to use iterate_and_apply
            if orig_op is torch.cat:  # torch.cat variants
                # input tensors
                new_first_arg = []
                for arg in args[0]:
                    # TODO: handle non-tensor inputs
                    quant_info = arg_quant_infos[tensor_arg_idx]
                    dequant_info = arg_dequant_infos[tensor_arg_idx]
                    if quant_info is not None:
                        scale, zp = quant_info
                        arg = torch.quantize_per_tensor(arg, scale, zp, torch.quint8)
                    elif dequant_info is True:
                        arg = arg.dequantize()
                    new_first_arg.append(arg)
                    tensor_arg_idx += 1
                new_args = [new_first_arg, *args[1:]]
            else:
                for arg in args:
                    # TODO: handle non-tensor inputs
                    # TODO: this is not handling non-tensor tuple args (for example,
                    # dilation in conv2d) correctly, it just happens to work but
                    # needs a fix.
                    quant_info = arg_quant_infos[tensor_arg_idx]
                    dequant_info = arg_dequant_infos[tensor_arg_idx]
                    if quant_info is not None:
                        scale, zp = quant_info
                        arg = torch.quantize_per_tensor(arg, scale, zp, torch.quint8)
                    elif dequant_info is True:
                        arg = arg.dequantize()
                    new_args.append(arg)
                    tensor_arg_idx += 1
        else:
            new_args = [*args]

        # if there is a packed param, replace the relevant args
        if packed_param_name is not None:
            new_args_with_packed = []
            packable_arg_idxs = get_packable_arg_idxs(orig_op)
            added_packed = False
            for idx, arg in enumerate(new_args):
                if packable_arg_idxs is not None and idx in packable_arg_idxs:
                    if not added_packed:
                        packed_param = getattr(root_module, packed_param_name)
                        new_args_with_packed.append(packed_param)
                        added_packed = True
                else:
                    new_args_with_packed.append(arg)
            new_args = new_args_with_packed

        # potentially extend kwargs with scale and zero_point
        # TODO move op-specific logic out of here
        if len(additional_kwargs):
            if orig_op not in (F.conv2d, F.linear):
                kwargs.update(**additional_kwargs)
            else:
                seen_op_info = self._get_cur_seen_op_info()
                if seen_op_info.output_tensor_infos[0].inf_dtype == torch.quint8:
                    new_args.append(additional_kwargs['scale'])
                    new_args.append(additional_kwargs['zero_point'])

        # TODO move op-specific logic out of here
        if op is torch.ops.quantized.linear:
            kwargs.pop('bias', None)

        return op, tuple(new_args), kwargs

    def op_convert_after_hook(
        self,
        op: Callable,
        output,
        global_op_idx: List[int],
    ) -> Any:
        """
        This function is called aftern an op call in a converted model.

        TODO: add dequant, if needed
        """
        if self.log_op_outputs:
            output_clone = clone_detach_tensor_without_dispatch(output)
            seen_op_info = self._get_cur_seen_op_info()
            self.op_outputs[-1].append(
                (global_op_idx[0], seen_op_info.fqn, seen_op_info.type, output_clone))
            global_op_idx[0] += 1

        return output

    def get_op_convert_info(
        self,
        op: Callable,
    ) -> OpConvertInfo:
        """
        Returns the information needed for convert time modifications to `op`.
        """
        return self.idx_to_op_convert_info[self.idx]

    def calculate_op_convert_info(
        self,
        seen_op_info: SeenOpInfo,
    ) -> OpConvertInfo:
        """
        This precalculates the information which will be returned by
        `get_op_convert_info`.
        """
        # calculate new op
        maybe_new_op = get_quantized_op(seen_op_info)

        # calculate quant infos
        arg_quant_infos, arg_dequant_infos, any_arg_quant_or_dequant_needed = \
            get_input_args_quant_dequant_info(
                seen_op_info, self.tensor_id_to_scale_zp)

        # get packed param name, if applicable
        packed_param_name = self._get_packed_param_name(seen_op_info)

        # calculate scale and zp for output
        # TODO: instead of always doing this if there is an observer,
        # calculate whether this is needed based on the op and dtypes
        additional_kwargs = {}
        needs_scale_zp = converted_func_needs_scale_zp(seen_op_info)
        if needs_scale_zp:
            output_tensor_infos = seen_op_info.output_tensor_infos
            tensor_id = output_tensor_infos[0].id
            scale, zp = self.tensor_id_to_scale_zp[tensor_id]
            additional_kwargs.update({'scale': scale, 'zero_point': zp})

        any_arg_kwarg_modification_needed = bool(
            any_arg_quant_or_dequant_needed or
            packed_param_name is not None or
            len(additional_kwargs)
        )  # the cast to bool is to make mypy recognize this as a bool

        return maybe_new_op, arg_quant_infos, arg_dequant_infos, \
            packed_param_name, additional_kwargs, any_arg_quant_or_dequant_needed, \
            any_arg_kwarg_modification_needed

    def _get_packed_param_name(self, seen_op_info: SeenOpInfo) -> Optional[str]:
        """
        If the op in seen_op_info has a quantized packed param, returns it.
        Otherwise, returns None.
        """
        return self.idx_to_packed_weight_name.get(seen_op_info.idx, None)

    def _first_call_assign_qtensor_infos_to_mod_outputs_tensor(
        self,
        output: torch.Tensor,
        qtensor_id: List[int],
    ) -> torch.Tensor:
        """
        This is a helper function for _first_call_assign_qtensor_infos_to_mod_outputs
        to handle iterables of tensors without code duplication.
        """
        if not hasattr(output, '_qtensor_info'):
            # TODO: use actual dtype instead of defaulting to float
            output._qtensor_info = QTensorInfo(  # type: ignore[attr-defined]
                qtensor_id[0], output.dtype, torch.float)
            qtensor_id[0] += 1
        self.output_qtensor_infos.append(output._qtensor_info)  # type: ignore[attr-defined]
        # TODO(future PR): add an observer if needed
        return output

    def _first_call_assign_qtensor_infos_to_mod_outputs(
        self,
        outputs: Any,
        qtensor_id: List[int],
    ) -> Any:
        """
        Takes `outputs`, which are a set of values about to be returned from
        the current module. If `_qtensor_info` attributes do not already exist
        on any tensors in `outputs`, this function adds them, initializing the
        dtype to `torch.float`. This allows us to reason about module output
        dtypes even if the last op in the module is not quantizeable.
        """
        # TODO: handle objects with deeper nested tensors
        if isinstance(outputs, torch.Tensor):
            self._first_call_assign_qtensor_infos_to_mod_outputs_tensor(outputs, qtensor_id)
        elif isinstance(outputs, tuple):
            # TODO: handle other tuple subclasses more generically
            new_outputs = []
            for output in outputs:
                if isinstance(output, torch.Tensor):
                    new_outputs.append(self._first_call_assign_qtensor_infos_to_mod_outputs_tensor(
                        output, qtensor_id))
                else:
                    new_outputs.append(output)
            # hacky check for collections.namedtuple, TODO improve this
            # https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
            if hasattr(outputs, '_fields'):
                outputs = outputs.__class__(*new_outputs)
            else:
                outputs = tuple(new_outputs)
        else:
            pass
        return outputs

    def set_needs_dtype_transform_on_outputs(self):
        """
        Calculates whether a dtype transform on module outputs is needed
        and stores it. This is used to skip the outputs hook if it is not
        needed.
        """
        self.needs_dtype_transform_on_outputs = False

        if not len(self.output_qtensor_infos):
            # if there are no tensor outputs, there is nothing to transform
            return

        qtensor_info = self.output_qtensor_infos[0]
        if self.output_dtypes is not None:
            assert qtensor_info is not None
            # check the output dtype, and do the conversion if needed
            output_dtype = self.output_dtypes[0]
            if qtensor_info.inf_dtype != output_dtype:
                assert output_dtype is torch.float, \
                    'non-float output dtypes not handled yet'
                self.needs_dtype_transform_on_outputs = True

    def _maybe_mod_outputs_dtype_transform(
        self,
        outputs: Any,
    ) -> Any:
        """
        Takes `outputs` which are about to be returned from this module
        to the caller. If this module has restrictions on the dtypes of
        tensors it has to return, does the dtype conversion. Otherwise,
        does nothing.
        """
        if not self.needs_dtype_transform_on_outputs:
            return outputs

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

    def _first_call_op_prepare_before_hook_create_subgraphs_tensor(
        self,
        op: Callable,
        arg: Any,
        arg_tensor_infos: List[Optional[QTensorInfo]],
        qtensor_id: List[int],
    ) -> None:
        """
        Runs the prepare hook during first_call for individual
        tensors. If the input argument is a tensor, this function is
        called directly. If the input argument is an iterable such
        as a list or a tuple, this function is called on each element of
        the iteratble.
        """
        # TODO(next): fix this for torch.cat
        if not isinstance(arg, torch.Tensor):
            arg_tensor_infos.append(None)
            return

        # If a tensor does not have an ID, add it. This allows
        # us to track inputs shared by multiple quantizeable modules.
        if not hasattr(arg, '_qtensor_info'):
            arg._qtensor_info = QTensorInfo(  # type: ignore[attr-defined]
                qtensor_id[0], arg.dtype, arg.dtype)
            qtensor_id[0] += 1
        arg_tensor_infos.append(arg._qtensor_info)  # type: ignore[attr-defined]

    def _first_call_op_prepare_before_hook_create_subgraphs(
        self,
        op: Callable,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        first_call: bool,
        qtensor_id: List[int],
        fqn: str,
        root_module: torch.nn.Module,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Given an op, args, kwargs about to be executed, records the subgraph
        of this op in `self`.
        """
        op_packing_only_uses_module_attributes = \
            get_op_packing_only_uses_module_attributes(op, args, kwargs, root_module)
        arg_tensor_infos: List[Optional[QTensorInfo]] = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                for inner_arg in arg:
                    self._first_call_op_prepare_before_hook_create_subgraphs_tensor(
                        op, inner_arg, arg_tensor_infos, qtensor_id)
            else:
                self._first_call_op_prepare_before_hook_create_subgraphs_tensor(
                    op, arg, arg_tensor_infos, qtensor_id)

        packable_tensor_idx_to_name = {}
        packable_nontensor_idx_to_arg = {}
        packable_tensor_kwarg_name_to_name = {}
        if op_packing_only_uses_module_attributes:
            packable_tensor_arg_idxs = get_packable_tensor_arg_idxs(op)
            if packable_tensor_arg_idxs is not None:
                for arg_idx in packable_tensor_arg_idxs:
                    if arg_idx >= len(args):
                        continue
                    arg = args[arg_idx]
                    param_name = get_param_name(root_module, arg)
                    packable_tensor_idx_to_name[arg_idx] = param_name

            packable_nontensor_arg_idxs = get_packable_nontensor_arg_idxs(op)
            if packable_nontensor_arg_idxs is not None:
                for arg_idx in packable_nontensor_arg_idxs:
                    packable_nontensor_idx_to_arg[arg_idx] = args[arg_idx]

            packable_tensor_kwarg_names = \
                get_packable_tensor_kwarg_names(op)
            if packable_tensor_kwarg_names is not None:
                for kwarg_name in packable_tensor_kwarg_names:
                    if kwarg_name not in kwargs:
                        continue
                    kwarg = kwargs[kwarg_name]
                    kwarg_name_on_module = get_param_name(root_module, kwarg)
                    packable_tensor_kwarg_name_to_name[kwarg_name] = \
                        kwarg_name_on_module

        if self.idx not in self.idx_to_seen_op_infos:
            op_type_is_module = isinstance(op, torch.nn.Module)
            op_type : Callable = type(op) if op_type_is_module else op  # type: ignore[assignment]
            qconfig = get_cur_qconfig(self.qconfig_dict, fqn, op_type)
            self.idx_to_seen_op_infos[self.idx] = SeenOpInfo(
                self.idx, op_type, op_type_is_module, fqn, arg_tensor_infos, [],
                packable_tensor_idx_to_name, packable_nontensor_idx_to_arg,
                packable_tensor_kwarg_name_to_name,
                op_packing_only_uses_module_attributes, qconfig)

        return args, kwargs

    def _first_call_op_prepare_after_hook_adjust_subgraphs(
        self,
        op: Callable,
        output: Any,
        args: Tuple[Any, ...],
        first_call: bool,
        qtensor_id: List[int],
        seen_op_info: SeenOpInfo,
    ) -> None:
        """
        After `op` was just executed, modifies the subgraph recorded
        for this op with the information about the output. Note, this
        has to be done in the "after" hook because the output of the op
        does not exist in the "before" hook.
        """
        # TODO(future PR): check if _qtensor_id needs to become an actual
        # attribute of Tensor
        # TODO(future PR): handle non-tensor outputs

        func_output_dtype_type = get_func_output_dtype_type(seen_op_info)
        if func_output_dtype_type == FuncOutputDTypeType.DTYPE_DEPENDS_ON_QCONFIG:
            if isinstance(op, torch.nn.Module):
                # For now, assume that eager mode convert has attached qconfig
                # objects to any leaf module which needs quantization
                if hasattr(op, 'activation_post_process'):
                    dtype_to_use = op.activation_post_process.dtype
                else:
                    dtype_to_use = torch.float
            else:
                qconfig = get_cur_qconfig(self.qconfig_dict, seen_op_info.fqn, op)
                if qconfig is None:
                    dtype_to_use = torch.float
                else:
                    dtype_to_use = qconfig.activation().dtype

        elif func_output_dtype_type == FuncOutputDTypeType.DTYPE_DEFAULT_BC_UNSUPPORTED_SYNTAX:
            dtype_to_use = torch.float
        else:
            # TODO(future PR): respect qconfig for torch.cat
            if isinstance(args[0], (tuple, list)):  # for torch.cat
                unique_arg_dtypes = [
                    arg._qtensor_info.inf_dtype for arg in args[0]]
                assert len(set(unique_arg_dtypes)) == 1, \
                    'an iterable with arguments with different inference ' + \
                    'dtypes is not supported yet'
                dtype_to_use = args[0][0]._qtensor_info.inf_dtype
            else:
                dtype_to_use = args[0]._qtensor_info.inf_dtype

        def _add_output_qtensor_info(output):
            output._qtensor_info = QTensorInfo(
                qtensor_id[0], output.dtype, dtype_to_use)  # type: ignore[arg-type]
            self.idx_to_seen_op_infos[self.idx].output_tensor_infos.append(
                output._qtensor_info)
            qtensor_id[0] += 1

        if isinstance(output, torch.Tensor):
            _add_output_qtensor_info(output)
        elif isinstance(output, tuple):
            for element in output:
                if isinstance(element, torch.Tensor):
                    _add_output_qtensor_info(element)

    def _maybe_insert_input_observers(self, seen_op_info: SeenOpInfo):
        func_output_dtype_type = get_func_output_dtype_type(seen_op_info)
        input_observed_arg_idxs = get_input_observed_arg_idxs(
            seen_op_info.type, seen_op_info.type_is_module)

        if func_output_dtype_type == FuncOutputDTypeType.DTYPE_DEPENDS_ON_QCONFIG:
            for idx, tensor_info in enumerate(seen_op_info.input_tensor_infos):
                if tensor_info is None:
                    continue
                if input_observed_arg_idxs is not None and \
                        idx not in input_observed_arg_idxs:
                    continue

                qconfig = get_cur_qconfig(
                    self.qconfig_dict, seen_op_info.fqn, seen_op_info.type)
                if qconfig is None:
                    # If qconfig is None, we do not need any input observers
                    continue

                elif tensor_info.inf_dtype != torch.quint8:
                    # TODO(future PR): this assumes current dtype is quint8,
                    # this is not always true
                    # TODO(future PR): currently this only handles float32 and
                    # quint8, we need to extend it to other dtypes
                    tensor_id = tensor_info.id  # type: ignore[attr-defined]
                    weight_arg_idx = get_weight_arg_idx(seen_op_info.type)
                    obs = qconfig.weight() if idx == weight_arg_idx else \
                        qconfig.activation()
                    self.tensor_id_to_observer[str(tensor_id)] = obs

    def _maybe_insert_output_observers(
        self,
        seen_op_info: SeenOpInfo,
        root_module: torch.nn.Module,
    ):
        func_output_obs_type = get_func_output_obs_type(seen_op_info)
        output_tensor_id = seen_op_info.output_tensor_infos[0].id

        if func_output_obs_type == FuncOutputObsType.NEW_OBS:
            # TODO(future PR): check qconfig is None
            qconfig = get_cur_qconfig(
                self.qconfig_dict, seen_op_info.fqn, seen_op_info.type)
            assert qconfig is not None
            self.tensor_id_to_observer[str(output_tensor_id)] = \
                qconfig.activation()
        elif func_output_obs_type == FuncOutputObsType.REUSES_FIRST_INPUT_OBS:
            assert seen_op_info.input_tensor_infos[0] is not None
            first_input_tensor_id = seen_op_info.input_tensor_infos[0].id

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
                prev_op = get_producer_of_seen_op_info(
                    self.idx_to_seen_op_infos, seen_op_info)
                assert prev_op is not None
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
                    first_input_obs = first_input_mod.activation_post_process
                else:
                    # TODO(future PR): check qconfig is None
                    qconfig = get_cur_qconfig(
                        self.qconfig_dict, seen_op_info.fqn, seen_op_info.type)
                    assert qconfig is not None
                    first_input_obs = qconfig.activation()

            self.tensor_id_to_observer[str(output_tensor_id)] = first_input_obs

    def insert_observers(self, root_module: torch.nn.Module):
        for idx, seen_op_info in self.idx_to_seen_op_infos.items():
            self._maybe_insert_input_observers(seen_op_info)
            self._maybe_insert_output_observers(seen_op_info, root_module)

    # This is a hack to enable nn.Sequential to properly work with
    # this class.
    # TODO(future): remove the hack
    def forward(self, x):
        raise NotImplementedError('Calling AutoQuantizationState.forward is not supported')
        # return x

    def add_seen_op_type_without_op_hooks(self, op_type: Callable) -> None:
        self.seen_op_types_without_op_hooks.add(op_type)
