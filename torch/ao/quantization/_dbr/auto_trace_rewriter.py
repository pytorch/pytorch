import copy
import math
import operator
from types import ModuleType
from typing import Callable, Any, Tuple, Dict

import torch
import torch.fx
import torch.nn.functional as F
from .quantization_state import AutoQuantizationState
from .utils import get_packable_arg_idxs

class AllModuleTracer(torch.fx.Tracer):
    """
    This is a tracer that knows how to convert quantizeable ops with
    dynamic dispatch into their corresponding quantized subgraphs.
    """

    node_name_to_dtype: Dict[str, Any]

    def __init__(self, autowrap_modules: Tuple[ModuleType] = (math, ),
                 autowrap_functions: Tuple[Callable, ...] = (),
                 param_shapes_constant: bool = False) -> None:
        super().__init__(
            autowrap_modules, autowrap_functions,
            param_shapes_constant)
        self.node_name_to_dtype = {}

    def is_leaf_module(self, m, module_qualified_name) -> bool:
        return True

    def _maybe_update_args_with_quants(self, args, arg_quant_infos, target):
        # insert quants for inputs, if needed
        if len(arg_quant_infos):
            new_args = []
            if target == torch.ops.quantized.cat:
                new_first_arg = []
                for idx, input_arg_quant_info in enumerate(arg_quant_infos):
                    if input_arg_quant_info is None:
                        new_first_arg.append(args[0][idx])
                    else:
                        # create a quant node
                        scale, zp = input_arg_quant_info
                        quant = super().create_node(
                            'call_function', torch.quantize_per_tensor,
                            (args[0][idx], scale.item(), zp.item(), torch.quint8), {}, None, None)
                        new_first_arg.append(quant)
                new_args = [new_first_arg, *args[1:]]
            elif target == torch.cat:
                return args
            else:
                # TODO: this is not handling non-tensor tuple args (for example,
                # dilation in conv2d) correctly, it just happens to work but
                # needs a fix.
                for idx, arg in enumerate(args):
                    input_arg_quant_info = arg_quant_infos[idx]
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

    def _maybe_update_args_with_dequants(self, args):
        new_args = []
        for arg in args:
            if (
                isinstance(arg, torch.fx.Node) and
                arg.name in self.node_name_to_dtype and
                self.node_name_to_dtype[arg.name] != torch.float
            ):
                dequant = torch.fx.Proxy(arg).dequantize().node
                new_args.append(dequant)
            else:
                new_args.append(arg)
        return tuple(new_args)

    def _maybe_update_outputs(self, outputs, output_qtensor_infos, output_dtypes):
        # TODO(future PR): handle other output types
        assert len(outputs) == 1 and len(output_qtensor_infos) == 1
        if output_dtypes is not None:
            assert len(output_dtypes) == 1
            output_dtype = output_dtypes[0]
            qtensor_info = output_qtensor_infos[0]
            if qtensor_info.inf_dtype != output_dtype:
                assert output_dtype is torch.float, \
                    'non-float dtypes not handled yet'
                dequant = torch.fx.Proxy(outputs[0]).dequantize().node
                outputs = (dequant,)
        return outputs

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        if target == operator.add:
            target = torch.add
        if target == operator.mul:
            target = torch.mul

        # TODO(future PR): move this into mappings
        if target == 'add':
            target = torch.add
            kind = 'call_function'
        if target == 'mul':
            target = torch.mul
            kind = 'call_function'

        dtype_to_use = torch.float

        if kind == 'call_function' or kind == 'call_method':
            qstate = self.root._auto_quant_state
            assert isinstance(qstate, AutoQuantizationState)
            if qstate.cur_op_needs_hooks(target):
                # need to test this path with call_method
                assert kind == 'call_function'
                qstate.validate_cur_op(target)

                old_target = target
                # TODO use arg_dequant_infos
                new_target, arg_quant_infos, arg_dequant_infos, packed_param_name, additional_kwargs, _, _ = \
                    qstate.get_op_convert_info(target)
                for k in ('scale', 'zero_point'):
                    if k in additional_kwargs:
                        additional_kwargs[k] = additional_kwargs[k].item()
                if new_target is not None:
                    target = new_target
                args = self._maybe_update_args_with_quants(args, arg_quant_infos, target)
                # if there is a packed param, replace the relevant args
                if packed_param_name is not None:
                    new_args_with_packed = []
                    packable_arg_idxs = get_packable_arg_idxs(old_target)
                    added_packed = False
                    for idx, arg in enumerate(args):
                        if packable_arg_idxs is not None and idx in packable_arg_idxs:
                            if not added_packed:
                                # packed_param = getattr(self.root, packed_param_name)
                                packed_param_node = super().create_node(
                                    'get_attr', packed_param_name, (), {}, None, None)
                                new_args_with_packed.append(packed_param_node)
                                added_packed = True
                        else:
                            new_args_with_packed.append(arg)
                    args = tuple(new_args_with_packed)

                # TODO move op-specific logic out of here
                if target is torch.ops.quantized.linear:
                    def linear_rewrite_args(input, weight, bias=None):
                        return (input, weight,
                                additional_kwargs['scale'],
                                additional_kwargs['zero_point'])
                    args = linear_rewrite_args(*args, **kwargs)
                    kwargs = {}
                elif old_target != F.conv2d or target is F.conv2d:
                    kwargs.update(**additional_kwargs)
                else:
                    new_args = [*args]
                    new_args.append(additional_kwargs['scale'])
                    new_args.append(additional_kwargs['zero_point'])
                    args = tuple(new_args)

                dtype_to_use = qstate.get_cur_output_inf_dtype()
                qstate.mark_cur_op_complete(old_target)

            else:
                args = self._maybe_update_args_with_dequants(args)

        elif kind == 'call_module':
            # TODO: handle fqn
            module_instance = getattr(self.root, target)
            qstate = self.root._auto_quant_state
            assert isinstance(qstate, AutoQuantizationState)
            if qstate.cur_op_needs_hooks(module_instance):
                qstate.validate_cur_op(module_instance)

                # TODO use arg_dequant_infos
                _, arg_quant_infos, arg_dequant_infos, _packed_param_name, additional_kwargs, _, _ = \
                    qstate.get_op_convert_info(module_instance)
                for k in ('scale', 'zero_point'):
                    if k in additional_kwargs:
                        additional_kwargs[k] = additional_kwargs[k].item()

                args = self._maybe_update_args_with_quants(args, arg_quant_infos, target)
                kwargs.update(**additional_kwargs)

                dtype_to_use = qstate.get_cur_output_inf_dtype()
                qstate.mark_cur_op_complete(module_instance)

            else:
                args = self._maybe_update_args_with_dequants(args)

        elif kind == 'output':
            qstate = self.root._auto_quant_state
            assert isinstance(qstate, AutoQuantizationState)
            output_qtensor_infos = qstate.get_output_qtensor_infos()
            output_dtypes = qstate.get_output_dtypes()
            args = self._maybe_update_outputs(
                args, output_qtensor_infos, output_dtypes)

        out = super().create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_dtype[out.name] = dtype_to_use
        return out

    # This is a hack to enable nn.Sequential to properly work with this
    # class.
    # TODO(future): remove the hack
    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any], args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
        if isinstance(m, AutoQuantizationState):
            return args[0]
        return super().call_module(m, forward, args, kwargs)

# TODO(future PR): handle cases where the module is not symbolically
# traceable
def rewrite_for_scripting(mod: torch.nn.Module) -> torch.nn.Module:
    """
    Makes the dynamically dispatched ops in `mod` be explicit, so they
    can be visibile to `torch.jit.script`. In detail:

    1. symbolically traces the forward with FX, without any leaves
    2. for each quantizeable op with dynamic dispatch, rewrites the graph to
       contain the quantized subgraph (quant if necessary, quantized op,
       dequant if necessary).
    3. recursively repeat (1 - 2) for each child
    """

    def rewrite_helper(mod : torch.nn.Module):
        copied = copy.copy(mod)
        for name, child in mod.named_children():
            setattr(copied, name, rewrite_helper(child))

        if hasattr(mod, '_auto_quant_state') and (
            mod._auto_quant_state.has_at_least_one_seen_op_info() or  # type: ignore[union-attr, operator]
            (mod._auto_quant_state.get_output_dtypes() is not None)  # type: ignore[union-attr, operator]
        ):
            copied._auto_quant_state.reset_to_new_call()  # type: ignore[union-attr, operator]

            graph = AllModuleTracer().trace(copied)
            return torch.fx.GraphModule(copied, graph, copied.__class__.__name__)
        else:
            return copied

    return rewrite_helper(mod)
