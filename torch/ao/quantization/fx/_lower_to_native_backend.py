import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.quantized as nnq
import torch.nn.quantized._reference as nnqr
from torch.nn.quantized.modules.utils import WeightedQuantizedModule
from . import subgraph_rewriter_FORKED_DO_NOT_USE
from .graph_module import QuantizedGraphModule
from .quantized_fusion_patterns_and_replacements import get_fbgemm_patterns_and_replacements
from .match_utils import is_match
from .match_utils import MatchAllNode
from .quantization_types import Pattern
from ..utils import _parent_name
from ..qconfig import QConfigAny
from ..quantization_mappings import get_quantized_operator
from .utils import create_node_from_old_node_preserve_meta
from typing import Dict, Tuple, Type, List, Callable, Any, Union
from torch.fx import Node
import operator

QOP_TO_ARG_NAMES_TO_SKIP = {
    torch._ops.ops.quantized.hardswish: ['inplace'],
    torch._ops.ops.quantized.elu: ['inplace'],
    torch._ops.ops.quantized.dropout: ['inplace'],
    torch._ops.ops.quantized.instance_norm:
    ['running_mean', 'running_var', 'use_input_stats', 'momentum'],
}

def _is_node_in_list(node, modules, func_list, method_list, module_type_list):
    is_call_function = node.op == "call_function" and node.target in func_list
    is_call_method = node.op == "call_method" and node.target in method_list
    is_call_module = node.op == "call_module" and type(modules[str(node.target)]) in module_type_list
    return is_call_function, is_call_method, is_call_module

def is_fixed_qparams_node(node, modules):
    func_list = [
        torch.nn.functional.hardsigmoid,
        torch.nn.functional.sigmoid,
        torch.sigmoid,
        torch.tanh,
    ]
    method_list = [
        "hardsigmoid",
        "hardsigmoid_",
        "sigmoid",
        "sigmoid_",
        "tanh",
        "tanh_",
    ]
    module_type_list = [
        torch.nn.Hardsigmoid,
        torch.nn.Sigmoid,
        torch.nn.Tanh,
    ]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)

def is_default_node(node, modules):
    func_list = [
        torch.nn.functional.elu,
        torch.nn.functional.hardswish,
        torch.nn.functional.instance_norm,
        torch.nn.functional.layer_norm,
        torch.nn.functional.leaky_relu,
        torch.nn.functional.dropout,
    ]
    method_list: List[Any] = []
    module_type_list = [
        nnqr.ConvTranspose1d,
        nnqr.ConvTranspose2d,
        torch.nn.ELU,
        torch.nn.LeakyReLU,
        torch.nn.Hardswish,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
    ]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)

def is_copy_node(node, modules):
    func_list = [
        torch.adaptive_avg_pool1d,
        torch.nn.functional.adaptive_avg_pool2d,
        torch.nn.functional.adaptive_avg_pool3d,
        torch.nn.functional.hardtanh,
        torch.nn.functional.hardtanh_,
        torch.nn.functional.interpolate,
        torch.nn.functional.max_pool1d,
        torch.nn.functional.max_pool2d,
        torch.nn.functional.max_pool3d,
        torch.nn.functional.relu,
        torch.nn.functional.relu6,
        torch.avg_pool1d,
        torch._C._nn.avg_pool2d,
        torch._C._nn.avg_pool3d,
        torch.clamp,
        torch.flatten,
        torch.mean,
        operator.floordiv,
    ]
    method_list = [
        "clamp",
        "mean",
        "relu",
        "relu_",
    ]
    module_type_list = [
        torch.nn.AdaptiveAvgPool1d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.AdaptiveAvgPool3d,
        torch.nn.AvgPool1d,
        torch.nn.AvgPool2d,
        torch.nn.AvgPool3d,
        torch.nn.Hardtanh,
        torch.nn.MaxPool1d,
        torch.nn.MaxPool2d,
        torch.nn.MaxPool3d,
        torch.nn.ReLU,
        torch.nn.ReLU6,
    ]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)

def is_general_tensor_shape_node(node, modules):
    func_list = [
        torch.transpose,
        torch.repeat_interleave,
        torch.squeeze,
        torch.stack,
        torch.unsqueeze,
    ]
    method_list = [
        "contiguous",
        "detach",
        "detach_",
        "permute",
        "repeat",
        "repeat_interleave",
        "reshape",
        "resize_",
        "shape",
        "size",
        "squeeze",
        "squeeze_",
        "transpose",
        "unsqueeze",
        "unsqueeze_",
        "view",
    ]
    module_type_list = [
        torch.nn.Identity,
    ]
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)

def is_other_node(node, modules):
    func_list = [
        torch.cat,
    ]
    method_list: List[Any] = []
    module_type_list: List[Any] = []
    return _is_node_in_list(node, modules, func_list, method_list, module_type_list)

def is_special_pattern_node(node, modules):
    res_function, res_method, res_module = False, False, False
    for checker in [is_fixed_qparams_node, is_default_node, is_copy_node, is_general_tensor_shape_node, is_other_node]:
        is_call_function, is_call_method, is_call_module = checker(node, modules)
        res_function = res_function or is_call_function
        res_method = res_method or is_call_method
        res_module = res_module or is_call_module
    return res_function, res_method, res_module


def is_dequantize_node(node):
    return isinstance(node, Node) and node.op == 'call_method' and node.target == 'dequantize'

# Mapping from reference module class to the replacement quantized module class for lowering
LOWER_MODULE_MAP: Dict[Type[nn.Module], Type[WeightedQuantizedModule]] = {
    nnqr.Linear: nnq.Linear,
    nnqr.Conv1d: nnq.Conv1d,
    nnqr.Conv2d: nnq.Conv2d,
    nnqr.Conv3d: nnq.Conv3d,
}

# TODO: merge with LOWER_MODULE_MAP after we merge
# _lower_weighted_ref_module and special_pattern_replacement
SPECIAL_PATTERN_LOWER_MODULE_MAP = {
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nnqr.ConvTranspose1d: nnq.ConvTranspose1d,
    nnqr.ConvTranspose2d: nnq.ConvTranspose2d,
    nn.ELU: nnq.ELU,
    nn.LeakyReLU: nnq.LeakyReLU,
    nn.Hardswish: nnq.Hardswish,
    nn.InstanceNorm1d: nnq.InstanceNorm1d,
    nn.InstanceNorm2d: nnq.InstanceNorm2d,
    nn.InstanceNorm3d: nnq.InstanceNorm3d,
    nn.LayerNorm: nnq.LayerNorm,
    nn.Dropout: nnq.Dropout,
}

# Mapping from fused module class to a 2-tuple of:
#   1) The inner reference module class
#   2) The replacement quantized module class for lowering
LOWER_FUSED_MODULE_MAP: Dict[Type[nn.Module], Tuple[Type[nn.Module], Type[WeightedQuantizedModule]]] = {
    nni.LinearReLU: (nnqr.Linear, nniq.LinearReLU)
}

def _lower_weighted_ref_module(model: QuantizedGraphModule) -> QuantizedGraphModule:
    """
    Traverse the graph and find dequantize - ref module - quantize patterns
    and replace them with the quantized version of the ref module.
    """
    for ref_class in list(LOWER_MODULE_MAP.keys()) + list(LOWER_FUSED_MODULE_MAP.keys()):
        pattern = (torch.quantize_per_tensor,
                   (ref_class, "dequantize"),
                   MatchAllNode, MatchAllNode, MatchAllNode)
        modules = dict(model.named_modules(remove_duplicate=False))
        nodes = list(model.graph.nodes)
        # TODO: maybe orgnize this better (e.g. break down to more functions)
        # to make this function more readable
        for n in model.graph.nodes:
            if not is_match(modules, n, pattern):
                continue
            q_node = n
            ref_node = q_node.args[0]
            dq_node = ref_node.args[0]
            # get output scale/zero_point/dtype from the quantize node
            scale_node = q_node.args[1]
            zero_point_node = q_node.args[2]
            dtype = q_node.args[3]

            # this can be removed if we add support for "get_attr" in is_match
            if scale_node.op != "get_attr" or zero_point_node.op != "get_attr":
                print("Find the pattern but scale_node and zero_point node are not `get_attr`,"
                      f"got: {scale_node.format_node} {zero_point_node.format_node()}")
                continue

            # this can be removed if we add support for constants in is_match
            if dtype != torch.quint8:
                print(f"Only qint8 output for quantized op is supported, got: {dtype}")
                continue

            # change this pattern to use the corresponding quantized module
            ref_module = modules[ref_node.target]
            output_scale = getattr(model, scale_node.target)
            output_zero_point = getattr(model, zero_point_node.target)
            # For fused modules, we also check whether the inner module is a reference module
            # If so, we replace the entire fused module with the corresponding quantized module
            if ref_class in LOWER_FUSED_MODULE_MAP:
                inner_ref_class, q_class = LOWER_FUSED_MODULE_MAP[ref_class]
                if type(ref_module[0]) != inner_ref_class:
                    continue
            else:
                q_class = LOWER_MODULE_MAP[type(ref_module)]
            assert issubclass(q_class, WeightedQuantizedModule)  # suppress mypy warnings
            q_module = q_class.from_reference(ref_module, output_scale, output_zero_point)

            # replace reference module with quantized module
            parent_name, module_name = _parent_name(ref_node.target)
            setattr(modules[parent_name], module_name, q_module)
            # remove dq node:
            dq_node_input = dq_node.args[0]

            dq_node.replace_all_uses_with(dq_node_input)
            model.graph.erase_node(dq_node)

            # remove q node and args:
            q_node.replace_all_uses_with(ref_node)
            model.graph.erase_node(q_node)
            model.graph.erase_node(scale_node)
            model.graph.erase_node(zero_point_node)
        model.recompile()
    return model

def _lower_quantized_binary_op(
    model: QuantizedGraphModule,
    qconfig_map: Dict[str, QConfigAny]
) -> QuantizedGraphModule:
    modules = dict(model.named_modules(remove_duplicate=False))

    def get_bop_patterns(bop: Any) -> List[Pattern]:
        patterns: List[Pattern] = []
        bop_pattern = (bop, MatchAllNode, MatchAllNode)
        for relu_op in [torch.relu, torch.nn.functional.relu, torch.nn.ReLU]:
            patterns.append(
                (torch.quantize_per_tensor,
                 (relu_op, bop_pattern),
                 MatchAllNode, MatchAllNode, MatchAllNode))
        patterns.append(
            (torch.quantize_per_tensor,
             bop_pattern,
             MatchAllNode, MatchAllNode, MatchAllNode))
        return patterns

    patterns: List[Pattern] = []
    for bop in [operator.add, torch.add, operator.mul, torch.mul]:
        patterns.extend(get_bop_patterns(bop))
    patterns.extend(
        [
            (torch.quantize_per_tensor,
             (torch.matmul, "dequantize", "dequantize"),
             MatchAllNode, MatchAllNode, MatchAllNode)
        ]
    )

    qbin_op_mapping: Dict[Union[Callable, str], Callable] = {
        operator.add: torch.ops.quantized.add,
        torch.add: torch.ops.quantized.add,
        operator.mul: torch.ops.quantized.mul,
        torch.mul: torch.ops.quantized.mul,
        torch.matmul: torch.ops.quantized.matmul,
    }
    qbin_relu_op_mapping: Dict[Union[Callable, str], Callable] = {
        operator.add: torch.ops.quantized.add_relu,
        torch.add: torch.ops.quantized.add_relu,
        operator.mul: torch.ops.quantized.mul_relu,
        torch.mul: torch.ops.quantized.mul_relu,
    }
    for pattern in patterns:
        for n in model.graph.nodes:
            if not is_match(modules, n, pattern):
                continue
            q_node = n
            is_quantize = q_node.target == torch.quantize_per_tensor
            is_to_fp16 = q_node.op == "call_method" and q_node.target == "to" and q_node.args[1] == torch.float16
            if not (is_quantize or is_to_fp16):
                continue

            # start tracing back from quantize node
            node = q_node.args[0]
            if not isinstance(node, Node):
                continue
            relu_node = None
            if (
                node.op == 'call_function' and
                    node.target in (torch.nn.functional.relu, torch.relu)
            ) or (
                node.op == 'call_module' and
                    isinstance(modules[str(node.target)], torch.nn.ReLU)
            ):
                relu_node = node
                node = node.args[0]

            # binary operator node, e.g. torch.add(x, y)
            bop_node = node
            if bop_node.op != "call_function" or \
               bop_node.target not in set([torch.add, operator.add, torch.mul, operator.mul, torch.matmul]):
                continue

            # skip lowering for ops that is configured with None qconfig
            # Note: maybe need to generalize this to also check for the dtype, and we
            # only lower when dtype matches, but right now fbgemm/qnnpack only support
            # a single dtype, so it is OK for now
            if bop_node.name in qconfig_map and qconfig_map[bop_node.name] is None:
                continue

            # remove dequant node
            arg0 = bop_node.args[0]
            arg1 = bop_node.args[1]
            dq_node0, dq_node1 = None, None
            if is_dequantize_node(arg0):
                dq_node0 = arg0
            if is_dequantize_node(arg1):
                dq_node1 = arg1
            if dq_node0 is None and dq_node1 is None:
                continue
            for dq_node in [dq_node0, dq_node1]:
                if dq_node is None:
                    continue
                # dequantize node is only used once, this is enforced by `is_match`
                dn_input = dq_node.args[0]
                dq_node.replace_all_uses_with(dn_input)
                model.graph.erase_node(dq_node)

            # swap binary op to quantized binary op
            assert bop_node.target in qbin_op_mapping
            binop_to_qbinop = qbin_op_mapping if relu_node is None else qbin_relu_op_mapping
            qbin_op = binop_to_qbinop[bop_node.target]
            # prepare the args for quantized bianry op
            # (x, y)
            qop_node_args = list(bop_node.args)
            # (x, y, scale, zero_point)
            # add scale and zero_point arguments for Tensor - Tensor operation
            if dq_node0 is not None and dq_node1 is not None:
                qop_node_args.extend([q_node.args[1], q_node.args[2]])

            # insert a call to quantized binary op and remove the original binary op
            with model.graph.inserting_after(q_node):
                qop_node = create_node_from_old_node_preserve_meta(
                    model.graph,
                    ("call_function", qbin_op, tuple(qop_node_args), {}),
                    bop_node)
                q_node.replace_all_uses_with(qop_node)

            # remove quantize node
            model.graph.erase_node(q_node)
            # remove relu node if any
            if relu_node is not None:
                model.graph.erase_node(relu_node)
            # remove binary op node
            model.graph.erase_node(bop_node)

    return model

def special_pattern_replacement(model: QuantizedGraphModule) -> QuantizedGraphModule:
    modules = dict(model.named_modules(remove_duplicate=False))
    for n in model.graph.nodes:
        q_node = n
        is_quantize = q_node.target == torch.quantize_per_tensor
        is_to_fp16 = q_node.op == "call_method" and q_node.target == "to" and q_node.args[1] == torch.float16
        if not (is_quantize or is_to_fp16):
            continue
        ref_node = q_node.args[0]
        # get output scale/zero_point/dtype from the quantize node
        # ref_node, scale_node, zero_point_node, dtype = q_node.args
        # TODO: add safety checks that users for the ref_node and dq_node needs to be one
        is_call_function, is_call_method, is_call_module = is_fixed_qparams_node(ref_node, modules)
        if is_to_fp16 and (is_call_function or is_call_method or is_call_module):
            # TODO: add a warning or error out here? (bc-breaking if error out)
            # warnings.warn(
            #     "Only reference patterns are currently supported for {dtype} dtype with {op} op"
            #     "".format(dtype=dtypes, op=ref_node))
            continue

        is_call_function, is_call_method, is_call_module = is_default_node(ref_node, modules)
        if is_to_fp16 and (is_call_function or is_call_method or is_call_module):
            # TODO: add a warning or error out here? (bc-breaking if error out)
            continue

        # This check includes all supported ops
        is_call_function, is_call_method, is_call_module = is_special_pattern_node(ref_node, modules)
        if not (is_call_module or is_call_function or is_call_method):
            continue
        dq_node_or_nodes = ref_node.args[0]
        assert isinstance(dq_node_or_nodes, Node) or isinstance(dq_node_or_nodes, (tuple, list))
        is_dequantize = False
        if isinstance(dq_node_or_nodes, Node):
            is_dequantize = dq_node_or_nodes.op == 'call_method' and \
                dq_node_or_nodes.target == 'dequantize'
        elif isinstance(dq_node_or_nodes, (tuple, list)):
            is_dequantize = all(
                x.op == 'call_method' and x.target == 'dequantize'
                for x in dq_node_or_nodes)

        if not is_dequantize:
            continue

        # TODO: enable we have patterns that needs to swap the modules
        if is_call_module:
            ref_module = modules[ref_node.target]
            if type(ref_module) in SPECIAL_PATTERN_LOWER_MODULE_MAP and is_quantize:
                qmodule_cls = SPECIAL_PATTERN_LOWER_MODULE_MAP.get(type(ref_module))
                scale_node = q_node.args[1]
                zero_point_node = q_node.args[2]
                output_scale = getattr(model, scale_node.target)
                output_zero_point = getattr(model, zero_point_node.target)

                qmodule = qmodule_cls.from_reference(ref_module, output_scale, output_zero_point)  # type:ignore[union-attr]
                # replace reference module with quantized module
                parent_name, module_name = _parent_name(ref_node.target)
                setattr(modules[parent_name], module_name, qmodule)

        # remove dq node:
        dq_nodes: List[Node] = []
        if isinstance(dq_node_or_nodes, Node):
            dq_nodes = [dq_node_or_nodes]
        elif isinstance(dq_node_or_nodes, (tuple, list)):
            dq_nodes = list(dq_node_or_nodes)

        for dq_node in dq_nodes:
            dn_input = dq_node.args[0]
            dq_node.replace_all_uses_with(dn_input)
            model.graph.erase_node(dq_node)

        # store q node args
        qnode_qparams = list(q_node.args)[1:]
        # replace uses of q node with input and remove q node
        q_node_input = q_node.args[0]
        q_node.replace_all_uses_with(q_node_input)
        model.graph.erase_node(q_node)

        is_call_function, is_call_method, is_call_module = is_default_node(ref_node, modules)
        if is_call_function:
            # pass scale/zer_point arguments from quantize_per_tensor to the default node operator
            # insert an op after the zero_point node so that the scale/zero_point
            # nodes are is available
            qop = get_quantized_operator(ref_node.target)
            args = list(ref_node.args)
            kwargs = dict(ref_node.kwargs)
            if qop in QOP_TO_ARG_NAMES_TO_SKIP:
                args_to_skip = QOP_TO_ARG_NAMES_TO_SKIP[qop]
                for arg in args_to_skip:
                    if arg in kwargs:
                        kwargs.pop(arg)
            kwargs["output_scale"] = qnode_qparams[0]
            kwargs["output_zero_point"] = qnode_qparams[1]
            with model.graph.inserting_after(qnode_qparams[1]):
                qop_node = create_node_from_old_node_preserve_meta(
                    model.graph,
                    ("call_function", qop, tuple(args), kwargs),
                    ref_node)
                ref_node.replace_all_uses_with(qop_node)
                model.graph.erase_node(ref_node)
        else:
            # remove scale/zero_point node for quantize node
            for n in qnode_qparams:
                if isinstance(n, Node):
                    model.graph.erase_node(n)


    model.recompile()
    return model

def _lower_to_native_backend(
    model: QuantizedGraphModule,
    qconfig_map: Dict[str, QConfigAny]
) -> QuantizedGraphModule:
    """ Lower a quantized reference model (with reference quantized operator patterns)
    to the native backend in PyTorch (fbgemm/qnnpack), both backends shares the same
    operator signature so they can be lowered with the same function
    """
    model = _lower_weighted_ref_module(model)
    for pattern, replacement in get_fbgemm_patterns_and_replacements():
        subgraph_rewriter_FORKED_DO_NOT_USE.replace_pattern(model, pattern, replacement)
    _lower_quantized_binary_op(model, qconfig_map)
    special_pattern_replacement(model)
    model.graph.lint()
    return model
