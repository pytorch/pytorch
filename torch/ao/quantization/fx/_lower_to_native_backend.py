import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.quantized as nnq
import torch.nn.quantized._reference as nnqr
from torch.nn.quantized.modules.utils import ReferenceableQuantizedModule
from . import subgraph_rewriter_FORKED_DO_NOT_USE
from .graph_module import QuantizedGraphModule
from .quantized_fusion_patterns_and_replacements import get_fbgemm_patterns_and_replacements
from .match_utils import is_match
from .match_utils import MatchAllNode
from ..utils import _parent_name, check_node
from typing import Dict, Tuple, Type

# Mapping from reference module class to the replacement quantized module class for lowering
LOWER_MODULE_MAP: Dict[Type[nn.Module], Type[ReferenceableQuantizedModule]] = {
    nnqr.Linear: nnq.Linear,
    nnqr.Conv1d: nnq.Conv1d,
    nnqr.Conv2d: nnq.Conv2d,
    nnqr.Conv3d: nnq.Conv3d,
}

# Mapping from fused module class to a 2-tuple of:
#   1) The inner reference module class
#   2) The replacement quantized module class for lowering
LOWER_FUSED_MODULE_MAP: Dict[Type[nn.Module], Tuple[Type[nn.Module], Type[ReferenceableQuantizedModule]]] = {
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
            assert issubclass(q_class, ReferenceableQuantizedModule)  # suppress mypy warnings
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

def special_pattern_replacement(model: QuantizedGraphModule) -> QuantizedGraphModule:
    modules = dict(model.named_modules(remove_duplicate=False))
    nodes = list(model.graph.nodes)
    for n in model.graph.nodes:
        q_node = n
        if q_node.target == torch.quantize_per_tensor:
            # get output scale/zero_point/dtype from the quantize node
            ref_node, scale_node, zero_point_node, dtype = q_node.args

            is_call_function, is_call_method, is_call_module = check_node(ref_node, modules)
            if is_call_module or is_call_function or is_call_method:
                dq_node = ref_node.args[0]
                if dq_node.target == 'dequantize':
                    if is_call_module:
                        ref_module = modules[ref_node.target]
                        # change this pattern to use the corresponding quantized module
                        # replace reference module with quantized module
                        parent_name, module_name = _parent_name(ref_node.target)
                        setattr(modules[parent_name], module_name, ref_module)
                    else:
                        dq_node.target = ref_node

                    # remove dq node:
                    dq_node_input = dq_node.args[0]
                    dq_node.replace_all_uses_with(dq_node_input)
                    model.graph.erase_node(dq_node)

                    # remove q node and args:
                    q_node_input = q_node.args[0]
                    q_node.replace_all_uses_with(q_node_input)
                    model.graph.erase_node(q_node)
                    model.graph.erase_node(scale_node)
                    model.graph.erase_node(zero_point_node)


    model.recompile()
    return model

def _lower_to_native_backend(model: QuantizedGraphModule) -> QuantizedGraphModule:
    """ Lower a quantized reference model (with reference quantized operator patterns)
    to the native backend in PyTorch (fbgemm/qnnpack), both backends shares the same
    operator signature so they can be lowered with the same function
    """
    model = _lower_weighted_ref_module(model)
    for pattern, replacement in get_fbgemm_patterns_and_replacements():
        subgraph_rewriter_FORKED_DO_NOT_USE.replace_pattern(model, pattern, replacement)
    special_pattern_replacement(model)
    model.graph.lint()
    return model
