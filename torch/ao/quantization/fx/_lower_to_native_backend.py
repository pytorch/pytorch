import torch
from torch.nn.quantized.modules.utils import ReferenceableQuantizedModule
from . import subgraph_rewriter_FORKED_DO_NOT_USE
from .graph_module import QuantizedGraphModule
from .quantized_fusion_patterns_and_replacements import get_fbgemm_patterns_and_replacements
from .match_utils import is_match
from .match_utils import MatchAllNode
from ..utils import _parent_name
from typing import Dict, Type

# Mapping from reference module class to the replacement quantized module class for lowering
LOWER_MODULE_MAP: Dict[Type[torch.nn.Module], Type[ReferenceableQuantizedModule]] = {
    torch.nn.quantized._reference.Linear: torch.nn.quantized.Linear,
    torch.nn.quantized._reference.Conv1d: torch.nn.quantized.Conv1d,
    torch.nn.quantized._reference.Conv2d: torch.nn.quantized.Conv2d,
    torch.nn.quantized._reference.Conv3d: torch.nn.quantized.Conv3d,
}

def _lower_weighted_ref_module(model: QuantizedGraphModule, ref_class: Type[torch.nn.Module]) -> QuantizedGraphModule:
    """
    Traverse the graph and find dequantize - ref module - quantize patterns
    and replace them with the quantized version of the ref module.
    """
    if ref_class not in LOWER_MODULE_MAP:
        raise ValueError("Lowering is currently not supported for reference module %s" % ref_class.__name__)
    q_class = LOWER_MODULE_MAP[ref_class]

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
        assert issubclass(q_class, ReferenceableQuantizedModule)  # suppress mypy warnings
        q_module = q_class.from_reference(ref_module, output_scale, output_zero_point)

        # replace reference module with quantized module
        parent_name, module_name = _parent_name(ref_node.target)
        setattr(modules[parent_name], module_name, q_module)
        # remvoe dq node:
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
    module_type_list = [
        torch.nn.ReLU,
        torch.nn.ReLU6,
        torch.nn.AdaptiveAvgPool1d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.AdaptiveAvgPool3d,
        torch.nn.AvgPool1d,
        torch.nn.AvgPool2d,
        torch.nn.AvgPool3d,
        torch.nn.MaxPool1d,
        torch.nn.MaxPool2d,
        torch.nn.MaxPool3d,
    ]
    func_list = [
        torch.nn.functional.adaptive_avg_pool1d,
        torch.nn.functional.adaptive_avg_pool2d,
        torch.nn.functional.adaptive_avg_pool3d,
        torch.nn.functional.max_pool1d,
        torch.nn.functional.max_pool2d,
        torch.nn.functional.max_pool3d,
        torch.nn.functional.relu,
        torch.nn.functional.hardtanh,
        torch.nn.functional.hardtanh_,
    ]
    method_list = [
        torch.mean,
        'relu',
        'relu_',
    ]

    for n in model.graph.nodes:
        q_node = n
        if q_node.target == torch.quantize_per_tensor:
            ref_node = q_node.args[0]

            is_call_function = ref_node.op == "call_function" and ref_node.target in func_list
            is_call_method = ref_node.op == "call_method" and ref_node.target in method_list
            is_call_module = ref_node.op == "call_module" and type(modules[str(ref_node.target)]) in module_type_list

            if is_call_module or is_call_function or is_call_method:
                dq_node = ref_node.args[0]
                if dq_node.target == 'dequantize':
                    # get output scale/zero_point/dtype from the quantize node
                    scale_node = q_node.args[1]
                    zero_point_node = q_node.args[2]
                    dtype = q_node.args[3]

                    if is_call_module:
                        ref_module = modules[ref_node.target]
                        # change this pattern to use the corresponding quantized module
                        # replace reference module with quantized module
                        parent_name, module_name = _parent_name(ref_node.target)
                        setattr(modules[parent_name], module_name, ref_module)
                    else:
                        dq_node.target = ref_node

                    # remvoe dq node:
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
    for ref_class in LOWER_MODULE_MAP.keys():
        model = _lower_weighted_ref_module(model, ref_class)
    model.recompile()

    for pattern, replacement in get_fbgemm_patterns_and_replacements():
        subgraph_rewriter_FORKED_DO_NOT_USE.replace_pattern(model, pattern, replacement)

    special_pattern_replacement(model)

    model.graph.lint()
    return model
