from torch.fx import subgraph_rewriter
from .graph_module import QuantizedGraphModule
from .quantized_fusion_patterns_and_replacements import get_fbgemm_patterns_and_replacements

def _lower_to_native_backend(model: QuantizedGraphModule) -> QuantizedGraphModule:
    """ Lower a quantized reference model (with reference quantized operator patterns)
    to the native backend in PyTorch (fbgemm/qnnpack), both backends shares the same
    operator signature so they can be lowered with the same function
    """
    module_dict = dict(model.named_modules())
    for pattern, replacement in get_fbgemm_patterns_and_replacements():
        subgraph_rewriter.replace_pattern(model, pattern, replacement)
    model.graph.lint()
    return model
