from typing import Dict, Tuple

from torch.ao.quantization.qconfig import QConfigAny
from torch.fx import GraphModule

from ._lower_to_native_backend import _lower_to_native_backend


__all__ = ["lower_to_fbgemm"]


def lower_to_fbgemm(
    model: GraphModule,
    qconfig_map: Dict[str, QConfigAny],
    node_name_to_scope: Dict[str, Tuple[str, type]],
    keep_original_weights: bool = False,
) -> GraphModule:
    """Lower a quantized reference model (with reference quantized operator patterns)
    to fbgemm
    """
    return _lower_to_native_backend(
        model, qconfig_map, node_name_to_scope, keep_original_weights
    )
