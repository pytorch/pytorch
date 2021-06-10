from torch.fx import (
    GraphModule,
)
from .prepare import _prepare
from .convert import _convert
from .graph_module import (
    ObservedGraphModule,
    QuantizedGraphModule,
)

from typing import Any, Dict, Tuple

class Quantizer:
    def prepare(
            self,
            model: GraphModule,
            qconfig_dict: Any,
            node_name_to_scope: Dict[str, Tuple[str, type]],
            prepare_custom_config_dict: Dict[str, Any] = None,
            equalization_qconfig_dict: Dict[str, Any] = None,
            is_standalone_module: bool = False) -> ObservedGraphModule:
        return _prepare(
            model, qconfig_dict, node_name_to_scope, prepare_custom_config_dict,
            equalization_qconfig_dict, is_standalone_module)

    def convert(self, model: GraphModule, is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None,
                is_standalone_module: bool = False,
                _remove_qconfig: bool = True) -> QuantizedGraphModule:
        quantized = _convert(
            model, is_reference, convert_custom_config_dict, is_standalone_module, _remove_qconfig_flag=_remove_qconfig)
        return quantized
