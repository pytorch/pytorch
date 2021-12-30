import torch
from torch.fx.graph import (
    Node,
    Graph,
)

from ..utils import (
    get_qconfig_dtypes,
    activation_dtype,
)

from .utils import (
    quantize_node,
)

from .quantization_patterns import (
    QuantizeHandler,
)

from ..qconfig import QConfigAny

from typing import Any, Callable, Dict, Tuple

class CommonQuantizeHandler(QuantizeHandler):
    """ Common quantized op, first input and first output will be quantized
    """
    def __init__(
            self,
            node: Node,
            modules: Dict[str, torch.nn.Module]):
        super().__init__(node, modules)
        if node.op == "call_function" or node.op == "call_method":
            self.op = node.target
        elif node.op == "call_module":
            self.op = type(modules[str(node.target)])

    def convert(self,
                node: Node,
                qconfig: QConfigAny,
                modules: Dict[str, torch.nn.Module],
                quantized_graph: Graph,
                node_name_to_scope: Dict[str, Tuple[str, type]],
                load_arg: Callable,
                is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None) -> Node:
        if not self.all_node_args_are_tensors:
            return NotImplemented
        assert node.op in ['call_module', 'call_function'], 'Only call_module and ' + \
            'call_function are handled in DefaultNode'
        assert is_reference
        if convert_custom_config_dict is None:
            convert_custom_config_dict = {}
        additional_static_quant_mapping = convert_custom_config_dict.get("static", {})

        dtypes = get_qconfig_dtypes(qconfig)
        # We can produce reference for a dtypes including
        # (torch.quint8, torch.qint8, torch.qint32, torch.float16)
        act_dtype = activation_dtype(qconfig)
        if act_dtype == torch.float:
            op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
            return op_out
        else:
            activation_post_process = \
                self._maybe_get_last_node_only_observer(modules)
            assert activation_post_process is not None
            # make sure the input is quantized to act_dtype
            load_arg(quantized={0: act_dtype})(node.args)
            args = load_arg(quantized=torch.float)(node.args)
            kwargs = load_arg(quantized=torch.float)(node.kwargs)
            op_out = quantized_graph.node_copy(node, load_arg(quantized=torch.float))
            return quantize_node(
                op_out, activation_post_process,
                node, modules, quantized_graph, node_name_to_scope, is_input=False)
