import torch
from torch.fx.node import map_arg
from torch.fx.passes.infra.pass_base import PassBase, PassResult


# Only for per tensor quant since permute may changes the channel idx
_QUANTIZE_OPS = [
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
]

_VIEW_OPS = [
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
    torch.ops.aten.view.default,
]


class QuantLiftUp(PassBase):
    """
    Lift up the quant node before view like nodes. It can benefit performance
    of Attention like block. For example, we have the pattern as:

             DQ
    DQ       LINEAR
    LINEAR   VIEW
    VIEW     PERMUTE
    PERMUTE  TRANSPOSE
    Q        Q
    DQ       DQ
       Matmul
        DIV
        ADD
      SOFTMAX

    We want to lift up the the quant nodes from matmul before view like nodes
    as the output of Linear node.

             DQ
    DQ       LINEAR
    LINEAR   Q
    Q        VIEW
    VIEW     PERMUTE
    PERMUTE  TRANSPOSE
    DQ       DQ
       Matmul
        DIV
        ADD
      SOFTMAX

    It produces a DQ->LINEAR->Q pattern which can be fused by backend.
    """

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if (
                node.op == "call_function"
                and node.target in _QUANTIZE_OPS
                and len(node.all_input_nodes) == 1
            ):
                quant_node = node
                input_node_of_quant = quant_node.args[0]

                if input_node_of_quant.target in _VIEW_OPS:
                    # Replace dequant's input from quant to quant's input
                    quant_node.replace_all_uses_with(input_node_of_quant)

                    # Find where to insert the new quant node
                    current_node = quant_node
                    input_node = current_node.args[0]
                    while input_node.target in _VIEW_OPS:
                        current_node = input_node
                        input_node = current_node.args[0]

                    # Insert the new quant node
                    with graph_module.graph.inserting_before(current_node):
                        new_quant_node = graph_module.graph.node_copy(quant_node)
                        input_node.replace_all_uses_with(new_quant_node)

                        def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
                            if n == input_node_of_quant:
                                return input_node
                            else:
                                return n

                        new_args = map_arg(new_quant_node.args, maybe_replace_node)
                        new_kwargs = map_arg(new_quant_node.kwargs, maybe_replace_node)
                        new_quant_node.args = new_args
                        new_quant_node.kwargs = new_kwargs

        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return PassResult(graph_module, True)
