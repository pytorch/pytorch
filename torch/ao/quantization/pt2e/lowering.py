import torch
from torch._inductor.constant_folding import constant_fold
from torch._inductor.fx_passes.freezing_patterns import freezing_passes


__all__ = [
    "lower_pt2e_quantized_to_x86",
]


def lower_pt2e_quantized_to_x86(
    model: torch.fx.GraphModule,
    example_inputs: tuple[torch.Tensor, ...],
) -> torch.fx.GraphModule:
    """Lower a PT2E-quantized model to x86 backend.

    Args:
    * `model` (torch.fx.GraphModule): a model quantized by PT2E quantization flow.
    * `example_inputs` (tuple[torch.Tensor, ...]): example inputs for the model.

    Return:
    A GraphModule lowered to x86 backend.
    """

    def _post_autograd_decomp_table():  # type: ignore[no-untyped-def]
        decomp_table = torch.export.default_decompositions()

        # if we are post-autograd, we shouldn't
        # decomp prim ops.
        for k in list(decomp_table.keys()):
            if not torch._export.utils._is_cia_op(k):
                del decomp_table[k]

        return decomp_table

    def _node_replace(m):  # type: ignore[no-untyped-def]
        # Replace aten.t(x) with aten.permute(x, [1, 0])
        aten = torch.ops.aten
        g = m.graph
        for node in g.nodes:
            if node.target is aten.t.default:
                with g.inserting_before(node):
                    x = node.args[0]
                    dims = [1, 0]
                    perm_node = g.call_function(aten.permute.default, args=(x, dims))
                    node.replace_all_uses_with(perm_node)
                    g.erase_node(node)

        g.lint()
        m.recompile()

    lowered_model = (
        torch.export.export(model, example_inputs, strict=True)
        .run_decompositions(_post_autograd_decomp_table())
        .module()
    )
    _node_replace(lowered_model)
    freezing_passes(lowered_model, example_inputs)
    constant_fold(lowered_model)
    return lowered_model
