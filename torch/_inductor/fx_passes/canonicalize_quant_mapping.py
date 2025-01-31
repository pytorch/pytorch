import operator

import torch


def canonicalize_quant_mapping(gm: torch.fx.GraphModule) -> None:
    """
    torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'quant_invoke_0_0', (arg0_1, arg1_1));
    ->
    torch.ops.higher_order.invoke_quant(repeated_subgraph0, arg0_1, arg1_1, scheme = 'nf4');
    """
    graph = gm.graph
    invoke_quant_invocations = graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_quant_packed
    )
    for invoke_quant in invoke_quant_invocations:
        kwargs = dict(invoke_quant.kwargs)

        quant_options_node = kwargs.pop("quant_options", None)
        # breakpoint()
        if quant_options_node is not None:
            assert isinstance(quant_options_node, torch.fx.Node)
            quant_options = torch._higher_order_ops.InvokeQuant(
                *invoke_quant.kwargs["quant_options"].args,
                **invoke_quant.kwargs["quant_options"].kwargs,
            )
        else:
            quant_options = None

        # breakpoint()
        subgraph, args = invoke_quant.args
        with gm.graph.inserting_before(invoke_quant):
            invoke_quant_replacement = graph.call_function(
                torch._higher_order_ops.invoke_quant,
                (subgraph, *args),
                kwargs,
            )
            invoke_quant_replacement.meta.update(subgraph.meta)
            invoke_quant_replacement.meta["quant_options"] = quant_options

            invoke_quant.replace_all_uses_with(invoke_quant_replacement)
            graph.erase_node(invoke_quant)

            if quant_options_node is not None and len(quant_options_node.users) == 0:
                graph.erase_node(quant_options_node)

            first_user = next(iter(invoke_quant_replacement.users))

            if (
                len(invoke_quant_replacement.users) == 1
                and len(subgraph.users) == 1
                and first_user.target == operator.getitem
                and first_user.args[1] == 0
            ):
                subgraph_graph = getattr(gm, subgraph.target)
                output_node = torch._inductor.utils.output_node(subgraph_graph)
                assert (
                    isinstance(output_node.args[0], tuple)
                    and len(output_node.args[0]) == 1
                )

                unpacked_output = output_node.args[0][0]
                output_node.args = (unpacked_output,)
                if "val" in output_node.meta:
                    output_node.meta["val"] = output_node.meta["val"][0]
                subgraph_graph.recompile()

                invoke_quant_replacement.meta.update(first_user.meta)
                first_user.replace_all_uses_with(invoke_quant_replacement)
                graph.erase_node(first_user)

    # breakpoint()
    pass
