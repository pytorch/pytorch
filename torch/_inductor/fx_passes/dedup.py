def assert_no_aliased_graph_inputs(graph: torch.fx.Graph) -> None:
    """
    Assert that there is no aliased graph inputs that share the same storage.
    """
    storage_id_to_graph_inputs = defaultdict(list)
    for node in graph.nodes:
        if node.op == "placeholder" and isinstance(node.meta.get('val', None), torch.Tensor):
            storage_id_to_graph_inputs[id(node.meta['val'].untyped_storage())].append(node)
    for aliased_graph_inputs in storage_id_to_graph_inputs.values():
        assert len(aliased_graph_inputs) == 1, f"Found aliased graph inputs: {aliased_graph_inputs}"
