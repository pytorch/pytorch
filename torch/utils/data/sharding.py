import torch.utils.data.graph


def apply_sharding(datapipe, num_of_instances, instance_id):
    graph = torch.utils.data.graph.traverse(datapipe)

    def traverse_graph(graph):
        results = set()
        for datapipe, sub_graph in graph.items():
            results.add(datapipe)
            sub_items = traverse_graph(sub_graph)
            for item in sub_items:
                results.add(item)
        return results

    all_pipes = traverse_graph(graph)

    for pipe in all_pipes:
        if hasattr(pipe, 'is_shardable'):
            if pipe.is_shardable():
                if hasattr(pipe, 'apply_sharding'):
                    pipe.apply_sharding(num_of_instances, instance_id)
