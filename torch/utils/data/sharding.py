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
    already_applied_to = None
    for pipe in all_pipes:
        if hasattr(pipe, 'is_shardable'):
            if pipe.is_shardable():
                if hasattr(pipe, 'apply_sharding'):
                    if already_applied_to is not None:
                        raise RuntimeError('This implementation of sharding can be only applied once per instance of DataPipeline.',
                                           'Already applied to', already_applied_to, 'while trying to apply to', pipe)
                    pipe.apply_sharding(num_of_instances, instance_id)
                    already_applied_to = pipe
