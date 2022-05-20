import torch.utils.data.graph
from torch.utils.data.datapipes.iter import Shuffler
import warnings

__all__ = [
    "get_all_graph_pipes",
    "apply_sharding",
    "apply_shuffle_settings",
]


def get_all_graph_pipes(graph):
    results = set()
    for datapipe, sub_graph in graph.items():
        results.add(datapipe)
        sub_items = get_all_graph_pipes(sub_graph)
        for item in sub_items:
            results.add(item)
    return results


def apply_sharding(datapipe, num_of_instances, instance_id):
    graph = torch.utils.data.graph.traverse(datapipe, only_datapipe=True)
    all_pipes = get_all_graph_pipes(graph)
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


def apply_shuffle_settings(datapipe, shuffle):
    if shuffle is None:
        return datapipe

    graph = torch.utils.data.graph.traverse(datapipe, only_datapipe=True)
    all_pipes = get_all_graph_pipes(graph)
    shufflers = {pipe for pipe in all_pipes if isinstance(pipe, Shuffler)}
    if not shufflers and shuffle:
        warnings.warn(
            "`shuffle=True` was set, but the datapipe does not contain a `Shuffler`. Adding one at the end. "
            "Be aware that the default buffer size might not be sufficient for your task."
        )
        datapipe = datapipe.shuffle()
        shufflers = {datapipe}

    for shuffler in shufflers:
        shuffler.set_shuffle(shuffle)

    return datapipe
