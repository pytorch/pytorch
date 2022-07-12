import warnings

from typing import Any, List, Optional, Set

import torch
import torch.utils.data.datapipes as dp

from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse

__all__ = [
    "apply_sharding",
    "apply_shuffle_seed",
    "apply_shuffle_settings",
    "get_all_graph_pipes",
]


def get_all_graph_pipes(graph: DataPipeGraph) -> List[DataPipe]:
    return _get_all_graph_pipes_helper(graph, set())

def _get_all_graph_pipes_helper(graph: DataPipeGraph, id_cache: Set[int]) -> List[DataPipe]:
    results: List[DataPipe] = []
    for dp_id, (datapipe, sub_graph) in graph.items():
        if dp_id in id_cache:
            continue
        id_cache.add(dp_id)
        results.append(datapipe)
        results.extend(_get_all_graph_pipes_helper(sub_graph, id_cache))
    return results


def apply_sharding(datapipe: DataPipe, num_of_instances: int, instance_id: int) -> DataPipe:
    graph = traverse(datapipe, only_datapipe=True)
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
    return datapipe


def apply_shuffle_settings(datapipe: DataPipe, shuffle: Optional[bool]) -> DataPipe:
    if shuffle is None:
        return datapipe

    graph = traverse(datapipe, only_datapipe=True)
    all_pipes = get_all_graph_pipes(graph)
    shufflers = [pipe for pipe in all_pipes if isinstance(pipe, (dp.iter.Shuffler, dp.map.Shuffler))]
    if not shufflers and shuffle:
        warnings.warn(
            "`shuffle=True` was set, but the datapipe does not contain a `Shuffler`. Adding one at the end. "
            "Be aware that the default buffer size might not be sufficient for your task."
        )
        datapipe = datapipe.shuffle()
        shufflers = [datapipe, ]  # type: ignore[list-item]

    for shuffler in shufflers:
        shuffler.set_shuffle(shuffle)

    return datapipe


def apply_shuffle_seed(datapipe: DataPipe, rng: Any) -> DataPipe:
    graph = traverse(datapipe, only_datapipe=True)
    all_pipes = get_all_graph_pipes(graph)
    shufflers = {pipe for pipe in all_pipes if isinstance(pipe, (dp.iter.Shuffler, dp.map.Shuffler))}

    for shuffler in shufflers:
        shuffle_seed = int(torch.empty((), dtype=torch.int64).random_(generator=rng).item())
        shuffler.set_seed(shuffle_seed)

    return datapipe
