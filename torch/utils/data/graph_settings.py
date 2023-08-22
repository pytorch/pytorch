import inspect
import warnings

from typing import Any, List, Optional, Set

import torch

from torch.utils.data.datapipes.iter.sharding import (
    _ShardingIterDataPipe,
    SHARDING_PRIORITIES,
)
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps

__all__ = [
    "apply_random_seed",
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


def _is_sharding_datapipe(datapipe: DataPipe) -> bool:
    if isinstance(datapipe, _ShardingIterDataPipe):
        return True
    if hasattr(datapipe, "apply_sharding") and inspect.ismethod(datapipe.apply_sharding):
        return True
    return False


def apply_sharding(datapipe: DataPipe,
                   num_of_instances: int,
                   instance_id: int,
                   sharding_group=SHARDING_PRIORITIES.DEFAULT) -> DataPipe:
    r"""
    Apply dynamic sharding over the ``sharding_filter`` DataPipe that has a method ``apply_sharding``.
    RuntimeError will be raised when multiple ``sharding_filter`` are presented in the same branch.
    """
    graph = traverse_dps(datapipe)

    def _helper(graph, prev_applied=None):
        for (dp, sub_graph) in graph.values():
            applied = None
            if _is_sharding_datapipe(dp):
                if prev_applied is not None:
                    raise RuntimeError("Sharding twice on a single pipeline is likely unintended and will cause data loss. "
                                       f"Sharding already applied to {prev_applied} while trying to apply to {dp}")
                # For BC, only provide sharding_group if accepted
                sig = inspect.signature(dp.apply_sharding)
                if len(sig.parameters) < 3:
                    dp.apply_sharding(num_of_instances, instance_id)
                else:
                    dp.apply_sharding(num_of_instances, instance_id, sharding_group=sharding_group)
                applied = dp
            if applied is None:
                applied = prev_applied
            _helper(sub_graph, applied)

    _helper(graph)

    return datapipe


def _is_shuffle_datapipe(datapipe: DataPipe) -> bool:
    if not hasattr(datapipe, "set_shuffle") or not hasattr(datapipe, "set_seed"):
        return False
    if not inspect.ismethod(datapipe.set_shuffle) or not inspect.ismethod(datapipe.set_seed):
        return False
    return True


def apply_shuffle_settings(datapipe: DataPipe, shuffle: Optional[bool] = None) -> DataPipe:
    r"""
    Traverse the graph of ``DataPipes`` to find and set shuffle attribute
    to each `DataPipe` that has APIs of ``set_shuffle`` and ``set_seed``.

    Args:
        datapipe: DataPipe that needs to set shuffle attribute
        shuffle: Shuffle option (default: ``None`` and no-op to the graph)
    """
    if shuffle is None:
        return datapipe

    graph = traverse_dps(datapipe)
    all_pipes = get_all_graph_pipes(graph)
    shufflers = [pipe for pipe in all_pipes if _is_shuffle_datapipe(pipe)]
    if not shufflers and shuffle:
        warnings.warn(
            "`shuffle=True` was set, but the datapipe does not contain a `Shuffler`. Adding one at the end. "
            "Be aware that the default buffer size might not be sufficient for your task.", stacklevel=2
        )
        datapipe = datapipe.shuffle()
        shufflers = [datapipe, ]  # type: ignore[list-item]

    for shuffler in shufflers:
        shuffler.set_shuffle(shuffle)

    return datapipe


def apply_shuffle_seed(datapipe: DataPipe, rng: Any) -> DataPipe:
    warnings.warn(
        "`apply_shuffle_seed` is deprecated since 1.12 and will be removed in the future releases."
        "\nPlease use `apply_random_seed` instead.", DeprecationWarning, stacklevel=2
    )
    return apply_random_seed(datapipe, rng)


def _is_random_datapipe(datapipe: DataPipe) -> bool:
    if hasattr(datapipe, "set_seed") and inspect.ismethod(datapipe.set_seed):
        return True
    return False


def apply_random_seed(datapipe: DataPipe, rng: torch.Generator) -> DataPipe:
    r"""
    Traverse the graph of ``DataPipes`` to find random ``DataPipe`` with an API of
    ``set_seed`` then set the random seed based on the provided RNG.

    Args:
        datapipe: DataPipe that needs to set randomness
        rng: Random number generator to generate random seeds
    """
    graph = traverse_dps(datapipe)
    all_pipes = get_all_graph_pipes(graph)
    # Using a set to track id of DataPipe to prevent setting randomness per DataPipe more than once.
    # And, `id` is used in case of unhashable DataPipe
    cache = set()
    random_datapipes = []
    for pipe in all_pipes:
        if id(pipe) in cache:
            continue
        if _is_random_datapipe(pipe):
            random_datapipes.append(pipe)
            cache.add(id(pipe))

    for pipe in random_datapipes:
        random_seed = int(torch.empty((), dtype=torch.int64).random_(generator=rng).item())
        pipe.set_seed(random_seed)

    return datapipe
