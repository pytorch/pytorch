from collections.abc import Sequence

import torch
import torch._C._lazy
import torch._C._lazy_ts_backend


def get_tensors_ts_device_data_node(
    tensors: Sequence[torch.Tensor],
) -> tuple[list[int], list[object]]:
    """Return tensor ids and eager tensors for DeviceData nodes in the
    IR for the passed in lazy tensors.

    TODO: This API is currently ts backend specific. We are working on
    generalizing it to all backends including XLA.
    """
    return torch._C._lazy_ts_backend._get_tensors_ts_device_data_node(list(tensors))


def get_graph_hash(tensors: Sequence[torch.Tensor]) -> str:
    """Return the graph hash for the passed in lazy tensors"""
    return torch._C._lazy._get_graph_hash(list(tensors))


def run_cached_graph(
    hash_str: str, graph_inputs: Sequence[object]
) -> list[torch.Tensor]:
    """Running the cached computation graph with the given inputs

    TODO: This API is currently ts backend specific. We are working on
    generalizing it to all backends including XLA.
    """
    return torch._C._lazy_ts_backend._run_cached_graph(hash_str, list(graph_inputs))
