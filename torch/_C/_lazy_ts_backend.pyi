# mypy: allow-untyped-defs
# defined in torch/csrc/lazy/python/init.cpp

from typing import Any, List, Tuple

from torch import Tensor

def _init(): ...
def _get_tensors_ts_device_data_node(
    tensors: List[Tensor],
) -> Tuple[List[int], List[Any]]: ...
def _run_cached_graph(hash_str: str, graph_inputs: List[Any]) -> List[Tensor]: ...
