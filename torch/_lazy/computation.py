import torch._C._lazy
import torch._C._lazy_ts_backend

def get_tensors_ts_device_data_node(tensors):
    """Return tensor ids and eager tensors for DeviceData nodes in the
       IR for the passed in lazy tensors."""
    return torch._C._lazy_ts_backend._get_tensors_ts_device_data_node(tensors)

def get_graph_hash(tensors):
    """Return the graph hash for the passed in lazy tensors"""
    return torch._C._lazy._get_graph_hash(tensors)

def run_cached_graph(hash_str, graph_inputs):
    """Running the cached computation graph with the given inputs"""
    return torch._C._lazy_ts_backend._run_cached_graph(hash_str, graph_inputs)
