from lazy_tensor_core import _LAZYC
from lazy_tensor_core.core import lazy_model as lm
import dataclasses
from typing import List, Dict, Any
from torch import Tensor
import copy

@dataclasses.dataclass
class GraphInputMatcher:
    tensor_id_to_arg_idx: Dict[int, int]
    graph_input_tensor_ids: List[int]
    # there are 2 categories of graph_input_tensors.
    # Category 1: those whose id are not found in tensor_id_to_arg_idx. These are
    # most likely const tensors and we can get its content from graph_input_tensors
    # Category 2: those whose id are found in tensor_id_to_arg_idx. We should get
    #  the tensor from method arguments
    graph_input_ivalues: List[Any]

    # get the real graph input tensors
    def __call__(self, args):
        real_input = []
        for tensor_id, traced_ivalue in zip(self.graph_input_tensor_ids, self.graph_input_ivalues):
            arg_idx = self.tensor_id_to_arg_idx.get(tensor_id, None)
            if arg_idx is None:
                inp = traced_ivalue
            else:
                inp = args[arg_idx]
            real_input.append(inp)
        return real_input

def optimize(model, example_inputs):
    """
    Optimize an eager model with LTC and returns a wrapper to execute the
    compiled graph directly without retracing. It depends on other mechanisms
    like TorchDynamo guards to guarantee the returned wrapper is only called
    when it's safe.
    """
    lazy_args = [arg.to(device="lazy") for arg in example_inputs]
    args_tensor_ids = [_LAZYC._ltc_get_tensor_id(lazy_arg) for lazy_arg in lazy_args]
    tensor_id_to_arg_idx = {tensor_id: i for i, tensor_id in enumerate(args_tensor_ids)}
    lazy_model = copy.deepcopy(model).to(device="lazy")
    lazy_out = lazy_model(*lazy_args)
    if not isinstance(lazy_out, (tuple, list)):
        lazy_out = (lazy_out,)

    print("LTC IR:", _LAZYC._get_ltc_tensors_text(lazy_out))
    print("TS IR:", _LAZYC._get_ltc_tensors_backend(lazy_out))

    graph_input_tensor_ids, graph_input_ivalues = _LAZYC._get_ltc_tensors_ts_device_data_node(lazy_out)
    graph_input_matcher = GraphInputMatcher(tensor_id_to_arg_idx, graph_input_tensor_ids, graph_input_ivalues)

    graph_hash = _LAZYC._get_graph_hash(lazy_out)

    print("graph_hash", graph_hash)

    print(f"args_tensor_ids {args_tensor_ids}")
    print("tensor ids from device data:", graph_input_tensor_ids)
    # sync the list of output tensors so the computation graph for these
    # tensors will be cached
    _LAZYC._ltc_sync_multi(lazy_out, [])
    print(f"out is {model(*example_inputs)}")

    _LAZYC.shunting_explore() # TODO will remove
    def optimized_mod(*args):
        graph_input = graph_input_matcher(args)
        return _LAZYC._run_cached_graph(graph_hash, graph_input)

    return optimized_mod
