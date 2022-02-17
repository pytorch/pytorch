from lazy_tensor_core import _LAZYC
_LAZYC._ltc_init_ts_backend()
import torch
from torch import Tensor
from torch import nn
import copy
import dis
import sys
import inspect
import dataclasses
import typing
from typing import List, Dict

class ModuleConstScale(nn.Module):
    def __init__(self):
        super(ModuleConstScale, self).__init__()

    def forward(self, a):
        return a * 2

class ModuleSub(nn.Module):
    def __init__(self):
        super(ModuleSub, self).__init__()

    def forward(self, a, b):
        return a - b

def gen_rand_args(mod):
    args = []
    for _ in range(len(inspect.signature(mod.forward).parameters)):
        args.append(torch.randn(2, 3))
    return args

@dataclasses.dataclass
class GraphInputMatcher:
    tensor_id_to_arg_idx: Dict[int, int]
    graph_input_tensor_ids: List[int]
    # there are 2 categories of graph_input_tensors.
    # Category 1: those whose id are not found in tensor_id_to_arg_idx. These are
    # most likely const tensors and we can get its content from graph_input_tensors
    # Category 2: those whose id are found in tensor_id_to_arg_idx. We should get
    #  the tensor from method arguments
    graph_input_tensors: List[Tensor]

    def getRealGraphInputTensors(self, args):
        real_input = []
        for tensor_id, traced_tensor in zip(self.graph_input_tensor_ids, self.graph_input_tensors):
            arg_idx = self.tensor_id_to_arg_idx.get(tensor_id, None)
            if arg_idx is None:
                inp = traced_tensor
            else:
                inp = args[arg_idx]
            real_input.append(inp)
        return real_input

    __call__ = getRealGraphInputTensors

def verify_reusing_compiled_graph(mod):
    args = gen_rand_args(mod)
    out = mod(*args)

    dis.dis(mod.forward)

    lazy_args = [arg.to(device="lazy") for arg in args]
    args_tensor_ids = [_LAZYC._ltc_get_tensor_id(lazy_arg) for lazy_arg in lazy_args]
    tensor_id_to_arg_idx = {tensor_id: i for i, tensor_id in enumerate(args_tensor_ids)}
    lazy_model = copy.deepcopy(mod).to(device="lazy")
    lazy_out = lazy_model(*lazy_args)

    print("LTC IR:", _LAZYC._get_ltc_tensors_text([lazy_out]))
    print("TS IR:", _LAZYC._get_ltc_tensors_backend([lazy_out]))

    graph_input_tensor_ids, graph_input_tensors = _LAZYC._get_ltc_tensors_ts_device_data_node([lazy_out])
    graph_input_matcher = GraphInputMatcher(tensor_id_to_arg_idx, graph_input_tensor_ids, graph_input_tensors)

    graph_hash = _LAZYC._get_graph_hash(lazy_out)

    print("graph_hash", graph_hash)

    print(f"args_tensor_ids {args_tensor_ids}")
    print("tensor ids from device data:", graph_input_tensor_ids)
    print(f"lazy_out {lazy_out.to('cpu')}")
    _LAZYC.shunting_explore() # TODO will remove
    print(f"out is {out}")

    def optimized_mod(*args):
        graph_input = graph_input_matcher(args)
        return _LAZYC._run_cached_graph(graph_hash, graph_input)

    print("return value of optimized_mod", optimized_mod(*args))

    # check correctness
    for i in range(10):
        rand_args = gen_rand_args(mod)
        expected = mod(*rand_args)
        actual = optimized_mod(*rand_args)[0]
        print(f"Check {i}, allclose? {torch.allclose(expected, actual)}, expected {expected}, actual {actual}")
    return optimized_mod

verify_reusing_compiled_graph(ModuleSub())
verify_reusing_compiled_graph(ModuleConstScale())
