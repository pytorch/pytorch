from lazy_tensor_core import _LAZYC
_LAZYC._ltc_init_ts_backend()
import torch
from torch import nn
import copy
import dis
import sys
import inspect

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

def verify_reusing_compiled_graph(mod):
    args = gen_rand_args(mod)
    out = mod(*args)

    lazy_args = [arg.to(device="lazy") for arg in args]
    args_tensor_ids = [_LAZYC._ltc_get_tensor_id(lazy_arg) for lazy_arg in lazy_args]
    tensor_id_to_arg_idx = {tensor_id: i for i, tensor_id in enumerate(args_tensor_ids)}
    lazy_model = copy.deepcopy(mod).to(device="lazy")
    lazy_out = lazy_model(*lazy_args)

    graph_input_tensor_ids = _LAZYC._get_ltc_tensors_ts_device_data_node([lazy_out])
    # TODO: it fails if some graph input is not a method parameter
    graph_input_arg_idx = [tensor_id_to_arg_idx[tensor_id] for tensor_id in graph_input_tensor_ids]

    print(f"args_tensor_ids {args_tensor_ids}")
    print("tensor ids from device data:", graph_input_tensor_ids)
    print(f"lazy_out {lazy_out.to('cpu')}")
    _LAZYC.shunting_explore() # TODO will remove
    print(f"out is {out}")
    print("graph_input_arg_idx", graph_input_arg_idx)

    def permute_args(args):
        return [args[arg_idx] for arg_idx in graph_input_arg_idx]

    def optimized_mod(*args):
        permuted_args = permute_args(args)
        return _LAZYC._run_cached_graph(permuted_args)

    print("return value of optimized_mod", optimized_mod(*args))

    # check correctness
    for i in range(10):
        rand_args = gen_rand_args(mod)
        expected = mod(*rand_args)
        actual = optimized_mod(*rand_args)[0]
        print(f"Check {i}, allclose? {torch.allclose(expected, actual)}")
    return optimized_mod

verify_reusing_compiled_graph(ModuleSub())
# XXX this one does not work yet since there are some graph input having no matching
# method parameter.
verify_reusing_compiled_graph(ModuleConstScale())
