import torch
import math
import timeit
from torch.distributed.tensor import (
    DTensor,
    DeviceMesh,
    distribute_tensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.placement_types import _StridedShard

from torch.distributed.tensor._ops._matrix_ops import mm_strategy, mm_single_dim_strategy
from torch.distributed.tensor._ops._pointwise_ops import pointwise_strategy, single_mesh_dim_pointwise_strategy
from torch.distributed.tensor._ops.utils import _expand_single_dim_strategy_to_mesh
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy, OpSpec


def _gen_tensor_meta(shape):
    empty_tensor = torch.empty(shape)
    return TensorMeta(
        empty_tensor.shape,
        empty_tensor.stride(),
        empty_tensor.dtype,
    )

def _op_schema(mesh, op_overload, inp_shapes, inp_dim_maps):
    """
    Not sure how dim_maps work, but they are some short-hand for specifying shardings on tensors
    """
    args_schema = tuple([
        OpStrategy([OpSpec(DTensorSpec.from_dim_map(mesh, dim_map, sums=[], tensor_meta=_gen_tensor_meta(shape)))])
        for (shape, dim_map) in zip(inp_shapes, inp_dim_maps)
    ])
    return OpSchema(op_overload, args_schema, {})

pow = 6
world_size = 2 ** pow
torch.distributed.init_process_group(
    "fake",
    rank=0,
    world_size=world_size,
)
initial_dims = [2,] * pow
dims_list = []
for i in range(1, pow+1):
    dims_list.append([math.prod(initial_dims[:i]),] + initial_dims[i:])
meshes = {
    f"mesh_{len(dims)}d": init_device_mesh("cpu", dims) for dims in dims_list
}

tests = {
    "mm": {
        "strategy": mm_strategy,
        "single_dim_strategy": _expand_single_dim_strategy_to_mesh(mm_single_dim_strategy),
        "inputs": (
            (256, 256),
            (256, 256),
        ),
        "dim_maps": (
            # hopefully -1 means replicate in dim_maps?
            [-1, -1],
            [-1, -1],
        ),
        "op_overload": torch.ops.aten.mm.default,
    },
    "add": {
        "strategy": pointwise_strategy,
        "single_dim_strategy": _expand_single_dim_strategy_to_mesh(single_mesh_dim_pointwise_strategy),
        "inputs": (
            (256, 256),
            (256, 256),
        ),
        "dim_maps": (
            # hopefully -1 means replicate in dim_maps?
            [-1, -1],
            [-1, -1],
        ),
        "op_overload": torch.ops.aten.div.Tensor,
    },
}

repeats = 10

def time_fn(mesh, test, key):
    fn = test[key]
    schema = _op_schema(mesh, test["op_overload"], test["inputs"], test["dim_maps"])

    strategies = None

    def f():
        nonlocal strategies
        strategies = fn(schema)

    t = timeit.timeit(f, number=repeats) / repeats
    return t, strategies

times = {}
strategies = {}
for test_name, test in tests.items():
    times[test_name] = {}
    strategies[test_name] = {}
    for mesh_name, mesh in meshes.items():
        orig_t, orig_s = time_fn(mesh, test, "strategy")
        single_t, single_s = time_fn(mesh, test, "single_dim_strategy")
        times[test_name][mesh_name] = {
            "strategy": orig_t,
            "single_dim_strategy": single_t,
        }
        strategies[test_name][mesh_name] = {
            "strategy": orig_s,
            "single_dim_strategy": single_s,
        }
        slowdown = single_t / orig_t
        print(f"{test_name=}, {mesh_name=}, {orig_t=:.2f}s {single_t=:.2f}s {slowdown=:.2f}x {len(orig_s.strategies)=}, {len(single_s.strategies)=}")

torch.distributed.destroy_process_group()
