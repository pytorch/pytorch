# PyTorch DTensor (Prototype Release)

This folder contains the DTensor (a.k.a DistributedTensor) implementation in PyTorch.

## Introduction
We propose distributed tensor primitives to allow easier distributed computation authoring in SPMD(Single Program Multiple Devices) paradigm. The primitives are simple but powerful when used to express tensor distributions with both sharding and replication parallelism strategies. This could empower native Tensor parallelism among other advanced parallelism explorations. For example, to shard a big tensor across devices with 3 lines of code:

```python
# to run this file (i.e. dtensor_example.py):
# torchrun --standalone --nnodes=1 --nproc-per-node=4 dtensor_example.py
import os
import torch
from torch.distributed._tensor import init_device_mesh, Shard, distribute_tensor

# Create a mesh topology with the available devices:
# 1. We can directly create the mesh using elastic launcher, (recommended)
# 2. If using mp.spawn, one need to initialize the world process_group first and set device
#   i.e. torch.distributed.init_process_group(backend="nccl", world_size=world_size)

mesh = init_device_mesh("cuda", (int(os.environ["WORLD_SIZE"]),))
big_tensor = torch.randn(100000, 88)
# Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.
my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
```

## Motivation

Today there are mainly three ways to scale up distributed training: Data Parallel, Tensor Parallel and Pipeline Parallel. Each of them works on a separate dimension where solutions have been built independently (i.e. PyTorch DDP, FSDP, ShardedTensor, PiPPy, etc.). When training really large models, users would like to use these technologies together (i.e. 3-D Parallelism), while the interoperability of the existing solutions are not great and often hard to use (i.e. users might want arbitrary combinations of the data parallel, tensor parallel and pipeline parallel). This is becoming an issue for users and one of the biggest reasons is that there is no common abstraction that build the bridge between different parallelism strategies.

An ideal scenario is that users could build their distributed program just like authoring in a single node/device, without worrying about how to do distributed training in a cluster, and our solutions could help them run distributed training in an efficient manner. For example, researchers just need to build the big transformer model, and PyTorch Distributed automatically figures out how to split the model and run pipeline parallel across different nodes, how to run data parallel and tensor parallel within each node. In order to achieve this, we need some common abstractions to distribute tensor values and distributed computations accordingly.

There're many recent works that working on tensor level parallelism to provide common abstractions, see the `Related Works` in the last section for more details. Inspired by [GSPMD](https://arxiv.org/pdf/2105.04663.pdf), [Oneflow](https://arxiv.org/pdf/2110.15032.pdf) and [TF’s DTensor](https://www.tensorflow.org/guide/dtensor_overview), we introduce PyTorch DTensor as the next generation of ShardedTensor to provide basic abstractions for distributing storage and computation. It serves as one of the basic building blocks for distributed program translations and describes the layout of a distributed training program. With the DTensor abstraction, we can seamlessly build parallelism strategies such as tensor parallelism, DDP and FSDP.

## Value Proposition

PyTorch DTensor primarily:
-   Offers a uniform way to save/load `state_dict` during checkpointing, even when there’re complex tensor storage distribution strategies such as combining tensor parallelism with parameter sharding in FSDP.
-   Enables Tensor Parallelism in eager mode. Compared to ShardedTensor, DistributedTensor allows additional flexibility to mix sharding and replication.
-   Serves as the entry point of an SPMD programming model and the foundational building block for compiler-based distributed training.

## PyTorch DTensor

### DTensor API

We offer both a lower level DistributedTensor API and a module level API to create a `nn.Module` with “distributed” parameters.

#### Basic DTensor API Examples

Here are some basic DTensor API examples that showcase:
1. How to construct a DTensor directly, to represent different types of sharding, replication, sharding + replication strategies.
2. How to create DTensor from a local `torch.Tensor`.
3. How to “reshard” an existing DTensor to a different DTensor with modified placement strategy or world size.

```python
# torchrun --standalone --nnodes=1 --nproc-per-node=4 dtensor_example.py
import torch
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh

# construct a device mesh with available devices (multi-host or single host)
device_mesh = init_device_mesh("cuda", (4,))
# if we want to do row-wise sharding
rowwise_placement=[Shard(0)]
# if we want to do col-wise sharding
colwise_placement=[Shard(1)]

big_tensor = torch.randn(888, 12)
# distributed tensor returned will be sharded across the dimension specified in placements
rowwise_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=rowwise_placement)

# if we want to do replication across a certain device list
replica_placement = [Replicate()]
# distributed tensor will be replicated to all four GPUs.
replica_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=replica_placement)

# if we want to distributed a tensor with both replication and sharding
device_mesh = init_device_mesh("cuda", (2, 2))
# replicate across the first dimension of device mesh, then sharding on the second dimension of device mesh
spec=[Replicate(), Shard(0)]
partial_replica = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=spec)

# create a DistributedTensor that shards on dim 0, from a local torch.Tensor
local_tensor = torch.randn((8, 8), requires_grad=True)
rowwise_tensor = DTensor.from_local(local_tensor, device_mesh, rowwise_placement)

# reshard the current row-wise tensor to a colwise tensor or replicate tensor
colwise_tensor = rowwise_tensor.redistribute(device_mesh, colwise_placement)
replica_tensor = colwise_tensor.redistribute(device_mesh, replica_placement)
```

#### High level User Facing APIs

Users can use DTensor tensor constructors directly to create a distributed tensor (i.e. `distributed.ones/empty`), but for existing modules like `nn.Linear` that are already having `torch.Tensor` as parameters, how to make them distributed parameters? We offer a way to directly distribute a `torch.Tensor` and a module level APIs to directly distribute the module parameters. Below is the high level API we introduce:

```python
def distribute_tensor(tensor: torch.Tensor, device_mesh: DeviceMesh=None, placements: List[Placement]=None):
    '''
    distribute the tensor according to device_mesh and placements, `tensor` could be a "meta" tensor.
    '''

def distribute_module(
    module: nn.Module,
    device_mesh: DeviceMesh=None,
    partition_fn: Callable[[str, nn.Module, DeviceMesh], ...]=None,
    input_fn: Callable[...., None]=None,
    output_fn: Callable[...., None]=None,
):
    '''
    This function converts all module parameters to distributed tensor parameters according to the `partition_fn` specified.
    It could also control the input/output of the module by specifying the `input_fn` and `output_fn`.
    '''
```

#### High level API examples:

```python
import torch.nn as nn
from torch.distributed._tensor import Shard, distribute_tensor, distribute_module, init_device_mesh

class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.fc1(input) + self.fc2(input))

mesh = init_device_mesh("cuda", (4,))

def shard_params(mod_name, mod, mesh):
    col_linear_placement = [Shard(0)]
    # shard fc1 and fc2
    if isinstance(mod, nn.Linear):
        for name, param in mod.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(param, mesh, col_linear_placement)
            )
            mod.register_parameter(name, dist_param)

sharded_module = distribute_module(MyModule(), mesh, partition_fn=shard_params)

```

## Compiler and PyTorch DTensor

DTensor provides efficient solutions for cases like Tensor Parallelism. But when using the DTensor's replication in a data parallel fashion, it might become observably slower compared to our existing solutions like DDP/FSDP. This is mainly because DDP/FSDP have a global view of the entire model architecture, thus could optimize for data parallel specifically, i.e. collective fusion and computation overlap, etc. In contrast, DistributedTensor as a Tensor-like object can only optimize within individual tensor operations.

To improve efficiency of DTensor-based data parallel training, we are exploring a compiler-based solution on top of DTensor, which can extract graph information from user programs to expose more performance optimization opportunities.

## Related Works

This work is mainly inspired by [GSPMD](https://arxiv.org/pdf/2105.04663.pdf), [Oneflow](https://arxiv.org/pdf/2110.15032.pdf) and [TF’s DTensor](https://www.tensorflow.org/guide/dtensor_overview). All of these three works use a single “distributed tensor” concept for both replication and sharding, and the solutions could enable users to build up their distributed training program in a uniform SPMD programming model. Specifically:

GSPMD:
-   GSPMD is now the fundamental component of JAX/TensorFlow distributed training and enables various optimizations with the XLA compiler to allow users to train their models efficiently in a large scale setting.
-   Fundamentally, GSPMD have three types of sharding strategies within a tensor: “tiled”, “replicated”, “partially tiled” to represent sharding and replication.
-   At the core of GSPMD Partitioner, it utilizes the XLA compiler to do advanced optimizations, i.e. sharding propagation and compiler based fusion.
-   XLA mark_sharding API: PyTorch XLA’s [mark_sharding](https://github.com/pytorch/xla/pull/3476) API uses [XLAShardedTensor](https://github.com/pytorch/xla/issues/3871) abstraction (i.e. sharding specs) in PyTorch/XLA. Under the hood XLAShardedTensor is utilizing the GSPMD partitioner to enable SPMD style training on TPU.

OneFlow GlobalTensor:

-  OneFlow is building up their own solution of the “GlobalTensor” concept, which is a variant form of GSPMD sharding, allowing users to explore different parallel strategies with GlobalTensor.
-  OneFlow also has three types of tensor, but they are slightly different from GSPMD: “split”, “broadcast”, and “partial sum”. They don’t use partially tiled and instead have a concept of partial sum to partition the values.

TensorFlow DTensor:
-   [DTensor Concepts](https://www.tensorflow.org/guide/dtensor_overview) is an extension of TensorFlow synchronous distributed training. its sharding API, supported features and its compilation passes with MLIR.
-   DTensor also allows sharding and replication on an n-d mesh like device network.
-   DTensor implements MLIR passes to do propagation and operator implementations.

There are also several cutting edge research fields that embeds tensor sharding as part of the system, i.e. [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf) for tensor parallelism on Transformer based models. [DeepSpeed](https://github.com/microsoft/DeepSpeed) for training large scale models with different optimization techniques on top of tensor sharding.

### Additional context

RFC: https://github.com/pytorch/pytorch/issues/88838

We are gathering early feedbacks about this proposal. We have also posted this [RFC](https://dev-discuss.pytorch.org/t/rfc-pytorch-distributedtensor/740) to the dev-discuss forum, please feel free to comment directly in the above issue or in the forum post. To see a complete design doc with additional details about DTensor, please refer to this [doc](https://docs.google.com/document/d/1nFeJ8NSFNhNlCkNgWK31ZGRqm1L9rd0i_XN_RprphaI/edit#heading=h.6sovjqv9jiqn)
