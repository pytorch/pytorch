
:::{currentmodule} torch.distributed.tensor
:::


# torch.distributed.tensor

:::{note}
`torch.distributed.tensor` is currently in alpha state and under
development, we are committing backward compatibility for the most APIs listed
in the doc, but there might be API changes if necessary.
:::

## PyTorch DTensor (Distributed Tensor)

PyTorch DTensor offers simple and flexible tensor sharding primitives that transparently handles distributed
logic, including sharded storage, operator computation and collective communications across devices/hosts.
`DTensor` could be used to build different parallelism solutions and support sharded state_dict representation
when working with multi-dimensional sharding.

Please see examples from the PyTorch native parallelism solutions that are built on top of `DTensor`:

- [Tensor Parallel](https://pytorch.org/docs/main/distributed.tensor.parallel.html)
- [FSDP2](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)

```{eval-rst}
.. automodule:: torch.distributed.tensor
```

{class}`DTensor` follows the SPMD (single program, multiple data) programming model to empower users to
write distributed program as if it's a **single-device program with the same convergence property**. It
provides a uniform tensor sharding layout (DTensor Layout) through specifying the {class}`DeviceMesh`
and {class}`Placement`:

- {class}`DeviceMesh` represents the device topology and the communicators of the cluster using
  an n-dimensional array.
- {class}`Placement` describes the sharding layout of the logical tensor on the {class}`DeviceMesh`.
  DTensor supports three types of placements: {class}`Shard`, {class}`Replicate` and {class}`Partial`.

### DTensor Class APIs

```{eval-rst}
.. currentmodule:: torch.distributed.tensor
```

{class}`DTensor` is a `torch.Tensor` subclass. This means once a {class}`DTensor` is created, it could be
used in very similar way to `torch.Tensor`, including running different types of PyTorch operators as if
running them in a single device, allowing proper distributed computation for PyTorch operators.

In addition to existing `torch.Tensor` methods, it also offers a set of additional methods to interact with
`torch.Tensor`, `redistribute` the DTensor Layout to a new DTensor, get the full tensor content
on all devices, etc.

```{eval-rst}
.. autoclass:: DTensor
    :members: from_local, to_local, full_tensor, redistribute, device_mesh, placements
    :member-order: groupwise
    :special-members: __create_chunk_list__

```

### DeviceMesh as the distributed communicator

```{eval-rst}
.. currentmodule:: torch.distributed.device_mesh
```

{class}`DeviceMesh` was built from DTensor as the abstraction to describe cluster's device topology and represent
multi-dimensional communicators (on top of `ProcessGroup`). To see the details of how to create/use a DeviceMesh,
please refer to the [DeviceMesh recipe](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html).

### DTensor Placement Types

```{eval-rst}
.. automodule:: torch.distributed.tensor.placement_types
```

```{eval-rst}
.. currentmodule:: torch.distributed.tensor.placement_types
```

DTensor supports the following types of {class}`Placement` on each {class}`DeviceMesh` dimension:

```{eval-rst}
.. autoclass:: Shard
  :members:
  :undoc-members:
```

```{eval-rst}
.. autoclass:: Replicate
  :members:
  :undoc-members:
```

```{eval-rst}
.. autoclass:: Partial
  :members:
  :undoc-members:
```

```{eval-rst}
.. autoclass:: MaskPartial
  :members:
  :undoc-members:
```

```{eval-rst}
.. autoclass:: Placement
  :members:
  :undoc-members:
```

(create_dtensor)=

## Different ways to create a DTensor

```{eval-rst}
.. currentmodule:: torch.distributed.tensor
```

There're three ways to construct a {class}`DTensor`:
: - {meth}`distribute_tensor` creates a {class}`DTensor` from a logical or "global" `torch.Tensor` on
    each rank. This could be used to shard the leaf `torch.Tensor` s (i.e. model parameters/buffers
    and inputs).
  - {meth}`DTensor.from_local` creates a {class}`DTensor` from a local `torch.Tensor` on each rank, which can
    be used to create {class}`DTensor` from a non-leaf `torch.Tensor` s (i.e. intermediate activation
    tensors during forward/backward).
  - DTensor provides dedicated tensor factory functions (e.g. {meth}`empty`, {meth}`ones`, {meth}`randn`, etc.)
    to allow different {class}`DTensor` creations by directly specifying the {class}`DeviceMesh` and
    {class}`Placement`. Compare to {meth}`distribute_tensor`, this could directly materializing the sharded memory
    on device, instead of performing sharding after initializing the logical Tensor memory.

### Create DTensor from a logical torch.Tensor

The SPMD (single program, multiple data) programming model in `torch.distributed` launches multiple processes
(i.e. via `torchrun`) to execute the same program, this means that the model inside the program would be
initialized on different processes first (i.e. the model might be initialized on CPU, or meta device, or directly
on GPU if enough memory).

`DTensor` offers a {meth}`distribute_tensor` API that could shard the model weights or Tensors to `DTensor` s,
where it would create a DTensor from the "logical" Tensor on each process. This would empower the created
`DTensor` s to comply with the single device semantic, which is critical for **numerical correctness**.

```{eval-rst}
.. autofunction::  distribute_tensor
```

Along with {meth}`distribute_tensor`, DTensor also offers a {meth}`distribute_module` API to allow easier
sharding on the {class}`nn.Module` level

```{eval-rst}
.. autofunction::  distribute_module

```

### DTensor Factory Functions

DTensor also provides dedicated tensor factory functions to allow creating {class}`DTensor` directly
using torch.Tensor like factory function APIs (i.e. torch.ones, torch.empty, etc), by additionally
specifying the {class}`DeviceMesh` and {class}`Placement` for the {class}`DTensor` created:

```{eval-rst}
.. autofunction:: zeros
```

```{eval-rst}
.. autofunction:: ones
```

```{eval-rst}
.. autofunction:: empty
```

```{eval-rst}
.. autofunction:: full
```

```{eval-rst}
.. autofunction:: rand
```

```{eval-rst}
.. autofunction:: randn

```

### Random Operations

DTensor provides distributed RNG functionality to ensure that random operations on sharded tensors get unique values, and random operations on replicated tensors get the same values. This system requires that all participating
ranks (e.g. SPMD ranks) start out using the same generator state before each dtensor random operation is performed,
and if this is true, it ensures they all end up at the same state after each dtensor random operation completes. There is no communication performed during random operations to synchronize RNG states.

Operators that accept a `generator` kwarg will utilize the user-passed generator, if passed, or the default generator for the device otherwise. Whichever generator is used, it will be advanced after the DTensor operation.  It is valid to use the same generator for both DTensor and non-DTensor operations, but care must be taken to ensure the non-DTensor operations advance the generator state equally on all ranks if so.

When using DTensor together with Pipeline Parallelism, ranks for each pipeline stage should use a distinct seed, and ranks within a pipeline stage should use the same seed.

DTensor's RNG infra is based on the philox based RNG algorithm, and supports any philox based backend (cuda, and other cuda-like devices), but unfortunately does not yet support the CPU backend.

## Debugging

```{eval-rst}
.. automodule:: torch.distributed.tensor.debug
```

```{eval-rst}
.. currentmodule:: torch.distributed.tensor.debug
```

### Logging

When launching the program, you can turn on additional logging using the `TORCH_LOGS` environment variable from
[torch._logging](https://pytorch.org/docs/main/logging.html#module-torch._logging) :

- `TORCH_LOGS=+dtensor` will display `logging.DEBUG` messages and all levels above it.
- `TORCH_LOGS=dtensor` will display `logging.INFO` messages and above.
- `TORCH_LOGS=-dtensor` will display `logging.WARNING` messages and above.

### Debugging Tools

To debug the program that applied DTensor, and understand more details about what collectives happened under the
hood, DTensor provides a {class}`CommDebugMode`:

```{eval-rst}
.. autoclass:: CommDebugMode
    :members:
    :undoc-members:
```

To visualize the sharding of a DTensor that have less than 3 dimensions, DTensor provides {meth}`visualize_sharding`:

```{eval-rst}
.. autofunction:: visualize_sharding

```

## Experimental Features

`DTensor` also provides a set of experimental features. These features are either in prototyping stage, or the basic
functionality is done and but looking for user feedbacks. Please submit a issue to PyTorch if you have feedbacks to
these features.

```{eval-rst}
.. automodule:: torch.distributed.tensor.experimental
```

```{eval-rst}
.. currentmodule:: torch.distributed.tensor.experimental
```

```{eval-rst}
.. autofunction:: context_parallel
```

```{eval-rst}
.. autofunction:: local_map
```

```{eval-rst}
.. autofunction:: register_sharding

```

% modules that are missing docs, add the doc later when necessary

```{eval-rst}
.. py:module:: torch.distributed.tensor.device_mesh
```

## Mixed Tensor and DTensor operations

So you got the following error message.
```
got mixed torch.Tensor and DTensor, need to convert all
torch.Tensor to DTensor before calling distributed operators!
```

There are two cases.

### Case 1: this is user error

The most common way to run into this error is to create a regular Tensor
(using a factory function) and then perform a Tensor-DTensor operation,
like the following:

```
tensor = torch.arange(10)
return tensor + dtensor
```

We disallow mixed Tensor-DTensor operations: if the input to any operations
(e.g. torch.add) is a DTensor, then all Tensor inputs must be DTensors.
This is because the semantics are ambiguous. We don't know if `tensor` is
the same across ranks or if it is different so we ask that the user
figure out how to construct a DTensor with accurate placements from `tensor`.

If each rank does have the same `tensor`, then please construct a replicated
DTensor:

```
tensor = torch.arange(10)
tensor = DTensor.from_local(tensor, placements=(Replicate(),))
return tensor + dtensor
```

If you wanted to create a DTensor with shards, below is how to do it.
Semantically this means that your Tensor data is split between the shards
and that operations act on the "full stacked data".

```
tensor = torch.full([], RANK)
tensor = DTensor.from_local(tensor, placements=(Shard(0),))
return tensor + dtensor
```

There are other things you may wish to do with your tensor beyond
these situations (these are not the only two options!).

## Case 2: the error came from PyTorch framework code

Sometimes the problem is that PyTorch framework code attempts to perform mixed
Tensor-DTensor operations. These are bugs in PyTorch, please file an issue
so that we can fix them.

On the user side, the only thing you can do is to avoid using the operation
that caused the issue and file a bug report.

For PyTorch Developers: one approach of fixing this is to rewrite PyTorch
framework code to avoid mixed Tensor-DTensor code (like in the previous section).

For PyTorch Developers: the second approach is to turn on DTensor implicit
replication inside the right places in PyTorch framework code.
When on, any mixed Tensor-DTensor operations will assume that the
non-DTensors can be replicated. Please be careful when using this as it
can lead to silent incorrectness.

- [Turning on implicit replication in Python](https://github.com/pytorch/pytorch/blob/d8e6b2fddc54c748d976e8f0ebe4b63ebe36d85b/torch/distributed/tensor/experimental/__init__.py#L15)
- [Turning on implicit replication in C++](https://github.com/pytorch/pytorch/blob/7a0f93344e2c851b9bcf2b9c3225a323d48fde26/aten/src/ATen/DTensorState.h#L10)
