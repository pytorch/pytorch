# Custom backends

Besides the built-in backends listed under {ref}`Backends <distributed-backends>`,
`torch.distributed` supports backends registered at run time.

## C++ backends

For references on how to develop a third-party backend through C++ Extension,
please refer to [Tutorials - Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html) and
`test/cpp_extensions/cpp_c10d_extension.cpp`. The capability of third-party
backends are decided by their own implementations.

The new backend derives from `c10d::Backend` (`cpp_c10d_extension.cpp` uses the
older `c10d::ProcessGroup` base) and registers the backend
name and the instantiating interface through {func}`torch.distributed.Backend.register_backend`
when imported.

When manually importing this backend and invoking {func}`torch.distributed.init_process_group`
with the corresponding backend name, the `torch.distributed` package runs on
the new backend.

:::{warning}
The support of third-party backend is experimental and subject to change.
:::

## Python backends

A backend can also be written in Python by subclassing `dist.BackendBase`. Each
overridden method is called from C++ in place of the corresponding
`c10d::Backend` virtual, so the backend plugs into `torch.distributed`,
`ProcessGroup` and `DeviceMesh` like a C++ backend.

A minimal single-process backend:

```{testcode}
import torch
import torch.distributed as dist


class MyWork(dist.Work):
    def __init__(self, result):
        super().__init__()
        self._future = torch.futures.Future()
        self._future.set_result(result)

    def wait(self, timeout=None):
        return True

    def get_future(self):
        return self._future

    def is_completed(self):
        return True


class MyBackend(dist.BackendBase):
    def __init__(self, store, rank, world_size, timeout):
        super().__init__(rank, world_size)
        self.num_allreduces = 0

    def getBackendName(self):
        return "my_backend"

    def allreduce(self, tensors, opts):
        # A real backend communicates here; with one rank, SUM is a no-op.
        self.num_allreduces += 1
        return MyWork(tensors)


dist.Backend.register_backend("my_backend", MyBackend, devices=["cpu"])
dist.init_process_group("my_backend", store=dist.HashStore(), rank=0, world_size=1)
t = torch.ones(2)
dist.all_reduce(t)
print(t)
print(dist.get_backend_impl().num_allreduces)
dist.destroy_process_group()
```

```{testoutput}
tensor([1., 1.])
1
```

The creator passed to {func}`~torch.distributed.Backend.register_backend` is
called with `(store, group_rank, group_size, timeout)`, or with
`(dist_backend_opts, pg_options)` when `extended_api=True`. It runs for every
process group created with the backend, including subgroups from
{func}`~torch.distributed.new_group`.

**Entry points.** Instead of calling `register_backend` on import, a package
can declare a `torch.distributed.backends` entry point naming a zero-argument
function that calls `register_backend`. It is loaded the first time the backend
name is used, for example by `init_process_group("my_backend")`. The entry
point name must match the registered backend name.

```toml
[project.entry-points."torch.distributed.backends"]
my_backend = "my_package:register_backend"
```

**Methods.** Override the collectives the backend supports; the rest raise an
error naming the backend. Arguments match the C++ virtuals, with options
objects such as `AllreduceOptions`:

| Method | Called by |
| --- | --- |
| `broadcast(tensors, opts)` | {func}`~torch.distributed.broadcast` |
| `allreduce(tensors, opts)` | {func}`~torch.distributed.all_reduce` |
| `allreduce_sparse(tensors, opts)` | {func}`~torch.distributed.all_reduce` on sparse CUDA tensors |
| `allreduce_coalesced(tensors, opts)` | {func}`~torch.distributed.all_reduce_coalesced` |
| `reduce(tensors, opts)` | {func}`~torch.distributed.reduce` |
| `allgather(output_lists, inputs, opts)` | {func}`~torch.distributed.all_gather` |
| `all_gather_single(output, input, opts)` | {func}`~torch.distributed.all_gather_single` |
| `allgather_coalesced(output_lists, inputs, opts)`, `all_gather_single_coalesced(outputs, inputs, opts)` | coalesced all-gathers |
| `gather(output_lists, inputs, opts)` | {func}`~torch.distributed.gather` |
| `gather_single(output, input, opts)` | {func}`~torch.distributed.gather_single` |
| `scatter(outputs, input_lists, opts)` | {func}`~torch.distributed.scatter` |
| `reduce_scatter(outputs, input_lists, opts)` | {func}`~torch.distributed.reduce_scatter` |
| `reduce_scatter_single(output, input, opts)` | {func}`~torch.distributed.reduce_scatter_single` |
| `reduce_scatter_single_coalesced(outputs, inputs, opts)` | coalesced reduce-scatter |
| `all_to_all_single(output, input, output_splits, input_splits, opts)` | {func}`~torch.distributed.all_to_all_single` |
| `alltoall(outputs, inputs, opts)` | {func}`~torch.distributed.all_to_all` |
| `send(tensors, dst, tag)`, `recv(tensors, src, tag)`, `recv_anysource(tensors, tag)` | point-to-point ops |
| `barrier(opts)`, `monitored_barrier(opts, wait_all_ranks)` | {func}`~torch.distributed.barrier`, {func}`~torch.distributed.monitored_barrier` |

**Work.** Collectives return a `Work`, except `monitored_barrier`;
`recv_anysource` and `end_coalescing` may return `None`. Subclass
`dist.Work` and override `wait(timeout)` and `get_future()`, as with `MyWork`
above. `is_completed()`, `synchronize()`, `source_rank()` (used by
`recv(src=None)`), `result()` and `block_current_stream()` can also be
overridden. Other `Work` methods can't be overridden from Python.

**Capabilities.** Features are opted into with read-only properties, for
example `supports_splitting`, `supports_coalescing` and `supports_shrinking`,
paired with the corresponding methods (`split(store, ranks, opts)`,
`start_coalescing()`/`end_coalescing()`,
`shrink(ranks_to_exclude, shrink_flags, opts_override)`). `split` backs
{func}`~torch.distributed.split_group` and returns the child backend, or `None`
on ranks outside the new group. Define the `options` property if `split_group`
is called without `pg_options`.

**Name.** Override `getBackendName()`, not `name()`; C++ calls the former and
`name()` returns it.

**Custom methods.** Attributes and methods beyond the `Backend` interface are
reached through {func}`~torch.distributed.get_backend_impl`, as with
`num_allreduces` above. See {ref}`backend-specific-collectives`.

**Testing.** `test/distributed/c10d_backend_common.py` defines `python-gloo`, a
Python backend that forwards to Gloo and runs the shared c10d suites. A new
backend can be added to `C10D_BACKENDS` there to run the same tests.

(backend-specific-collectives)=

## Backend-specific collectives

A backend can expose collectives that `torch.distributed` doesn't have as
ordinary methods. {func}`~torch.distributed.get_backend_impl` returns the
backend object of a process group for a device, so callers can use such a
method when the group runs on that backend and fall back to a standard
collective otherwise. This example adds an int8-quantized all-reduce to
`MyBackend`:

```{testcode}
class MyQuantBackend(MyBackend):
    def getBackendName(self):
        return "my_quant_backend"

    def allreduce_int8(self, tensor):
        # A real backend sends int8 values; with one rank, only round-trip.
        scale = tensor.abs().max().clamp(min=1e-12) / 127
        tensor.copy_((tensor / scale).round().clamp(-127, 127) * scale)
        return MyWork([tensor])


def all_reduce_int8(tensor, group=None):
    impl = dist.get_backend_impl(group, tensor.device)
    if not isinstance(impl, MyQuantBackend):
        dist.all_reduce(tensor, group=group)
        return
    impl.allreduce_int8(tensor).wait()


dist.Backend.register_backend("my_quant_backend", MyQuantBackend, devices=["cpu"])
dist.init_process_group("my_quant_backend", store=dist.HashStore(), rank=0, world_size=1)
t = torch.tensor([1.0, -127.0])
all_reduce_int8(t)
print(t)
dist.destroy_process_group()
```

```{testoutput}
tensor([   1., -127.])
```

Backend-specific methods bypass `torch.compile` tracing and other
`torch.distributed` hooks, and must be called in the same order on every rank
of the group, like any collective.
