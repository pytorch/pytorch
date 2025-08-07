import torch.library

from ._distributed_c10d import HAS_DISTRIBUTED


# NB: This is separate from _distributed_c10d because the other module is
# imported too early before operator registrations are ready.  Import this
# from all modules that interact with operators.

if not HAS_DISTRIBUTED:
    # Define missing c10d operators
    with torch.library._scoped_library("c10d", "DEF") as lib_def:
        # Define basic signatures for the operators we need
        op_signatures = {
            "broadcast_": 'broadcast_(Tensor[] tensors, int src, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
            "allreduce_": 'allreduce_(Tensor[] tensors, str? op="sum", str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
            "allgather_": 'allgather_(Tensor[][] output_tensors, Tensor[] input_tensors, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
            "_allgather_base_": '_allgather_base_(Tensor output, Tensor input, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
            "reduce_scatter_": 'reduce_scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, str? op="sum", str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
            "_reduce_scatter_base_": '_reduce_scatter_base_(Tensor output, Tensor input, str? op="sum", str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
            "reduce_": 'reduce_(Tensor[] tensors, int dst, str? op="sum", str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
            "gather_": 'gather_(Tensor[][] output_tensors, Tensor[] input_tensors, int dst, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
            "scatter_": 'scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, int src, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
            "alltoall_": 'alltoall_(Tensor[] output_tensors, Tensor[] input_tensors, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
            "alltoall_base_": 'alltoall_base_(Tensor output, Tensor input, SymInt[]? output_split_sizes=None, SymInt[]? input_split_sizes=None, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
            "barrier": 'barrier(str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
            "monitored_barrier_": 'monitored_barrier_(str? tag="", SymInt[]? ranks=None, int group_size=0, bool wait_all_ranks=False) -> ()',
            "send": 'send(Tensor[] tensors, int dst, str? tag="") -> ()',
            "recv_": 'recv_(Tensor[] tensors, int src, str? tag="") -> ()',
            "recv_any_source_": 'recv_any_source_(Tensor[] tensors, str? tag="") -> ()',
        }

        for signature in op_signatures.values():
            lib_def.define(signature)

    # Register functional collective operators when not available
    def _raise_not_implemented(op_name):
        def _impl(*args, **kwargs):
            raise RuntimeError(
                f"Distributed collective operation '{op_name}' is not available in non-distributed builds"
            )

        return _impl

    functional_lib_def = torch.library.Library("_c10d_functional", "DEF")  # noqa: TOR901
    # Core functional collective operators (from _functional_collectives.py)
    functional_lib_def.define(
        "all_reduce(Tensor input, str reduce_op, str group_name) -> Tensor"
    )
    functional_lib_def.define(
        "all_reduce_(Tensor(a!) input, str reduce_op, str group_name) -> Tensor(a!)"
    )
    functional_lib_def.define(
        "all_reduce_coalesced(Tensor[] inputs, str reduce_op, str group_name) -> Tensor[]"
    )
    functional_lib_def.define(
        "all_reduce_coalesced_(Tensor[](a!) inputs, str reduce_op, str group_name) -> Tensor[](a!)"
    )
    functional_lib_def.define(
        "all_gather_into_tensor_out(Tensor input, int group_size, str group_name, *, Tensor(a!) out) -> Tensor(a!)"
    )
    functional_lib_def.define(
        "all_gather_into_tensor(Tensor input, int group_size, str group_name) -> Tensor"
    )
    functional_lib_def.define(
        "all_gather_into_tensor_coalesced(Tensor[] inputs, int group_size, str group_name) -> Tensor[]"
    )
    functional_lib_def.define(
        "reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, str group_name) -> Tensor"
    )
    functional_lib_def.define(
        "reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduce_op, int group_size, str group_name) -> Tensor[]"
    )
    functional_lib_def.define(
        "all_to_all_single(Tensor input, SymInt[] output_split_sizes, SymInt[] input_split_sizes, str group_name) -> Tensor"
    )
    functional_lib_def.define(
        "broadcast(Tensor input, int src, str group_name) -> Tensor"
    )
    functional_lib_def.define(
        "broadcast_(Tensor(a!) input, int src, str group_name) -> Tensor(a!)"
    )
    functional_lib_def.define("wait_tensor(Tensor tensor) -> Tensor")

    # Register fallback implementations that raise errors
    functional_lib_impl_fallback = torch.library.Library("_c10d_functional", "IMPL")  # noqa: TOR901
    functional_lib_impl_fallback.impl(
        "all_reduce", _raise_not_implemented("all_reduce"), "CompositeExplicitAutograd"
    )
    functional_lib_impl_fallback.impl(
        "all_reduce_",
        _raise_not_implemented("all_reduce_"),
        "CompositeExplicitAutograd",
    )
    functional_lib_impl_fallback.impl(
        "all_reduce_coalesced",
        _raise_not_implemented("all_reduce_coalesced"),
        "CompositeExplicitAutograd",
    )
    functional_lib_impl_fallback.impl(
        "all_reduce_coalesced_",
        _raise_not_implemented("all_reduce_coalesced_"),
        "CompositeExplicitAutograd",
    )
    functional_lib_impl_fallback.impl(
        "all_gather_into_tensor_out",
        _raise_not_implemented("all_gather_into_tensor_out"),
        "CompositeExplicitAutograd",
    )
    functional_lib_impl_fallback.impl(
        "all_gather_into_tensor",
        _raise_not_implemented("all_gather_into_tensor"),
        "CompositeExplicitAutograd",
    )
    functional_lib_impl_fallback.impl(
        "all_gather_into_tensor_coalesced",
        _raise_not_implemented("all_gather_into_tensor_coalesced"),
        "CompositeExplicitAutograd",
    )
    functional_lib_impl_fallback.impl(
        "reduce_scatter_tensor",
        _raise_not_implemented("reduce_scatter_tensor"),
        "CompositeExplicitAutograd",
    )
    functional_lib_impl_fallback.impl(
        "reduce_scatter_tensor_coalesced",
        _raise_not_implemented("reduce_scatter_tensor_coalesced"),
        "CompositeExplicitAutograd",
    )
    functional_lib_impl_fallback.impl(
        "all_to_all_single",
        _raise_not_implemented("all_to_all_single"),
        "CompositeExplicitAutograd",
    )
    functional_lib_impl_fallback.impl(
        "broadcast", _raise_not_implemented("broadcast"), "CompositeExplicitAutograd"
    )
    functional_lib_impl_fallback.impl(
        "broadcast_", _raise_not_implemented("broadcast_"), "CompositeExplicitAutograd"
    )
    functional_lib_impl_fallback.impl(
        "wait_tensor",
        _raise_not_implemented("wait_tensor"),
        "CompositeExplicitAutograd",
    )

    # Register DTensor operators when not available
    dtensor_lib_def = torch.library.Library("_dtensor", "DEF")  # noqa: TOR901
    dtensor_lib_def.define(
        "shard_dim_alltoall(Tensor input, int gather_dim, int shard_dim, str group_name) -> Tensor"
    )

    # Provide CPU and CUDA implementations for DTensor operators
    @torch.library.impl(dtensor_lib_def, "shard_dim_alltoall", "CPU")
    def _shard_dim_alltoall_cpu_impl(input, gather_dim, shard_dim, group_name):
        raise NotImplementedError("not built with distributed")

    @torch.library.impl(dtensor_lib_def, "shard_dim_alltoall", "CUDA")
    def _shard_dim_alltoall_cuda_impl(input, gather_dim, shard_dim, group_name):
        raise NotImplementedError("not built with distributed")

    # Register autograd operators when not available
    autograd_lib_def = torch.library.Library("_c10d_functional_autograd", "DEF")  # noqa: TOR901
    autograd_lib_def.define(
        "all_to_all_single(Tensor input, SymInt[] output_split_sizes, SymInt[] input_split_sizes, str group_name) -> Tensor"
    )
    autograd_lib_def.define(
        "reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, str group_name) -> Tensor"
    )
    autograd_lib_def.define(
        "all_gather_into_tensor(Tensor input, int group_size, str group_name) -> Tensor"
    )
