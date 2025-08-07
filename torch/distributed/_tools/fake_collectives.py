import random
from typing import Any, Callable

import torch

# Import centralized distributed components
from torch.distributed._distributed_c10d import (
    _resolve_process_group,
    FakeWork,
    HAS_DISTRIBUTED,
    ProcessGroup,
    Work,
)


# In distributed builds, assume operators always exist
OPERATORS_AVAILABLE = HAS_DISTRIBUTED


from torch.utils._pytree import tree_map_only


torch.distributed.batch_isend_irecv

c10d = torch.ops.c10d
_c10d_functional = torch.ops._c10d_functional
_c10d_functional_autograd = torch.ops._c10d_functional_autograd
_dtensor = torch.ops._dtensor
used_ids: set[int] = set()


def generate_unique_id() -> int:
    while True:
        new_id = random.randint(1, 10**9)
        if new_id not in used_ids:
            used_ids.add(new_id)
            return new_id


# Function to create and return FakeWork object
def create_fakework(args, return_first_arg=True):  # type: ignore[no-untyped-def]
    work = FakeWork()
    work.seq_id = generate_unique_id()
    fakework_script_obj = work.boxed()
    return (args[0], fakework_script_obj) if return_first_arg else fakework_script_obj


# Dictionary mapping collective operations to their meta functions
# All 20 ops from torch.csrc.distributed.c10d.Ops.cpp are included
# _DEPRECATED_META_FUNCTIONS = {
#     "allreduce_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
#     "allgather_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
#     "allgather_into_tensor_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
#     "reduce_scatter_tensor_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
# }
_META_FUNCTIONS = {
    "broadcast_": lambda *args: create_fakework(args),
    "allreduce_": lambda *args: create_fakework(args),
    "allgather_": lambda *args: create_fakework(args),
    "_allgather_base_": lambda *args: create_fakework(args),
    "reduce_scatter_": lambda *args: create_fakework(args),
    "_reduce_scatter_base_": lambda *args: create_fakework(args),
    "reduce_": lambda *args: create_fakework(args, return_first_arg=False),
    "gather_": lambda *args: create_fakework(args, return_first_arg=False),
    "scatter_": lambda *args: create_fakework(args),
    "alltoall_": lambda *args: create_fakework(args),
    "alltoall_base_": lambda *args: create_fakework(args, return_first_arg=False),
    "barrier": lambda *args: create_fakework(args, return_first_arg=False),
    "monitored_barrier_": lambda *args: None,
    "send": lambda *args: create_fakework(args, return_first_arg=False),
    "recv_": lambda *args: create_fakework(args, return_first_arg=False),
    "recv_any_source_": lambda *args: create_fakework(args, return_first_arg=False),
}


# Register operators when C++ implementations are not available
def _create_fake_op_impl(op_name: str, default_impl: Callable) -> Callable:
    """Create a fake implementation for an operator that raises a descriptive error."""

    def _impl(*args: Any, **kwargs: Any) -> Any:
        try:
            return default_impl(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Distributed collective operation '{op_name}' is not available in non-distributed builds"
            ) from e

    return _impl


# First, try to define the missing c10d operators
try:
    # Check if operators exist, if not define them
    torch.ops.c10d.broadcast_
except (AttributeError, RuntimeError):
    # Define missing c10d operators
    try:
        with torch.library._scoped_library("c10d", "DEF") as lib_def:
            # Define basic signatures for the operators we need
            op_signatures = {
                "broadcast_": 'broadcast_(Tensor[] tensors, int src, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
                "allreduce_": 'allreduce_(Tensor[] tensors, str? op="sum", str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',  # noqa: B950
                "allgather_": 'allgather_(Tensor[][] output_tensors, Tensor[] input_tensors, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',  # noqa: B950
                "_allgather_base_": '_allgather_base_(Tensor output, Tensor input, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',  # noqa: B950
                "reduce_scatter_": 'reduce_scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, str? op="sum", str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',  # noqa: B950
                "_reduce_scatter_base_": '_reduce_scatter_base_(Tensor output, Tensor input, str? op="sum", str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',  # noqa: B950
                "reduce_": 'reduce_(Tensor[] tensors, int dst, str? op="sum", str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',  # noqa: B950
                "gather_": 'gather_(Tensor[][] output_tensors, Tensor[] input_tensors, int dst, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',  # noqa: B950
                "scatter_": 'scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, int src, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',  # noqa: B950
                "alltoall_": 'alltoall_(Tensor[] output_tensors, Tensor[] input_tensors, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',  # noqa: B950
                "alltoall_base_": 'alltoall_base_(Tensor output, Tensor input, SymInt[]? output_split_sizes=None, SymInt[]? input_split_sizes=None, str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',  # noqa: B950
                "barrier": 'barrier(str? tag="", SymInt[]? ranks=None, int group_size=0) -> ()',
                "monitored_barrier_": 'monitored_barrier_(str? tag="", SymInt[]? ranks=None, int group_size=0, bool wait_all_ranks=False) -> ()',  # noqa: B950
                "send": 'send(Tensor[] tensors, int dst, str? tag="") -> ()',
                "recv_": 'recv_(Tensor[] tensors, int src, str? tag="") -> ()',
                "recv_any_source_": 'recv_any_source_(Tensor[] tensors, str? tag="") -> ()',
            }

            for signature in op_signatures.values():
                try:
                    lib_def.define(signature)
                except Exception:
                    # Skip if already defined or can't define
                    pass
    except Exception:
        # Failed to create library, continue
        pass

# Register functional collective operators when not available
try:
    # Check if _c10d_functional operators are available
    torch.ops._c10d_functional.wait_tensor
except (AttributeError, RuntimeError):
    # Register the operators from Python when C++ is not built
    try:
        functional_lib_def = torch.library.Library("_c10d_functional", "DEF")  # noqa: TOR901
        # Core functional collective operators (same as in _functional_collectives.py)
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
    except Exception:
        pass

# Register DTensor operators when not available
try:
    torch.ops._dtensor.shard_dim_alltoall
except (AttributeError, RuntimeError):
    try:
        dtensor_lib_def = torch.library.Library("_dtensor", "DEF")  # noqa: TOR901
        dtensor_lib_def.define(
            "shard_dim_alltoall(Tensor input, int gather_dim, int shard_dim, str group_name) -> Tensor"
        )
    except Exception:
        pass

# Register autograd operators when not available
try:
    torch.ops._c10d_functional_autograd.all_to_all_single
except (AttributeError, RuntimeError):
    try:
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
    except Exception:
        pass

try:
    lib_impl = torch.library.Library("c10d", "IMPL")  # noqa: TOR901
    for op, meta_func in _META_FUNCTIONS.items():
        lib_impl.impl(op, meta_func, "Meta")
except (RuntimeError, AttributeError):
    # Skip library registration in non-distributed builds
    pass

# List of collective operation functions including functional collectives
# Note: The following collectives might be deprecated soon hence not adding them
# depcreated_non_functional_collectives = [
#     c10d.allreduce_coalesced_.default,
#     c10d.reduce_scatter_tensor_coalesced_.default,
#     c10d.allgather_into_tensor_coalesced_.default,
#     c10d.allgather_coalesced_.default,
# ]
non_functional_collectives: set[torch._ops.OpOverload] = {
    c10d.broadcast_.default,
    c10d.allreduce_.default,
    c10d.reduce_.default,
    c10d.send.default,
    c10d.recv_.default,
    c10d.recv_any_source_.default,
    c10d.allgather_.default,
    c10d.reduce_scatter_.default,
    c10d._reduce_scatter_base_.default,
    c10d._allgather_base_.default,
    c10d.gather_.default,
    c10d.scatter_.default,
    c10d.alltoall_.default,
    c10d.alltoall_base_.default,
    c10d.barrier.default,
    c10d.monitored_barrier_.default,
}
try:
    functional_collectives: set[torch._ops.OpOverload] = {
        _c10d_functional.broadcast.default,
        _c10d_functional.all_reduce.default,
        _c10d_functional.all_gather_into_tensor.default,
        _c10d_functional.reduce_scatter_tensor.default,
        _c10d_functional.all_to_all_single.default,
        _c10d_functional_autograd.all_to_all_single.default,
        _c10d_functional.wait_tensor.default,
        _c10d_functional.all_reduce_.default,
        _c10d_functional.all_reduce_coalesced.default,
        _c10d_functional.all_reduce_coalesced_.default,
        _c10d_functional.all_gather_into_tensor_out.default,
        _c10d_functional.all_gather_into_tensor_coalesced.default,
        _c10d_functional_autograd.all_gather_into_tensor.default,
        _c10d_functional.reduce_scatter_tensor_coalesced.default,
        _c10d_functional_autograd.reduce_scatter_tensor.default,
        _c10d_functional.broadcast_.default,
        _dtensor.shard_dim_alltoall.default,
    }
except (AttributeError, RuntimeError):
    # Fallback for non-distributed builds
    functional_collectives: set[torch._ops.OpOverload] = set()  # type: ignore[no-redef]

try:
    sync_ops: set[torch._ops.OpOverload] = {
        c10d.barrier.default,
        c10d.monitored_barrier_.default,
        _c10d_functional.wait_tensor.default,
    }
except (AttributeError, RuntimeError):
    # Fallback for non-distributed builds
    sync_ops: set[torch._ops.OpOverload] = set()  # type: ignore[no-redef]

collective_ops = set.union(functional_collectives, non_functional_collectives)


class CollectiveOp:
    # Static sets for performance optimization
    PG_ARG_1 = {
        c10d.broadcast_.default,
        c10d.allreduce_.default,
        c10d.reduce_.default,
        c10d.send.default,
        c10d.recv_.default,
        c10d.recv_any_source_.default,
        c10d.barrier.default,
        # c10d.allreduce_coalesced_.default
    }

    PG_ARG_2 = {
        c10d.allgather_.default,
        c10d._allgather_base_.default,
        c10d.reduce_scatter_.default,
        c10d._reduce_scatter_base_.default,
        c10d.gather_.default,
        c10d.scatter_.default,
        c10d.alltoall_.default,
        c10d.alltoall_base_.default,
        # c10d.allgather_coalesced_.default,
        # c10d.allgather_into_tensor_coalesced_.default
        # c10d.reduce_scatter_tensor_coalesced_.default
    }

    if OPERATORS_AVAILABLE:
        PG_ARG_3 = {
            _c10d_functional.broadcast.default,
            _c10d_functional.broadcast_.default,
            _c10d_functional.all_reduce.default,
            _c10d_functional.all_reduce_.default,
            _c10d_functional.all_reduce_coalesced.default,
            _c10d_functional.all_reduce_coalesced_.default,
            _c10d_functional.all_gather_into_tensor.default,
            _c10d_functional.all_gather_into_tensor_out.default,
            _c10d_functional_autograd.all_gather_into_tensor.default,
            _c10d_functional.all_gather_into_tensor_coalesced.default,
        }

    PG_ARG_4 = {
        _c10d_functional.reduce_scatter_tensor.default,
        _c10d_functional.reduce_scatter_tensor_coalesced.default,
        _c10d_functional_autograd.reduce_scatter_tensor.default,
        _c10d_functional.all_to_all_single.default,
        _c10d_functional_autograd.all_to_all_single.default,
        _dtensor.shard_dim_alltoall.default,
    }

    WK_ARG_1 = {
        c10d.broadcast_.default,
        c10d.allreduce_.default,
        c10d.allgather_.default,
        c10d.reduce_scatter_.default,
        c10d._reduce_scatter_base_.default,
        c10d._allgather_base_.default,
        c10d.scatter_.default,
        c10d.alltoall_.default,
    }

    WK = {
        c10d.send.default,
        c10d.recv_.default,
        c10d.recv_any_source_.default,
        c10d.reduce_.default,
        c10d.gather_.default,
        c10d.alltoall_base_.default,
        c10d.barrier.default,
    }

    COMM_TENSOR_ARG_0 = {
        c10d.allreduce_.default,
        c10d.send.default,
        c10d.recv_.default,
        c10d.recv_any_source_.default,
        c10d.allgather_.default,
        c10d.gather_.default,
        c10d.reduce_.default,
        c10d.broadcast_.default,
        _c10d_functional.all_reduce_coalesced.default,
        _c10d_functional.all_reduce_coalesced_.default,
        # c10d.allreduce_coalesced_.default
        # c10d.allgather_coalesced_.default
        # c10d.allgather_into_tensor_coalesced_.default,
    }

    COMM_TENSOR_ARG_1 = {
        c10d.reduce_scatter_.default,
        c10d.scatter_.default,
        # c10d.reduce_scatter_tensor_coalesced_.default,
    }

    COMM_TENSOR_ARG_RES = {
        _c10d_functional.all_gather_into_tensor.default,
        _c10d_functional_autograd.all_gather_into_tensor.default,
    }

    COMM_TENSOR_SINGLE_UNTYPED_STORAGE = {
        c10d._allgather_base_.default,
        _c10d_functional.broadcast.default,
        _c10d_functional.broadcast_.default,
        _c10d_functional.all_reduce.default,
        _c10d_functional.all_reduce_.default,
        _c10d_functional.reduce_scatter_tensor.default,
        _c10d_functional_autograd.reduce_scatter_tensor.default,
    }

    COMM_TENSOR_ARG_0_AND_RES = {
        _c10d_functional.all_to_all_single.default,
        _c10d_functional_autograd.all_to_all_single.default,
        _dtensor.shard_dim_alltoall.default,
    }

    COMM_TENSOR_RES_SUM = {
        _c10d_functional.all_gather_into_tensor_coalesced.default,
        _c10d_functional.reduce_scatter_tensor_coalesced.default,
    }

    @staticmethod
    def sum_tensors(arg: Any) -> int:
        """Calculate total memory consumed by the tensors in the argument."""
        total_memory = 0

        def sum_bytes(t: torch.Tensor) -> None:
            nonlocal total_memory
            total_memory += t.untyped_storage().nbytes()

        tree_map_only(torch.Tensor, sum_bytes, arg)
        return total_memory

    @staticmethod
    def get_process_group(func, args) -> ProcessGroup:  # type: ignore[no-untyped-def]
        """Retrieve the process group for collective operations, except `wait_tensor`."""
        if func in CollectiveOp.PG_ARG_1:
            return ProcessGroup.unbox(args[1])
        if func in CollectiveOp.PG_ARG_2:
            return ProcessGroup.unbox(args[2])
        if func in CollectiveOp.PG_ARG_3:
            return _resolve_process_group(args[2])
        if func in CollectiveOp.PG_ARG_4:
            return _resolve_process_group(args[3])
        raise TypeError(f"Func {func} not found in {collective_ops}")

    @staticmethod
    def get_comm_tensor_size(func, res, args, kwargs) -> int:  # type: ignore[no-untyped-def]
        """Compute the communication tensor size, except for `wait_tensor`, `barrier`, and `monitored_barrier`."""
        if func in CollectiveOp.COMM_TENSOR_ARG_0:
            return CollectiveOp.sum_tensors(args[0])
        if func in CollectiveOp.COMM_TENSOR_ARG_1:
            return CollectiveOp.sum_tensors(args[1])
        if func in CollectiveOp.COMM_TENSOR_ARG_RES:
            return res.untyped_storage().nbytes()
        if func in CollectiveOp.COMM_TENSOR_SINGLE_UNTYPED_STORAGE:
            return args[0].untyped_storage().nbytes()
        if func == c10d._reduce_scatter_base_.default:
            return args[1].untyped_storage().nbytes()
        if func == c10d.alltoall_.default:
            # TODO(@sanketpurandare) - Confirm size computation
            return max(
                CollectiveOp.sum_tensors(args[0]), CollectiveOp.sum_tensors(args[1])
            )
        if func == c10d.alltoall_base_.default:
            # TODO(@sanketpurandare) - Confirm size computation
            return max(
                args[0].untyped_storage().nbytes(), args[1].untyped_storage().nbytes()
            )
        if func == _c10d_functional.all_gather_into_tensor_out.default:
            return args[-1].untyped_storage().nbytes()
        if func in CollectiveOp.COMM_TENSOR_RES_SUM:
            return CollectiveOp.sum_tensors(res)
        if func in CollectiveOp.COMM_TENSOR_ARG_0_AND_RES:
            # TODO(@sanketpurandare) - Confirm size computation
            return args[0].untyped_storage().nbytes() + res.untyped_storage().nbytes()
        raise TypeError(f"Unknown function: {func} in {collective_ops}")

    @staticmethod
    def get_work(func, res) -> Work:  # type: ignore[no-untyped-def]
        if func in CollectiveOp.WK:
            return FakeWork.unbox(res)
        elif func in CollectiveOp.WK_ARG_1:
            return FakeWork.unbox(res[1])
        raise TypeError(f"Func {func} not found in {collective_ops}")
