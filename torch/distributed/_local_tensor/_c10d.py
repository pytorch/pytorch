import functools
import math
import operator
from collections.abc import Callable, Sequence
from datetime import timedelta

import torch
from torch._C import ScriptObject
from torch._C._distributed_c10d import FakeWork, PythonCallbackWork
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.distributed_c10d import (
    _check_op,
    _get_default_group,
    _resolve_process_group,
    ProcessGroup,
    ReduceOp,
    Work,
)


# NOTE: Most of the c10d collectives often take a Tensor[] (or Tensor[][])
# when you would expect Tensor (or Tensor[]).  In fact, there will only ever
# be one Tensor in this case; the old signature was to support dispatching a
# collective on multiple devices (ala DataParallel) but we don't support that
# API anymore.  Note that we are not 100% consistent about this; some more
# modern collectives like _allgather_base_ got rid of the unnecessary list.
# When in doubt, consult the code that dispatches to the collective on the PG
# in distributed_c10d.py e.g., work = group.allgather([tensor_list], [tensor],
# opts) indicates its always a list.


def _gcd_list(numbers: Sequence[int]) -> int:
    return 0 if not numbers else functools.reduce(math.gcd, numbers)


def _indices_to_layout(indices: list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    # Base case: A single index represents a point, not a dimension.
    if len(indices) <= 1:
        return (), ()

    # The smallest stride is likely the GCD of the differences between consecutive indices.
    # For a sorted, unique list, all differences will be positive.
    diffs = [indices[i] - indices[i - 1] for i in range(1, len(indices))]
    last_stride = _gcd_list(diffs)

    assert last_stride != 0, (
        # This case should not be reached if indices are unique and sorted.
        "Cannot determine stride; indices may not be unique."
    )

    # Identify the starting index of each "row" in the last dimension.
    # An index starts a new row if the preceding index (index - stride) is not present.
    indices_set = set(indices)
    higher_dim_indices = [indices[0]]
    for index in indices[1:]:
        if (index - last_stride) not in indices_set:
            higher_dim_indices.append(index)

    # From the number of rows, we can deduce the shape of the last dimension.
    assert len(indices) % len(higher_dim_indices) == 0, (
        "Indices do not form a regular grid. "
        f"Found {len(higher_dim_indices)} subgroups for {len(indices)} total elements."
    )
    last_shape = len(indices) // len(higher_dim_indices)

    # Recurse on the higher-dimensional indices (the start of each row).
    higher_shapes, higher_strides = _indices_to_layout(higher_dim_indices)

    # Combine the results from the recursion with the current dimension's results.
    final_shapes = higher_shapes + (last_shape,)
    final_strides = higher_strides + (last_stride,)

    return final_shapes, final_strides


def _prepare_collective_groups(
    process_group_so: ScriptObject | ProcessGroup,
) -> tuple[list[int], list[int], int]:
    process_group = (
        ProcessGroup.unbox(process_group_so)
        if isinstance(process_group_so, ScriptObject)
        else process_group_so
    )

    ranks = torch.distributed.get_process_group_ranks(process_group)
    assert ranks
    # TODO: We can handle permutations but the layout inference algorithm will
    # lose the permutation so we will have to reapply it
    assert ranks == sorted(ranks), ranks
    offset = ranks[0]
    ranks = [r - offset for r in ranks]

    shape, strides = _indices_to_layout(ranks)
    layout = _MeshLayout(shape, strides)

    global_pg = _get_default_group()
    group_offsets = layout.complement(global_pg.size()).all_ranks_from_zero()

    return ranks, group_offsets, offset


# NB: There are two flavors of the collectives: regular and functional. Regular collectives
# allocate outputs to write the result to, accept process group and support async ops (return
# work object). Functional collectives expect the implementation to allocate outputs, accept
# process group name that must be resolved and do not support async ops (return output).
def _local_functional_all_gather_into_tensor(
    tensor: torch.Tensor, group_size: int, group_name: str
) -> torch.Tensor:
    # "all_gather_into_tensor(Tensor input, int group_size, str group_name) -> Tensor"
    from . import LocalTensor

    ranks, group_offsets, offset = _prepare_collective_groups(
        _resolve_process_group(group_name)
    )

    assert isinstance(tensor, LocalTensor), "Input tensor must be a LocalTensor"
    output_local_tensors: dict[int, torch.Tensor] = {}

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        group_tensors = []
        for rank in group_ranks:
            group_tensors.append(tensor._local_tensors[rank])

        gathered_tensor = torch.cat(group_tensors, dim=0)

        for rank in group_ranks:
            output_local_tensors[rank] = gathered_tensor.clone()

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    output = LocalTensor(output_local_tensors)

    return output


def _local_functional_reduce_scatter_tensor(
    tensor: torch.Tensor, reduce_op: str, group_size: int, group_name: str
) -> torch.Tensor:
    #  "reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, str group_name) -> Tensor"
    from . import _zero_sized_like, LocalTensor

    ranks, group_offsets, offset = _prepare_collective_groups(
        _resolve_process_group(group_name)
    )

    assert isinstance(tensor, LocalTensor), "Input tensor must be a LocalTensor"
    output_local_tensors: dict[int, torch.Tensor] = {}

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        group_tensors = []
        for rank in group_ranks:
            group_tensors.append(tensor._local_tensors[rank])

        reduced_tensor = _local_reduce(reduce_op, group_tensors)

        scattered_tensor = torch.split(
            reduced_tensor,
            reduced_tensor.size(0) // len(group_ranks),
            dim=0,
        )

        for i, rank in enumerate(group_ranks):
            if i < len(scattered_tensor):
                output_local_tensors[rank] = scattered_tensor[i].clone()
            else:
                output_local_tensors[rank] = _zero_sized_like(reduced_tensor, 0)

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    output = LocalTensor(output_local_tensors)

    return output


def _local_functional_shard_dim_alltoall(
    tensor: torch.Tensor, gather_dim: int, shard_dim: int, group_name: str
) -> torch.Tensor:
    # "shard_dim_alltoall(Tensor input, int gather_dim, int shard_dim, str group_name) -> Tensor"
    from . import _zero_sized_like, LocalTensor

    ranks, group_offsets, offset = _prepare_collective_groups(
        _resolve_process_group(group_name)
    )

    assert isinstance(tensor, LocalTensor), "Input tensor must be a LocalTensor"
    output_local_tensors: dict[int, torch.Tensor] = {}

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        group_tensors = []
        for rank in group_ranks:
            group_tensors.append(tensor._local_tensors[rank])

        gathered_tensor = torch.cat(group_tensors, dim=gather_dim)

        split_tensor = torch.split(
            gathered_tensor,
            gathered_tensor.size(shard_dim) // len(group_ranks),
            dim=shard_dim,
        )

        for i, rank in enumerate(group_ranks):
            if i < len(split_tensor):
                output_local_tensors[rank] = split_tensor[i].clone()
            else:
                output_local_tensors[rank] = _zero_sized_like(
                    gathered_tensor, shard_dim
                )

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    output = LocalTensor(output_local_tensors)

    return output


def _local_functional_all_to_all_single(
    tensor: torch.Tensor,
    output_split_sizes: list[torch.SymInt],
    input_split_sizes: list[torch.SymInt],
    group_name: str,
) -> torch.Tensor:
    # "all_to_all_single(Tensor input, SymInt[] output_split_sizes, SymInt[] input_split_sizes, str group_name) -> Tensor"
    from . import LocalIntNode, LocalTensor

    ranks, group_offsets, offset = _prepare_collective_groups(
        _resolve_process_group(group_name)
    )

    assert isinstance(tensor, LocalTensor), "Input tensor must be a LocalTensor"

    split_local_sizes: dict[int, list[int]] = {}
    for input_split_size in input_split_sizes:
        if isinstance(input_split_size, torch.SymInt) and isinstance(
            input_split_size.node, LocalIntNode
        ):
            local_ints = dict(input_split_size.node._local_ints.items())
        else:
            local_ints = {
                rank: int(input_split_size) for rank in tensor._local_tensors.keys()
            }
        for rank, split_size in local_ints.items():
            if rank not in split_local_sizes:
                split_local_sizes[rank] = []
            split_local_sizes[rank].append(split_size)

    split_local_tensors: dict[int, list[torch.Tensor]] = {}

    for rank, split_sizes in split_local_sizes.items():
        split_local_tensors[rank] = list(
            torch.split(tensor._local_tensors[rank], split_sizes)
        )

    output_local_tensors: dict[int, torch.Tensor] = {}

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        for i, dst in enumerate(group_ranks):
            splits = []
            for j, src in enumerate(group_ranks):
                splits.append(split_local_tensors[src][i])
            output_local_tensors[dst] = torch.cat(splits)

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    output = LocalTensor(output_local_tensors)

    return output


def _local_functional_wait_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # "wait_tensor(Tensor input) -> Tensor"
    from . import LocalTensor

    assert isinstance(tensor, LocalTensor), "Input tensor must be a LocalTensor"

    return tensor


def _local_broadcast_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    root_rank: int,
    root_tensor: int,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[torch.Tensor], ScriptObject]:
    # "broadcast_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int root_rank, int root_tensor, bool async_op=True, int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)"
    from . import LocalTensor

    assert len(tensors) == 1
    assert root_tensor == 0
    tensor = tensors[0]

    ranks, group_offsets, offset = _prepare_collective_groups(process_group_so)

    # We're going to assume SPMD where for every rank group the root_rank is
    # the same relative to others
    relative_root_rank = root_rank - offset

    assert isinstance(tensor, LocalTensor), "Input tensor must be a LocalTensor"

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the broadcast on them
        group_ranks = [group_offset + r for r in ranks]
        source_rank = group_offset + relative_root_rank
        source_tensor = tensor._local_tensors[source_rank]

        # Broadcast the source tensor to all ranks in this group
        for rank in group_ranks:
            if source_rank != rank:
                tensor._local_tensors[rank].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (tensors, work_so)


def _local_reduce(
    reduce_op: ReduceOp | str,
    tensors: list[torch.Tensor],
) -> torch.Tensor:
    if reduce_op == ReduceOp.SUM or reduce_op == "sum":
        op = operator.add
    elif reduce_op == ReduceOp.AVG or reduce_op == "avg":
        op = None
    elif reduce_op == ReduceOp.PRODUCT or reduce_op == "product":
        op = operator.mul
    elif reduce_op == ReduceOp.MIN or reduce_op == "min":
        op = torch.minimum
    elif reduce_op == ReduceOp.MAX or reduce_op == "max":
        op = torch.maximum
    elif reduce_op == ReduceOp.BAND or reduce_op == "band":
        op = torch.bitwise_and
    elif reduce_op == ReduceOp.BOR or reduce_op == "bor":
        op = torch.bitwise_or
    elif reduce_op == ReduceOp.BXOR or reduce_op == "bxor":
        op = torch.bitwise_xor
    elif reduce_op == ReduceOp.PREMUL_SUM or reduce_op == "premul_sum":
        raise NotImplementedError("PREMUL_SUM: need to add binding for scaling factor")
    else:
        raise NotImplementedError(f"ReduceOp {reduce_op} not implemented")

    if reduce_op == ReduceOp.AVG or reduce_op == "avg":
        return functools.reduce(operator.add, tensors) / len(tensors)
    else:
        assert op is not None
        return functools.reduce(op, tensors)


def _local_all_reduce_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    reduce_op_so: ScriptObject,
    sparse_indices: torch.Tensor | None = None,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[torch.Tensor], ScriptObject]:
    # "allreduce_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "__torch__.torch.classes.c10d.ReduceOp reduce_op, Tensor? sparse_indices, bool async_op=True, "
    # "int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
    from . import LocalTensor

    assert len(tensors) == 1
    tensor = tensors[0]
    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    assert isinstance(tensor, LocalTensor), "Input tensor must be a LocalTensor"

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allreduce on them
        group_ranks = [group_offset + r for r in ranks]

        # Collect tensors from the specified ranks in this group
        group_tensors = []
        for rank in group_ranks:
            group_tensors.append(tensor._local_tensors[rank])

        # Perform the reduction operation
        reduced_tensor = _local_reduce(reduce_op, group_tensors)

        # Update all tensors in the group with the reduced result
        for rank in group_ranks:
            tensor._local_tensors[rank].copy_(reduced_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (tensors, work_so)


def _local_allreduce_coalesced_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    reduce_op_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # "allreduce_coalesced_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "__torch__.torch.classes.c10d.ReduceOp reduce_op, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work"
    from . import LocalTensor

    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allreduce on all tensors together
        group_ranks = [group_offset + r for r in ranks]

        # For each tensor, perform the reduction operation
        for tensor in tensors:
            assert isinstance(tensor, LocalTensor), "Input tensor must be a LocalTensor"
            # Collect tensors from the specified ranks in this group
            group_tensors = []
            for rank in group_ranks:
                group_tensors.append(tensor._local_tensors[rank])

            # Perform the reduction operation
            reduced_tensor = _local_reduce(reduce_op, group_tensors)

            # Update all tensors in the group with the reduced result
            for rank in group_ranks:
                tensor._local_tensors[rank].copy_(reduced_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_reduce_scatter_tensor_coalesced_(
    output_tensors: list[torch.Tensor],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    reduce_op_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # "reduce_scatter_tensor_coalesced_(Tensor[] outputs, Tensor[] inputs, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "__torch__.torch.classes.c10d.ReduceOp reduce_op, bool async_op=True, "
    # "int timeout=-1) -> __torch__.torch.classes.c10d.Work"

    from . import LocalTensor

    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allreduce on all tensors together
        group_ranks = [group_offset + r for r in ranks]

        # For each tensor, perform the reduction operation
        for input_tensor, output_tensor in zip(input_tensors, output_tensors):
            assert isinstance(input_tensor, LocalTensor), (
                "Input tensor must be a LocalTensor"
            )
            assert isinstance(output_tensor, LocalTensor), (
                "Output tensor must be a LocalTensor"
            )
            # Collect tensors from the specified ranks in this group
            group_inputs = []
            for rank in group_ranks:
                group_inputs.append(input_tensor._local_tensors[rank])

            # Perform the reduction operation
            reduced_input = _local_reduce(reduce_op, group_inputs)

            reduced_input_splits = torch.split(
                reduced_input, reduced_input.size(0) // len(group_ranks), dim=0
            )

            # Update all tensors in the group with the reduced result
            for i, rank in enumerate(group_ranks):
                output_tensor._local_tensors[rank].copy_(reduced_input_splits[i])

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_allgather_base_(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    process_group_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[torch.Tensor, ScriptObject]:
    # "_allgather_base_(Tensor output_tensor, Tensor input_tensor, __torch__.torch.classes.c10d.ProcessGroup
    # process_group, bool async_op=True, int timeout=-1) -> (Tensor, __torch__.torch.classes.c10d.Work)");
    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    assert isinstance(output_tensor, LocalTensor), "Output tensor must be a LocalTensor"
    assert isinstance(input_tensor, LocalTensor), "Input tensor must be a LocalTensor"

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        gathered_tensors = []
        for rank_i in group_ranks:
            gathered_tensors.append(input_tensor._local_tensors[rank_i])

        gathered_tensor = torch.cat(gathered_tensors, dim=0)

        for rank_i in group_ranks:
            output_tensor._local_tensors[rank_i].copy_(gathered_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return output_tensor, work_so


def _local_reduce_scatter_base_(  # type: ignore[no-untyped-def]
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    process_group_so: ScriptObject,
    reduce_op_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[torch.Tensor, ScriptObject]:
    # "_reduce_scatter_base_(Tensor output_tensor, Tensor input_tensor,
    # __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op,
    # bool async_op=True, int timeout=-1) -> (Tensor, __torch__.torch.classes.c10d.Work)"

    from . import LocalTensor

    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    assert isinstance(output_tensor, LocalTensor), "Output tensor must be a LocalTensor"
    assert isinstance(input_tensor, LocalTensor), "Input tensor must be a LocalTensor"

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        gathered_tensors = []
        for rank_i in group_ranks:
            gathered_tensors.append(input_tensor._local_tensors[rank_i])

        reduced_tensor = _local_reduce(reduce_op, gathered_tensors)

        scattered_tensor = torch.split(
            reduced_tensor,
            reduced_tensor.size(0) // len(group_ranks),
            dim=0,
        )

        for i, rank_i in enumerate(group_ranks):
            output_tensor._local_tensors[rank_i].copy_(scattered_tensor[i].clone())

    work = FakeWork()
    work_so = Work.boxed(work)
    return output_tensor, work_so


def _local_all_gather_(
    output_tensors: list[list[torch.Tensor]],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[list[torch.Tensor]], ScriptObject]:
    # "allgather_(Tensor[][] output_tensors, Tensor[] input_tensors, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True, "
    # "int timeout=-1) -> (Tensor[][], __torch__.torch.classes.c10d.Work)");

    from . import LocalTensor

    assert len(output_tensors) == 1
    assert len(input_tensors) == 1

    input_tensor = input_tensors[0]
    # pyrefly: ignore [bad-assignment]
    output_tensors = output_tensors[0]

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    for i in range(len(output_tensors)):
        assert isinstance(output_tensors[i], LocalTensor), (
            "Output tensor must be a LocalTensor"
        )

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the all_gather on them
        group_ranks = [group_offset + r for r in ranks]

        # For each rank in the group, gather from their input tensor
        for i, rank_i in enumerate(group_ranks):
            # allgather object happens to create pure tensor, so we special case it here
            source_tensor = input_tensor
            if isinstance(input_tensor, LocalTensor):
                source_tensor = input_tensor._local_tensors[rank_i]
            # pyrefly: ignore [missing-attribute]
            output_tensors[i].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    # pyrefly: ignore [bad-return]
    return ([output_tensors], work_so)


def _local_allgather_into_tensor_coalesced_(
    output_tensors: list[torch.Tensor],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    async_op: bool = True,
) -> ScriptObject:
    # "allgather_into_tensor_coalesced_(Tensor[] outputs, Tensor[] inputs, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True) "
    # "-> __torch__.torch.classes.c10d.Work"
    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    # Each output tensor should be sized to hold all gathered inputs
    # outputs[i] will contain all inputs[i] from all ranks
    assert len(output_tensors) == len(input_tensors), (
        f"Number of outputs ({len(output_tensors)}) must match number of inputs ({len(input_tensors)})"
    )

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allgather_into_tensor on them
        group_ranks = [group_offset + r for r in ranks]

        # For each input/output pair
        for input_tensor, output_tensor in zip(input_tensors, output_tensors):
            assert isinstance(input_tensor, LocalTensor), (
                "Input tensor must be a LocalTensor"
            )
            assert isinstance(output_tensor, LocalTensor), (
                "Output tensor must be a LocalTensor"
            )
            # Gather input_tensor from all ranks into output_tensor
            # The output should be a concatenation of all inputs along the first dimension
            gathered_tensors = []
            for rank in group_ranks:
                gathered_tensors.append(input_tensor._local_tensors[rank])

            # Concatenate along first dimension and copy to output
            if gathered_tensors:
                concatenated = torch.cat(gathered_tensors, dim=0)
                for rank in group_ranks:
                    output_tensor._local_tensors[rank].copy_(concatenated)

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_gather_(
    output_tensors: list[list[torch.Tensor]],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    root_rank: int,
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # "gather_(Tensor[][] output_tensors, Tensor[] input_tensors, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, "
    # "bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work"
    raise NotImplementedError(
        "LocalTensor does not support MPMD operations like gather "
        "(only root rank receives data). Use SPMD collective operations like allgather instead."
    )


def _local_scatter_(
    output_tensors: list[torch.Tensor],
    input_tensors: list[list[torch.Tensor]],
    process_group_so: ScriptObject,
    root_rank: int,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[torch.Tensor], ScriptObject]:
    # "scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, "
    # "bool async_op=True, int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)");

    from . import LocalTensor

    assert len(output_tensors) == 1
    assert len(input_tensors) == 1
    output_tensor = output_tensors[0]
    # pyrefly: ignore [bad-assignment]
    input_tensors = input_tensors[0]

    ranks, group_offsets, offset = _prepare_collective_groups(process_group_so)

    # We're going to assume SPMD where for every rank group the root_rank is
    # the same relative to others
    relative_root_rank = root_rank - offset

    assert isinstance(output_tensor, LocalTensor), "Output tensor must be a LocalTensor"
    assert len(ranks) == len(input_tensors), (ranks, input_tensors)

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the scatter on them
        group_ranks = [group_offset + r for r in ranks]

        # Root rank scatters its input tensors to all ranks in this group
        for i, rank in enumerate(group_ranks):
            input_tensor = input_tensors[i]
            assert isinstance(input_tensor, LocalTensor)
            # Each rank i gets the i-th input tensor from the root
            source_tensor = input_tensor._local_tensors[
                group_offset + relative_root_rank
            ]
            output_tensor._local_tensors[rank].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (output_tensors, work_so)


def _local_alltoall_(
    output_tensors: list[torch.Tensor],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[torch.Tensor], ScriptObject]:
    # "alltoall_(Tensor[] output_tensors, Tensor[] input_tensors, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True, "
    # "int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)";

    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    assert len(input_tensors) == len(output_tensors) == len(ranks), (
        f"Number of input tensors ({len(input_tensors)}), "
        f"output tensors ({len(output_tensors)}), and ranks ({len(ranks)}) must match"
    )

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the alltoall on them
        group_ranks = [group_offset + r for r in ranks]

        # In alltoall, rank i sends input_tensors[j] to rank j and receives into output_tensors[i] from rank j
        for i, rank_i in enumerate(group_ranks):
            output_tensor = output_tensors[i]
            assert isinstance(output_tensor, LocalTensor), (
                "Output tensor must be a LocalTensor"
            )
            for j, rank_j in enumerate(group_ranks):
                input_tensor = input_tensors[j]
                assert isinstance(input_tensor, LocalTensor), (
                    "Input tensor must be a LocalTensor"
                )
                # Rank i's j-th input tensor goes to rank j's i-th output tensor
                source_tensor = input_tensor._local_tensors[rank_i]
                output_tensor._local_tensors[rank_j].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (output_tensors, work_so)


def _local_alltoall_base_(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    process_group_so: ScriptObject,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # "alltoall_base_(Tensor output, Tensor input, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int[] output_split_sizes, int[] input_split_sizes, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work";

    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    assert isinstance(input_tensor, LocalTensor), "Input tensor must be a LocalTensor"
    assert isinstance(output_tensor, LocalTensor), "Output tensor must be a LocalTensor"
    # Convert split sizes to lists if they aren't already
    if output_split_sizes is not None:
        output_split_sizes = list(output_split_sizes)
    if input_split_sizes is not None:
        input_split_sizes = list(input_split_sizes)

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the alltoall_base on them
        group_ranks = [group_offset + r for r in ranks]

        for i, rank_i in enumerate(group_ranks):
            # Split input tensor from rank_i according to input_split_sizes
            rank_tensor = input_tensor._local_tensors[rank_i]

            if input_split_sizes is not None and len(input_split_sizes) > 0:
                # Split the input tensor
                input_splits = torch.split(rank_tensor, input_split_sizes, dim=0)
            else:
                # No split sizes specified, split evenly
                split_size = rank_tensor.size(0) // len(group_ranks)
                input_splits = torch.split(rank_tensor, split_size, dim=0)

            # Send each split to the corresponding rank
            for j, rank_j in enumerate(group_ranks):
                if j < len(input_splits):
                    split_tensor = input_splits[j]

                    # Determine where to place this split in the output tensor
                    if output_split_sizes is not None and len(output_split_sizes) > 0:
                        # Calculate offset based on output split sizes
                        output_offset = sum(output_split_sizes[:i]) if i > 0 else 0
                        end_offset = (
                            output_offset + output_split_sizes[i]
                            if i < len(output_split_sizes)
                            else output_tensor._local_tensors[rank_j].size(0)
                        )
                    else:
                        # No output split sizes, use even splits
                        split_size = output_tensor._local_tensors[rank_j].size(
                            0
                        ) // len(group_ranks)
                        output_offset = i * split_size
                        end_offset = min(
                            (i + 1) * split_size,
                            output_tensor._local_tensors[rank_j].size(0),
                        )

                    # Copy the split to the appropriate section of the output tensor
                    output_section = output_tensor._local_tensors[rank_j][
                        output_offset:end_offset
                    ]
                    if output_section.numel() > 0:
                        # Reshape split_tensor to match output_section if necessary
                        if split_tensor.size() != output_section.size():
                            split_tensor = split_tensor.view(output_section.size())
                        output_section.copy_(split_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_barrier(
    tensor: torch.Tensor,
    process_group_so: ScriptObject,
    device_ids: list[int],
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # "barrier(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int[] device_ids, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work";

    from . import LocalTensor

    # Barrier is a synchronization primitive - in local simulation,
    # we don't need to do any actual work since all "ranks" are in the same process
    # Just validate that the tensor is a LocalTensor
    assert isinstance(tensor, LocalTensor)

    # In a real distributed setting, barrier would synchronize all processes
    # In local simulation, this is essentially a no-op since all ranks are local
    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_monitored_barrier_(
    tensor: torch.Tensor,
    process_group_so: ScriptObject,
    device_ids: list[int],
    timeout: int,
    wait_all_ranks: bool,
) -> None:
    # "monitored_barrier_(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int[] device_ids, int timeout, bool wait_all_ranks) -> ()";

    from . import LocalTensor

    # Monitored barrier is a synchronization primitive with monitoring - in local simulation,
    # we don't need to do any actual work since all "ranks" are in the same process
    # Just validate that the tensor is a LocalTensor
    assert isinstance(tensor, LocalTensor)

    # In a real distributed setting, monitored barrier would synchronize all processes
    # and provide monitoring capabilities. In local simulation, this is essentially a no-op
    # since all ranks are local and no actual synchronization is needed
    return


def _local_send(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    dst: int,
    tag: int,
) -> ScriptObject:
    # "send(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int dst, int tag) -> __torch__.torch.classes.c10d.Work";

    from . import LocalRunnerMode, LocalTensor

    assert len(tensors) == 1
    tensor = tensors[0]

    assert isinstance(tensor, LocalTensor), "Input tensor must be a Tensor"
    src = int(tensor.__src_rank__)

    LocalRunnerMode.current()._signal_send(src, dst, tensor._local_tensors[src])

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_recv_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    src: int,
    tag: int,
) -> ScriptObject:
    # "recv_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int src, int tag) -> __torch__.torch.classes.c10d.Work";
    from . import LocalRunnerMode, LocalTensor

    assert len(tensors) == 1
    tensor = tensors[0]

    assert isinstance(tensor, LocalTensor), "Input tensor must be a Tensor"
    dst = int(tensor.__src_rank__)

    def _recv_and_store(timeout: timedelta) -> bool:
        def _wait_and_store(obj: object) -> None:
            assert isinstance(obj, torch.Tensor), "Expected to receive a Tensor"
            assert isinstance(tensor, LocalTensor), "Input tensor must be a Tensor"
            tensor._local_tensors[dst] = obj

        LocalRunnerMode.current()._wait_recv(src, dst, _wait_and_store)
        return True

    work = PythonCallbackWork(_recv_and_store)
    work_so = Work.boxed(work)
    return work_so


def _local_recv_any_source_(
    tensors: list[torch.Tensor], process_group_so: ScriptObject, tag: int
) -> ScriptObject:
    # "recv_any_source_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int tag) -> __torch__.torch.classes.c10d.Work";

    return _local_recv_(tensors, process_group_so, -1, tag)


def _attach_rank(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Attaches rank as an attribute to given tensor so that the send or recv implementation
    knows which rank initiates the operation (note under local tensor mode ).
    """
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        tensor = tensor._local_tensor

    tensor.__src_rank__ = rank  # type: ignore[attr-defined]
    return tensor


def local_p2p_op(
    dst: torch.SymInt,
    tensor: torch.Tensor,
    op: Callable[[torch.Tensor, int], Work | None],
) -> Work | None | list[Work | None]:
    """
    Runs a point-to-point (P2P) operation for all combinations of source and destination ranks.
    """
    _check_op(op)

    from . import LocalIntNode

    assert isinstance(dst.node, LocalIntNode), (
        "Expected 'dst' to be a LocalIntNode where the value is the destination rank and key is the source rank"
    )

    w = []
    for s, d in dst.node._local_ints.items():
        tensor = _attach_rank(tensor, s)
        w.append(op(tensor, d))
    return w


def wait_all(work: Work | None | list[Work | None]) -> None:
    """
    Waits for all work objects in the input to complete.

    A single Work object, None, or a list of Work objects (possibly containing None).
    If None, does nothing. If a single Work, waits for it to complete. If a list, waits
    for each non-None Work in the list to complete.
    """

    if work is None:
        return
    if isinstance(work, Work):
        work = [work]
    for w in work:
        if w is None:
            continue
        w.wait()
