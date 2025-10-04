import functools
import operator
from itertools import product

import torch
from torch._C import ScriptObject
from torch._C._distributed_c10d import FakeWork
from torch.distributed._pycute.int_tuple import flatten
from torch.distributed._pycute.layout import complement, Layout
from torch.distributed.distributed_c10d import ProcessGroup, ReduceOp, Work


# NOTE: Most of the c10d collectives often take a Tensor[] (or Tensor[][])
# when you would expect Tensor (or Tensor[]).  In fact, there will only ever
# be one Tensor in this case; the old signature was to support dispatching a
# collective on multiple devices (ala DataParallel) but we don't support that
# API anymore.  Note that we are not 100% consistent about this; some more
# modern collectives like _allgather_base_ got rid of the unnecessary list.
# When in doubt, consult the code that dispatches to the collective on the PG
# in distributed_c10d.py e.g., work = group.allgather([tensor_list], [tensor],
# opts) indicates its always a list.


# TODO: this claude code implementation sucks, redo it
def _indices_to_layout(indices: list[int]) -> Layout:
    """
    Convert a sorted list of indices to a pycute Layout using the mathematical
    approach based on the admissible for complement property.

    Args:
        indices: A sorted list of integers starting from 0

    Returns:
        Layout: A pycute Layout that generates the given indices
    """
    if not indices:
        return Layout(0, 0)

    if len(indices) == 1:
        if indices[0] == 0:
            return Layout(1, 1)
        else:
            raise ValueError("Single index must be 0")

    strides = []
    shapes = []
    remaining = set(indices)

    # Always start with stride 1
    current_stride = 1
    max_iterations = len(indices)  # Safety limit
    iteration = 0

    while remaining and iteration < max_iterations:
        iteration += 1

        # Count consecutive multiples of current_stride starting from 0
        size = 0
        while size * current_stride in remaining:
            size += 1

        if size > 0:
            # Found a valid dimension - remove all multiples
            for i in range(size):
                remaining.discard(i * current_stride)
            strides.append(current_stride)
            shapes.append(size)

            # Calculate next stride using admissible property
            current_stride = current_stride * size
        else:
            # No pattern from 0, jump to minimum remaining element
            if remaining:
                current_stride = min(remaining)
            else:
                break

    if iteration >= max_iterations:
        raise RuntimeError(
            f"Algorithm did not converge after {max_iterations} iterations"
        )

    # Convert to proper format for Layout
    if len(shapes) == 1:
        return Layout(shapes[0], strides[0])
    else:
        return Layout(tuple(shapes), tuple(strides))


# NB: lifted from https://github.com/pytorch/pytorch/pull/161016
def _layout_to_indices(layout: Layout) -> list[int]:
    return [
        sum(c * s for c, s in zip(coord, flatten(layout.stride)))
        for coord in product(*(range(s) for s in flatten(layout.shape)))
    ]


def _prepare_collective_groups(
    process_group_so: ScriptObject,
) -> tuple[list[int], Layout, ProcessGroup, list[int], int]:
    """
    Common helper function to prepare process group information for collective operations.

    Returns:
        tuple: (ranks, layout, global_pg, group_offsets, offset)
    """

    process_group = ProcessGroup.unbox(process_group_so)

    ranks = torch.distributed.get_process_group_ranks(process_group)
    assert ranks
    # TODO: We can handle permutations but the layout inference algorithm will
    # lose the permutation so we will have to reapply it
    assert ranks == sorted(ranks), ranks
    offset = ranks[0]
    ranks = [r - offset for r in ranks]
    layout = _indices_to_layout(ranks)

    global_pg = torch.distributed.distributed_c10d._get_default_group()
    group_offsets = _layout_to_indices(complement(layout, global_pg.size()))

    return ranks, layout, global_pg, group_offsets, offset


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
    tensor = tensors[0]
    assert root_tensor == 0

    ranks, layout, global_pg, group_offsets, offset = _prepare_collective_groups(
        process_group_so
    )

    # We're going to assume SPMD where for every rank group the root_rank is
    # the same relative to others
    relative_root_rank = root_rank - offset

    if not isinstance(tensor, LocalTensor):
        raise ValueError(f"Expected LocalTensor for local broadcast, got {tensor}")

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

    assert len(tensors) == 1
    tensor = tensors[0]
    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]
    from . import LocalTensor

    ranks, layout, global_pg, group_offsets, _offset = _prepare_collective_groups(
        process_group_so
    )

    if not isinstance(tensor, LocalTensor):
        raise ValueError(f"Expected LocalTensor for local all_reduce, got {tensor}")

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allreduce on them
        group_ranks = [group_offset + r for r in ranks]

        # Collect tensors from the specified ranks in this group
        group_tensors = []
        for rank in group_ranks:
            assert rank in tensor._local_tensors
            group_tensors.append(tensor._local_tensors[rank])

        # Perform the reduction operation
        if reduce_op == ReduceOp.SUM:
            op = operator.add
        elif reduce_op == ReduceOp.AVG:
            op = None
        elif reduce_op == ReduceOp.PRODUCT:
            op = operator.mul
        elif reduce_op == ReduceOp.MIN:
            op = torch.minimum
        elif reduce_op == ReduceOp.MAX:
            op = torch.maximum
        elif reduce_op == ReduceOp.BAND:
            op = torch.bitwise_and
        elif reduce_op == ReduceOp.BOR:
            op = torch.bitwise_or
        elif reduce_op == ReduceOp.BXOR:
            op = torch.bitwise_xor
        elif reduce_op == ReduceOp.PREMUL_SUM:
            raise NotImplementedError(
                "PREMUL_SUM: need to add binding for scaling factor"
            )
        else:
            raise NotImplementedError(f"ReduceOp {reduce_op} not implemented")

        if reduce_op == ReduceOp.AVG:
            denom = len(group_tensors)
            reduced_tensor = functools.reduce(operator.add, group_tensors) / len(
                group_tensors
            )
        else:
            assert op is not None
            reduced_tensor = functools.reduce(op, group_tensors)

        # Update all tensors in the group with the reduced result
        for rank in group_ranks:
            if rank in tensor._local_tensors:
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
    ranks, layout, global_pg, group_offsets, _offset = _prepare_collective_groups(
        process_group_so
    )

    for tensor in tensors:
        if not isinstance(tensor, LocalTensor):
            raise ValueError(
                f"Expected LocalTensor for local allreduce_coalesced, got {tensor}"
            )

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allreduce on all tensors together
        group_ranks = [group_offset + r for r in ranks]

        # For each tensor, perform the reduction operation
        for tensor in tensors:
            assert isinstance(tensor, LocalTensor)
            # Collect tensors from the specified ranks in this group
            group_tensors = []
            for rank in group_ranks:
                assert rank in tensor._local_tensors
                group_tensors.append(tensor._local_tensors[rank])

            # Perform the reduction operation
            if reduce_op == ReduceOp.SUM:
                op = operator.add
            elif reduce_op == ReduceOp.PRODUCT:
                op = operator.mul
            elif reduce_op == ReduceOp.MIN:
                op = torch.minimum
            elif reduce_op == ReduceOp.MAX:
                op = torch.maximum
            else:
                raise NotImplementedError(f"ReduceOp {reduce_op} not implemented")

            reduced_tensor = functools.reduce(op, group_tensors)

            # Update all tensors in the group with the reduced result
            for rank in group_ranks:
                if rank in tensor._local_tensors:
                    tensor._local_tensors[rank].copy_(reduced_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_reduce_scatter_tensor_coalesced_(
    outputs: list[torch.Tensor],
    inputs: list[torch.Tensor],
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
    ranks, layout, global_pg, group_offsets, _offset = _prepare_collective_groups(
        process_group_so
    )

    # Validate inputs
    for input in inputs:
        if not isinstance(input, LocalTensor):
            raise ValueError(
                f"Expected LocalTensor for local allgather_into_tensor_coalesced, got {input}"
            )

    for output in outputs:
        if not isinstance(output, LocalTensor):
            raise ValueError(
                f"Expected LocalTensor for local allgather_into_tensor_coalesced, got {output}"
            )

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allreduce on all tensors together
        group_ranks = [group_offset + r for r in ranks]

        # For each tensor, perform the reduction operation
        for input, output in zip(inputs, outputs):
            assert isinstance(input, LocalTensor)
            assert isinstance(output, LocalTensor)
            # Collect tensors from the specified ranks in this group
            group_inputs = []
            for rank in group_ranks:
                assert rank in input._local_tensors
                group_inputs.append(input._local_tensors[rank])

            # Perform the reduction operation
            if reduce_op == ReduceOp.SUM:
                op = operator.add
            elif reduce_op == ReduceOp.PRODUCT:
                op = operator.mul
            elif reduce_op == ReduceOp.MIN:
                op = torch.minimum
            elif reduce_op == ReduceOp.MAX:
                op = torch.maximum
            else:
                raise NotImplementedError(f"ReduceOp {reduce_op} not implemented")

            reduced_input = functools.reduce(op, group_inputs)
            reduced_inpit_splits = torch.split(
                reduced_input, reduced_input.size(0) // len(group_ranks), dim=0
            )

            # Update all tensors in the group with the reduced result
            for rank in group_ranks:
                if rank in output._local_tensors:
                    output._local_tensors[rank].copy_(reduced_inpit_splits[rank])

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


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

    ranks, layout, global_pg, group_offsets, _offset = _prepare_collective_groups(
        process_group_so
    )

    if not isinstance(input_tensor, LocalTensor):
        raise ValueError(
            f"Expected LocalTensor for local all_gather, got {input_tensor}"
        )

    for i in range(len(output_tensors)):
        assert len(ranks) == len(output_tensors[i]), (ranks, output_tensors[i])
        for j in range(len(output_tensors[i])):
            assert isinstance(output_tensors[i][j], LocalTensor)

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the all_gather on them
        group_ranks = [group_offset + r for r in ranks]

        # For each rank in the group, gather from their input tensor
        for i, rank_i in enumerate(group_ranks):
            for j in range(len(output_tensors)):
                output_tensors[j][i].copy_(input_tensor._local_tensors[rank_i])

    work = FakeWork()
    work_so = Work.boxed(work)
    return (output_tensors, work_so)


def _local_allgather_into_tensor_coalesced_(
    outputs: list[torch.Tensor],
    inputs: list[torch.Tensor],
    process_group_so: ScriptObject,
    async_op: bool = True,
) -> ScriptObject:
    # "allgather_into_tensor_coalesced_(Tensor[] outputs, Tensor[] inputs, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True) "
    # "-> __torch__.torch.classes.c10d.Work"
    from . import LocalTensor

    ranks, layout, global_pg, group_offsets, _offset = _prepare_collective_groups(
        process_group_so
    )

    # Validate inputs
    for input_tensor in inputs:
        if not isinstance(input_tensor, LocalTensor):
            raise ValueError(
                f"Expected LocalTensor for local allgather_into_tensor_coalesced, got {input_tensor}"
            )

    for output_tensor in outputs:
        if not isinstance(output_tensor, LocalTensor):
            raise ValueError(
                f"Expected LocalTensor for local allgather_into_tensor_coalesced, got {output_tensor}"
            )

    # Each output tensor should be sized to hold all gathered inputs
    # outputs[i] will contain all inputs[i] from all ranks
    assert len(outputs) == len(inputs), (
        f"Number of outputs ({len(outputs)}) must match number of inputs ({len(inputs)})"
    )

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allgather_into_tensor on them
        group_ranks = [group_offset + r for r in ranks]

        # For each input/output pair
        for input_idx, (input_tensor, output_tensor) in enumerate(zip(inputs, outputs)):
            assert isinstance(input_tensor, LocalTensor)
            assert isinstance(output_tensor, LocalTensor)
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
    input_tensors = input_tensors[0]

    ranks, layout, global_pg, group_offsets, offset = _prepare_collective_groups(
        process_group_so
    )

    # We're going to assume SPMD where for every rank group the root_rank is
    # the same relative to others
    relative_root_rank = root_rank - offset

    if not isinstance(output_tensor, LocalTensor):
        raise ValueError(f"Expected LocalTensor for local scatter, got {output_tensor}")

    for t in input_tensors:
        assert isinstance(t, LocalTensor)

    assert len(ranks) == len(input_tensors), (ranks, input_tensors)

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the scatter on them
        group_ranks = [group_offset + r for r in ranks]

        # Root rank scatters its input tensors to all ranks in this group
        for i, rank in enumerate(group_ranks):
            # Each rank i gets the i-th input tensor from the root
            source_tensor = input_tensors[i]._local_tensors[
                group_offset + relative_root_rank
            ]
            output_tensor._local_tensors[rank].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (output_tensors, work_so)


# TODO: I haven't carefully checked if the alltoall implementations look
# correct yet


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

    ranks, layout, global_pg, group_offsets, _offset = _prepare_collective_groups(
        process_group_so
    )

    # Validate inputs and outputs
    for input_tensor in input_tensors:
        if not isinstance(input_tensor, LocalTensor):
            raise ValueError(
                f"Expected LocalTensor for local alltoall, got {input_tensor}"
            )

    for output_tensor in output_tensors:
        if not isinstance(output_tensor, LocalTensor):
            raise ValueError(
                f"Expected LocalTensor for local alltoall, got {output_tensor}"
            )

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
            for j, rank_j in enumerate(group_ranks):
                # Rank i's j-th input tensor goes to rank j's i-th output tensor
                source_tensor = input_tensors[j]._local_tensors[rank_i]
                output_tensors[i]._local_tensors[rank_j].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (output_tensors, work_so)


def _local_alltoall_base_(
    output: torch.Tensor,
    input: torch.Tensor,
    process_group_so: ScriptObject,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # "alltoall_base_(Tensor output, Tensor input, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int[] output_split_sizes, int[] input_split_sizes, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work";

    from . import LocalTensor

    ranks, layout, global_pg, group_offsets, _offset = _prepare_collective_groups(
        process_group_so
    )

    if not isinstance(input, LocalTensor):
        raise ValueError(
            f"Expected LocalTensor for local alltoall_base input, got {input}"
        )

    if not isinstance(output, LocalTensor):
        raise ValueError(
            f"Expected LocalTensor for local alltoall_base output, got {output}"
        )

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
            input_tensor = input._local_tensors[rank_i]

            if input_split_sizes is not None and len(input_split_sizes) > 0:
                # Split the input tensor
                input_splits = torch.split(input_tensor, input_split_sizes, dim=0)
            else:
                # No split sizes specified, split evenly
                split_size = input_tensor.size(0) // len(group_ranks)
                input_splits = torch.split(input_tensor, split_size, dim=0)

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
                            else output._local_tensors[rank_j].size(0)
                        )
                    else:
                        # No output split sizes, use even splits
                        split_size = output._local_tensors[rank_j].size(0) // len(
                            group_ranks
                        )
                        output_offset = i * split_size
                        end_offset = min(
                            (i + 1) * split_size, output._local_tensors[rank_j].size(0)
                        )

                    # Copy the split to the appropriate section of the output tensor
                    output_section = output._local_tensors[rank_j][
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

    ranks, layout, global_pg, group_offsets, _offset = _prepare_collective_groups(
        process_group_so
    )

    # Barrier is a synchronization primitive - in local simulation,
    # we don't need to do any actual work since all "ranks" are in the same process
    # Just validate that the tensor is a LocalTensor
    if not isinstance(tensor, LocalTensor):
        raise ValueError(f"Expected LocalTensor for local barrier, got {tensor}")

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

    ranks, layout, global_pg, group_offsets, _offset = _prepare_collective_groups(
        process_group_so
    )

    # Monitored barrier is a synchronization primitive with monitoring - in local simulation,
    # we don't need to do any actual work since all "ranks" are in the same process
    # Just validate that the tensor is a LocalTensor
    if not isinstance(tensor, LocalTensor):
        raise ValueError(
            f"Expected LocalTensor for local monitored_barrier, got {tensor}"
        )

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

    raise NotImplementedError(
        "LocalTensor does not support MPMD operations like send. "
        "Use SPMD collective operations instead."
    )


def _local_recv_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    src: int,
    tag: int,
) -> ScriptObject:
    # "recv_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int src, int tag) -> __torch__.torch.classes.c10d.Work";

    raise NotImplementedError(
        "LocalTensor does not support MPMD operations like recv. "
        "Use SPMD collective operations instead."
    )


def _local_recv_any_source_(
    tensors: list[torch.Tensor], process_group_so: ScriptObject, tag: int
) -> ScriptObject:
    # "recv_any_source_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int tag) -> __torch__.torch.classes.c10d.Work";

    raise NotImplementedError(
        "LocalTensor does not support MPMD operations like recv_any_source. "
        "Use SPMD collective operations instead."
    )
