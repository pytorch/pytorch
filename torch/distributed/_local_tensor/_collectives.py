import functools
import operator
from typing import Union

import torch
from torch.distributed._distributed_c10d import FakeWork
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


def _prepare_collective_groups(process_group_so):
    """
    Common helper function to prepare process group information for collective operations.

    Returns:
        tuple: (ranks, layout, global_pg, group_offsets, offset)
    """
    from pycute.layout import complement

    from . import indices_to_layout, layout_to_indices

    process_group = ProcessGroup.unbox(process_group_so)

    ranks = torch.distributed.get_process_group_ranks(process_group)
    assert ranks
    # TODO: We can handle permutations but the layout inference algorithm will
    # lose the permutation so we will have to reapply it
    assert ranks == sorted(ranks), ranks
    offset = ranks[0]
    ranks = [r - offset for r in ranks]
    layout = indices_to_layout(ranks)

    global_pg = torch.distributed.distributed_c10d._get_default_group()
    group_offsets = layout_to_indices(complement(layout, global_pg.size()))

    return ranks, layout, global_pg, group_offsets, offset


def _local_broadcast_(
    tensors, process_group_so, root_rank, root_tensor, async_op=True, timeout=-1
):
    # "broadcast_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, int root_tensor, bool async_op=True, int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)");

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
    tensors, process_group_so, reduce_op_so, sparse_indices, async_op=True, timeout=-1
):
    # "allreduce_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, Tensor? sparse_indices, bool async_op=True, int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)");

    from . import LocalTensor

    assert len(tensors) == 1
    tensor = tensors[0]

    reduce_op = ReduceOp.unbox(reduce_op_so)
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
    return (tensors, work_so)


# TODO: "allreduce_coalesced_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work")


def _local_all_gather_(
    output_tensorss, input_tensors, process_group_so, async_op=True, timeout=-1
):
    # "allgather_(Tensor[][] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True, int timeout=-1) -> (Tensor[][], __torch__.torch.classes.c10d.Work)");

    from . import LocalTensor

    assert len(output_tensorss) == 1
    assert len(input_tensors) == 1
    output_tensors = output_tensorss[0]
    input_tensor = input_tensors[0]

    ranks, layout, global_pg, group_offsets, _offset = _prepare_collective_groups(
        process_group_so
    )

    if not isinstance(input_tensor, LocalTensor):
        raise ValueError(
            f"Expected LocalTensor for local all_gather, got {input_tensor}"
        )

    for t in output_tensors:
        assert isinstance(t, LocalTensor)

    assert len(ranks) == len(output_tensors), (ranks, output_tensors)
    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the all_gather on them
        group_ranks = [group_offset + r for r in ranks]

        # For each rank in the group, gather from their input tensor
        for i, rank_i in enumerate(group_ranks):
            output_tensors[i].copy_(input_tensor._local_tensors[rank_i])

    work = FakeWork()
    work_so = Work.boxed(work)
    return (output_tensorss, work_so)


# TODO: "gather_(Tensor[][] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work");

def _local_scatter_(
    output_tensors, input_tensorss, process_group_so, root_rank, async_op=True, timeout=-1
):
    # "scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, bool async_op=True, int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)");

    from . import LocalTensor

    assert len(output_tensors) == 1
    assert len(input_tensorss) == 1
    output_tensor = output_tensors[0]
    input_tensors = input_tensorss[0]

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
            source_tensor = input_tensors[i]._local_tensors[group_offset + relative_root_rank]
            output_tensor._local_tensors[rank].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (output_tensors, work_so)

# TODO: "alltoall_(Tensor[] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True, int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
# TODO: "alltoall_base_(Tensor output, Tensor input, __torch__.torch.classes.c10d.ProcessGroup process_group, int[] output_split_sizes, int[] input_split_sizes, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work");
# TODO: "barrier(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, int[] device_ids, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work");
# TODO: "monitored_barrier_(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, int[] device_ids, int timeout, bool wait_all_ranks) -> ()");
# TODO: "send(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int dst, int tag) -> __torch__.torch.classes.c10d.Work");
# TODO: "recv_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int src, int tag) -> __torch__.torch.classes.c10d.Work");
# TODO: "recv_any_source_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int tag) -> __torch__.torch.classes.c10d.Work");
