# mypy: allow-untyped-defs
import logging
from typing import cast, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch.utils._ordered_set import OrderedSet

from . import config, ir
from .virtualized import V


log = logging.getLogger(__name__)


# NOTE [lowering-time collective optimization]
#
# In collective communication libraries such as NCCL, every rank maintains
# communication buffers that are remotely accessible by some peers. Depending
# on the underlying transport, remote accessibility may be established via
# mechanisms such as ib_reg_mr, CUDA P2P, or CUDA multicast. Typically, these
# buffers are private to the communication library by default, and
# communication ops copy user data in and out of these buffers.
#
# To prevent these copies, an optimization commonly known as "user buffer
# registration" can be employed. This allows direct establishment of remote
# accessibility on user buffers, eliminating the need for copying. However,
# this optimization introduces stringent usage requirements, which are
# typically hard to satisfy without being intrusive to the user code:
#
# - Establishing remote accessibility is expensive and often done ahead of
# time. In such implementations, all ranks must agree on the set of allocations
# used for every collective op. Failing to meet this requirement can
# lead to runtime errors or even silent correctness issues.
# - Even if the collective communication library supports gracefully falling
# back to "unregistered" implementations, the fallback mechanism would nullify
# the optimization.
# - Some communication mechanisms impose stricter requirements than others. For
# example, CUDA's multicast + multi-mem instructions require all ranks to agree
# not only on the allocations used for every collective but also on the offsets
# within these allocations.
#
# To support all different mechanisms with optimal results, we aim to satisfy
# the strictest requirement for this family of optimizations - we ensures that
# every collective op invocation is guaranteed to operate on the same
# allocation, at the same offset, in every iteration.
#
# For eligible collective ops, we identify communication buffers at lowering
# time and optionally choose to lower the op to a different kernel
# (ommunication libraries like NCCL handle both registered and non-registered
# buffers transparently within the same op, though some may require different
# ops for different cases). Later, the codegen will perform "persistent
# allocation" to satisfy the aforementioned constraints, and optionally,
# perform buffer planning to optimize overall memory usage.
def can_realize_as_comm_buffer(buffer: ir.Buffer, comm_buffer_type: str) -> bool:
    """
    Check if a buffer can be realized as the specified `comm_buffer_type`.
    """
    assert isinstance(buffer, ir.Buffer), type(buffer)

    # The buffer is already realized as a comm buffer
    if buffer.get_name() in V.graph.comm_buffer_type:
        return comm_buffer_type == V.graph.comm_buffer_type[buffer.get_name()]

    # We can realized a buffer as comm buffer only if we know its size at
    # codegen time.
    try:
        V.graph.sizevars.size_hint(buffer.get_numel())
    except Exception:
        return False

    # The buffer isn't allocated by the codegen in the first place.
    # TODO(yifu): consider copying the extern/input buffer in question.
    if isinstance(
        buffer,
        (ir.InputBuffer, ir.ExternKernelAlloc, ir.MultiOutput),
    ):
        return False

    # Other cases for which the codegen won't allocate buffer.
    layout = buffer.get_layout()
    if isinstance(
        layout,
        (ir.MutationLayoutSHOULDREMOVE, ir.NoneLayout, ir.NonOwningLayout),
    ):
        return False

    return True


def realize_as_comm_buffer(buffer: ir.Buffer, comm_buffer_type: str) -> None:
    """
    Realize a buffer as comm buffer. Specifically this does two things:

    - Add an entry in `V.graph.comm_buffer_type` for the buffer. The info will
      be used by the codegen to dispatch to the appropriate allocator for the
      comm buffer.
    - Add the buffer to the `never_reuse_buffers` set.
    """
    buffer.realize()
    V.graph.comm_buffer_type[buffer.get_name()] = comm_buffer_type
    V.graph.never_reuse_buffers.add(buffer.get_name())


_bufs_to_skip_wait: OrderedSet[Tuple[int, str]] = OrderedSet()


def mark_as_skip_wait(x: ir.IRNode) -> None:
    """
    If a non-blocking collective is lowered as a blocking collective, the wait
    node in the original graph becomes useless and we can skip the lowering it.
    """
    _bufs_to_skip_wait.add((id(V.graph), x.get_name()))


def should_skip_wait(x: ir.IRNode) -> bool:
    return (id(V.graph), x.get_name()) in _bufs_to_skip_wait


def _get_buffer(x: ir.TensorBox) -> ir.Buffer:
    if isinstance(x.data, ir.BaseView):
        # TensorBox -> View -> StorageBox -> Buffer
        return x.data.unwrap_view().data
    elif isinstance(x.data, ir.StorageBox):
        # TensorBox -> StorageBox -> Buffer
        return cast(ir.Buffer, x.data.data)
    else:
        raise AssertionError(
            "Expect the data attr of a TensorBox to be either "
            f"a view or a storage box (got {x.data})"
        )


def _should_lower_as_one_shot_all_reduce(
    inp: ir.TensorBox, reduce_op: str, group_name: str
):
    from torch.distributed._symmetric_memory import is_symm_mem_enabled_for_group

    buffer = _get_buffer(inp)
    inp_size = buffer.get_numel() * buffer.get_dtype().itemsize
    return (
        config._collective.auto_select
        and is_symm_mem_enabled_for_group(group_name)
        and can_realize_as_comm_buffer(buffer, "symm_mem")
        and reduce_op in ("sum",)
        and inp_size <= config._collective.one_shot_all_reduce_threshold_bytes
    )


def _one_shot_all_reduce(inp: ir.TensorBox, reduce_op, group_name):
    buffer = _get_buffer(inp)
    realize_as_comm_buffer(buffer, "symm_mem")
    return pytree.tree_map(
        ir.TensorBox.create,
        ir.FallbackKernel.create(
            torch.ops.symm_mem.one_shot_all_reduce.default,
            inp,
            reduce_op,
            group_name,
        ),
    )


def register_comm_lowerings():
    c10d: Optional[torch._ops._OpNamespace] = None  # type: ignore[assignment]
    try:
        c10d = torch.ops._c10d_functional
    except (ImportError, AttributeError):
        log.info(
            "Inductor support for distributed collectives depends on building "
            "torch.distributed"
        )
        return

    assert c10d is not None

    from .lowering import clone, copy_, register_lowering

    @register_lowering(c10d.all_reduce)  # type: ignore[misc]
    def _all_reduce(inp: ir.TensorBox, reduce_op: str, group_name: str) -> ir.TensorBox:
        inp.realize()
        if _should_lower_as_one_shot_all_reduce(inp, reduce_op, group_name):
            return _one_shot_all_reduce(inp, reduce_op, group_name)

        # Lower as c10d.all_reduce_
        inp = clone(inp)
        if config.reorder_for_compute_comm_overlap:
            # The horizontal fusion of this clone often severely delays the
            # scheduling of the all_reduce_ node. Horizontally fusing this
            # clone can almost never out-perform scheduling the all_reduce_
            # earlier. Also in most cases, this clone is eliminated via
            # in-place reuse. Therefore, we tell the scheduler to not fuse it.
            inp.realize()
            V.graph.no_fuse_buffer_names.add(inp.get_name())
        inp = ir.ExternKernel.require_contiguous(inp)
        ir._CollectiveKernel.create_inplace(
            c10d.all_reduce_.default, inp, reduce_op, group_name
        )
        return inp

    @register_lowering(c10d.all_reduce_)  # type: ignore[misc]
    def _all_reduce_(
        inp: ir.TensorBox, reduce_op: str, group_name: str
    ) -> ir.TensorBox:
        inp.realize()
        if _should_lower_as_one_shot_all_reduce(inp, reduce_op, group_name):
            ret = copy_(
                inp,
                _one_shot_all_reduce(inp, reduce_op, group_name),
            )
            mark_as_skip_wait(ret)
            return inp

        # Lower as c10d.all_reduce_
        ir._CollectiveKernel.create_inplace(
            c10d.all_reduce_.default, inp, reduce_op, group_name
        )
        return inp

    @register_lowering(c10d.all_reduce_coalesced)
    def _all_reduce_coalesced(inputs, reduce_op, group_name):
        inputs = [clone(inp) for inp in inputs]
        ir._CollectiveKernel.create_inplace(
            c10d.all_reduce_coalesced_.default,
            inputs,
            reduce_op,
            group_name,
        )
        return inputs

    @register_lowering(c10d.all_reduce_coalesced_)
    def _all_reduce_coalesced_(inputs, reduce_op, group_name):
        ir._CollectiveKernel.create_inplace(
            c10d.all_reduce_coalesced_.default,
            inputs,
            reduce_op,
            group_name,
        )
        return inputs

    @register_lowering(c10d.all_gather_into_tensor)
    def _all_gather_into_tensor(inp, group_size, group_name):
        return ir.TensorBox.create(
            ir._CollectiveKernel.create_out_of_place(
                c10d.all_gather_into_tensor.default,
                inp,
                group_size,
                group_name,
            )
        )

    @register_lowering(c10d.all_gather_into_tensor_coalesced)
    def _all_gather_into_tensor_coalesced(inputs, group_size, group_name):
        return pytree.tree_map(
            ir.TensorBox.create,
            ir._CollectiveKernel.create_out_of_place(
                c10d.all_gather_into_tensor_coalesced.default,
                inputs,
                group_size,
                group_name,
            ),
        )

    @register_lowering(c10d.all_gather_into_tensor_out)
    def _all_gather_into_tensor_out(inp, group_size, group_name, *, out):
        ir._CollectiveKernel.create_inplace(
            c10d.all_gather_into_tensor_out.default,
            inp,
            group_size,
            group_name,
            out=out,
        )
        return out

    @register_lowering(c10d.reduce_scatter_tensor)
    def _reduce_scatter_tensor(inp, reduce_op, group_size, group_name):
        return ir.TensorBox.create(
            ir._CollectiveKernel.create_out_of_place(
                c10d.reduce_scatter_tensor.default,
                inp,
                reduce_op,
                group_size,
                group_name,
            )
        )

    @register_lowering(c10d.reduce_scatter_tensor_coalesced)
    def _reduce_scatter_tensor_coalesced(inputs, reduce_op, group_size, group_name):
        return pytree.tree_map(
            ir.TensorBox.create,
            ir._CollectiveKernel.create_out_of_place(
                c10d.reduce_scatter_tensor_coalesced.default,
                inputs,
                reduce_op,
                group_size,
                group_name,
            ),
        )

    @register_lowering(c10d.all_to_all_single)
    def _all_to_all_single(inp, output_split_sizes, input_split_sizes, group_name):
        return ir.TensorBox.create(
            ir._CollectiveKernel.create_out_of_place(
                c10d.all_to_all_single.default,
                inp,
                output_split_sizes,
                input_split_sizes,
                group_name,
            )
        )

    @register_lowering(c10d.broadcast)
    def _broadcast(inp, src, group_name):
        inp = clone(inp)
        ir._CollectiveKernel.create_inplace(
            c10d.broadcast_.default, inp, src, group_name
        )
        return inp

    @register_lowering(c10d.broadcast_)
    def _broadcast_(inp, src, group_name):
        ir._CollectiveKernel.create_inplace(
            c10d.broadcast_.default, inp, src, group_name
        )
        return inp

    @register_lowering(torch.ops._dtensor.shard_dim_alltoall)
    def _shard_dim_alltoall(inp, gather_dim, shard_dim, group_name):
        return ir.TensorBox.create(
            ir._CollectiveKernel.create_out_of_place(
                torch.ops._dtensor.shard_dim_alltoall.default,
                inp,
                gather_dim,
                shard_dim,
                group_name,
            )
        )

    @register_lowering(c10d.wait_tensor)
    def _wait_tensor(inp):
        if should_skip_wait(inp):
            return inp

        ir._WaitKernel.create_wait(c10d.wait_tensor.default, inp)
        return inp
