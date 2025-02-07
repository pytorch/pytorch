# mypy: allow-untyped-defs
import logging
from typing import cast

import torch
import torch.utils._pytree as pytree
from torch._inductor.utils import is_symbolic
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
def can_realize_as_comm_buffer(
    x: ir.TensorBox, comm_buffer_type: ir.CommBufferType
) -> bool:
    """
    Check if an input can be realized as a comm buffer of the specified
    `comm_buffer_type`.
    """
    data = _get_data(x)

    if isinstance(data, ir.Loops):
        return True

    layout = data.get_output_spec()
    if isinstance(layout, ir.CommBufferLayout):
        return True

    if isinstance(layout, ir.FlexibleLayout) and not is_symbolic(data.get_numel()):
        return True

    return False


def realize_as_comm_buffer(
    x: ir.TensorBox, comm_buffer_type: ir.CommBufferType, group_name: str
) -> None:
    """
    Realize an input as a comm buffer of the specified `comm_buffer_type`.

    Specifically, this realizes the underlying buffer if it's still unrealized
    and changes the layout of the buffer to `ir.CommBufferLayout`.
    """
    x.realize()
    buffer = _get_data(x)
    assert isinstance(buffer, ir.Buffer)

    layout = buffer.get_output_spec()
    if isinstance(layout, ir.CommBufferLayout):
        return

    if not isinstance(layout, ir.FlexibleLayout):
        raise AssertionError(
            "A buffer can only be realized as a comm buffer if it "
            f"has `FlexibleLayout` (got {layout})."
        )

    if is_symbolic(buffer.get_numel()):
        raise AssertionError(
            "A buffer with symbolic shape cannot be converted to "
            f"a comm buffer (got {layout})."
        )

    buffer.layout = ir.CommBufferLayout(
        layout=layout,
        comm_buffer_type=comm_buffer_type,
        group_name=group_name,
    )


def _get_data(x: ir.TensorBox) -> ir.IRNode:
    if isinstance(x.data, ir.BaseView):
        # TensorBox -> *View -> StorageBox -> IRNode
        return x.data.unwrap_view().data
    elif isinstance(x.data, ir.StorageBox):
        # TensorBox -> StorageBox -> IRNode
        return cast(ir.Buffer, x.data.data)
    else:
        raise AssertionError(
            "Expect the data attr of a `TensorBox` to be either "
            f"an `ir.BaseView` or `ir.StorageBox` (got {x.data})."
        )


_bufs_to_skip_wait = OrderedSet[tuple[int, str]]()


def mark_as_skip_wait(x: ir.IRNode) -> None:
    """
    If a non-blocking collective is lowered as a blocking collective, the wait
    node in the original graph becomes useless and we can skip the lowering it.
    """
    _bufs_to_skip_wait.add((id(V.graph), x.get_name()))


def should_skip_wait(x: ir.IRNode) -> bool:
    return (id(V.graph), x.get_name()) in _bufs_to_skip_wait


def _should_lower_as_one_shot_all_reduce(
    inp: ir.TensorBox, reduce_op: str, group_name: str
):
    from torch.distributed._symmetric_memory import is_symm_mem_enabled_for_group

    inp_size = inp.get_numel() * inp.get_dtype().itemsize
    return (
        config._collective.auto_select
        and is_symm_mem_enabled_for_group(group_name)
        and can_realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM)
        and reduce_op in ("sum",)
        and inp_size <= config._collective.one_shot_all_reduce_threshold_bytes
    )


def _one_shot_all_reduce(inp: ir.TensorBox, reduce_op, group_name):
    realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM, group_name)
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
    try:
        torch.ops._c10d_functional.all_reduce
    except AttributeError:
        log.info(
            "Inductor support for distributed collectives depends on building "
            "torch.distributed"
        )
        return

    from .lowering import clone, copy_, register_lowering

    c10d = torch.ops._c10d_functional

    @register_lowering(c10d.all_reduce)  # type: ignore[misc]
    def _all_reduce(inp: ir.TensorBox, reduce_op: str, group_name: str) -> ir.TensorBox:
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
        if _should_lower_as_one_shot_all_reduce(inp, reduce_op, group_name):
            ret = copy_(
                inp,
                _one_shot_all_reduce(inp, reduce_op, group_name),
            )
            mark_as_skip_wait(ret)
            return inp

        # Lower as c10d.all_reduce_
        inp = ir.ExternKernel.require_contiguous(inp)
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
