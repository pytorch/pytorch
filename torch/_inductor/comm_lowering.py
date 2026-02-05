# mypy: allow-untyped-defs
import logging

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
# (communication libraries like NCCL handle both registered and non-registered
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

    # We cannot realize buffers as comm buffers if we don't control their
    # allocation.
    if isinstance(data, ir.Buffer) and not data.should_allocate():
        return False

    layout = data.get_output_spec()
    if isinstance(layout, ir.CommBufferLayout):
        return True

    if isinstance(layout, ir.FixedLayout):
        return True

    if isinstance(layout, ir.FlexibleLayout) and not is_symbolic(data.get_numel()):
        return True

    return False


def realize_as_comm_buffer(
    x: ir.TensorBox,
    comm_buffer_type: ir.CommBufferType,
    group_name: "torch.distributed.distributed_c10d.GroupName",
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

    # The buffer may have already been frozen to FixedLayout if it was used
    # by another operation before the comm operation.
    if not isinstance(layout, (ir.FlexibleLayout, ir.FixedLayout)):
        raise AssertionError(
            "A buffer can only be realized as a comm buffer if it "
            f"has `FlexibleLayout` or `FixedLayout` (got {layout})."
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
        node = x.data.unwrap_view()
        assert isinstance(node, (ir.BaseView, ir.MutableBox))
        return node.data
    elif isinstance(x.data, ir.StorageBox):
        # TensorBox -> StorageBox -> IRNode
        return x.data.data
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
    inp: ir.TensorBox,
    reduce_op: str,
    group_name: "torch.distributed.distributed_c10d.GroupName",
):
    from torch.distributed._symmetric_memory import is_symm_mem_enabled_for_group

    inp_size = inp.get_numel() * inp.get_dtype().itemsize
    return (
        config._collective.auto_select
        and is_symm_mem_enabled_for_group(group_name)
        and can_realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM)
        and reduce_op == "sum"
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
    """
    Register lowerings for the comm subsystem.
    """
    try:
        torch.ops._c10d_functional.all_reduce
    except AttributeError:
        log.info(
            "Inductor support for distributed collectives depends on building "
            "torch.distributed"
        )
        return

    from .lowering import (
        add_layout_constraint,
        clone,
        constrain_to_fx_strides,
        copy_,
        register_lowering,
    )

    def register_comm_lowering(fn):
        add_layout_constraint(fn, constrain_to_fx_strides)
        return register_lowering(fn)

    c10d = torch.ops._c10d_functional

    @register_comm_lowering(c10d.all_reduce)  # type: ignore[misc]
    def _all_reduce(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: "torch.distributed.distributed_c10d.GroupName",
    ) -> ir.TensorBox:
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
        # pyrefly: ignore [bad-assignment]
        inp = ir.ExternKernel.require_contiguous(inp)
        # Because we are lowering as inplace c10d.all_reduce_, we should generate
        # _AllReduce_Kernel instead of _AllReduceKernel.
        ir._AllReduce_Kernel.create_inplace(
            c10d.all_reduce_.default,
            inp,  # type: ignore[arg-type]
            reduce_op,
            group_name,  # type: ignore[arg-type]
        )
        return inp  # type: ignore[return-value]

    @register_comm_lowering(c10d.all_reduce_)  # type: ignore[misc]
    def _all_reduce_(
        inp: ir.TensorBox,
        reduce_op: str,
        group_name: "torch.distributed.distributed_c10d.GroupName",
    ) -> ir.TensorBox:
        if _should_lower_as_one_shot_all_reduce(inp, reduce_op, group_name):
            ret = copy_(
                inp,
                _one_shot_all_reduce(inp, reduce_op, group_name),
            )
            mark_as_skip_wait(ret)
            return inp

        # Lower as c10d.all_reduce_
        # pyrefly: ignore [bad-assignment]
        inp = ir.ExternKernel.require_contiguous(inp)
        ir._AllReduce_Kernel.create_inplace(
            c10d.all_reduce_.default,
            inp,  # type: ignore[arg-type]
            reduce_op,
            group_name,  # type: ignore[arg-type]
        )
        return inp  # type: ignore[return-value]

    @register_comm_lowering(c10d.all_reduce_coalesced)
    def _all_reduce_coalesced(inputs, reduce_op, group_name):
        inputs = [clone(inp) for inp in inputs]
        ir._CollectiveKernel.create_inplace(
            c10d.all_reduce_coalesced_.default,
            inputs,
            reduce_op,
            group_name,
        )
        return inputs

    @register_comm_lowering(c10d.all_reduce_coalesced_)
    def _all_reduce_coalesced_(inputs, reduce_op, group_name):
        ir._CollectiveKernel.create_inplace(
            c10d.all_reduce_coalesced_.default,
            inputs,
            reduce_op,
            group_name,
        )
        return inputs

    def _create_out_of_place(kernel, inputs, *args) -> ir.IRNode:
        node = ir._CollectiveKernel.create_out_of_place(kernel, inputs, *args)
        assert isinstance(node, ir.IRNode)
        return ir.TensorBox.create(node)

    @register_comm_lowering(c10d.all_gather_into_tensor)
    def _all_gather_into_tensor(inp, group_size, group_name):
        return _create_out_of_place(
            c10d.all_gather_into_tensor.default,
            inp,
            group_size,
            group_name,
        )

    @register_comm_lowering(c10d.all_gather_into_tensor_coalesced)
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

    @register_comm_lowering(c10d.all_gather_into_tensor_out)
    def _all_gather_into_tensor_out(inp, group_size, group_name, *, out):
        ir._CollectiveKernel.create_inplace(
            c10d.all_gather_into_tensor_out.default,
            inp,
            group_size,
            group_name,
            out=out,
        )
        return out

    @register_comm_lowering(c10d.reduce_scatter_tensor)
    def _reduce_scatter_tensor(inp, reduce_op, group_size, group_name):
        return _create_out_of_place(
            c10d.reduce_scatter_tensor.default,
            inp,
            reduce_op,
            group_size,
            group_name,
        )

    @register_comm_lowering(c10d.reduce_scatter_tensor_out)
    def _reduce_scatter_tensor_out(inp, reduce_op, group_size, group_name, *, out):
        ir._CollectiveKernel.create_inplace(
            c10d.reduce_scatter_tensor_out.default,
            inp,
            reduce_op,
            group_size,
            group_name,
            out=out,
        )
        return out

    @register_comm_lowering(c10d.reduce_scatter_tensor_coalesced)
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

    @register_comm_lowering(c10d.all_to_all_single)
    def _all_to_all_single(inp, output_split_sizes, input_split_sizes, group_name):
        return _create_out_of_place(
            c10d.all_to_all_single.default,
            inp,
            output_split_sizes,
            input_split_sizes,
            group_name,
        )

    @register_comm_lowering(c10d.broadcast)
    def _broadcast(inp, src, group_name):
        inp = clone(inp)
        ir._CollectiveKernel.create_inplace(
            c10d.broadcast_.default, inp, src, group_name
        )
        return inp

    @register_comm_lowering(c10d.broadcast_)
    def _broadcast_(inp, src, group_name):
        ir._CollectiveKernel.create_inplace(
            c10d.broadcast_.default, inp, src, group_name
        )
        return inp

    @register_comm_lowering(torch.ops._dtensor.shard_dim_alltoall)
    def _shard_dim_alltoall(inp, gather_dim, shard_dim, group_name):
        return _create_out_of_place(
            torch.ops._dtensor.shard_dim_alltoall.default,
            inp,
            gather_dim,
            shard_dim,
            group_name,
        )

    @register_comm_lowering(c10d.wait_tensor)
    def _wait_tensor(inp):
        if should_skip_wait(inp):
            return inp

        ir._WaitKernel.create_wait(c10d.wait_tensor.default, inp)
        return inp


def register_symm_mem_lowerings():
    """
    Register lowerings for symmetric memory (symm_mem) operations.

    This function automatically registers lowerings for all operations that have
    symm_mem args metadata registered via Library.register_symm_mem_args().
    """
    try:
        symm_mem = torch.ops.symm_mem
        # Check for an actual operation, not just the namespace.
        # torch.ops.symm_mem is a lazy namespace that always exists,
        # but the operations may not exist on non-CUDA platforms or
        # when USE_DISTRIBUTED is disabled.
        symm_mem.one_shot_all_reduce
    except AttributeError:
        log.info("symm_mem ops not available, skipping symm_mem lowerings")
        return

    from torch._library.simple_registry import singleton

    from .lowering import register_lowering

    def _maybe_realize_symm_mem(
        inp: ir.TensorBox,
        group_name: str,  # type: ignore[arg-type]
    ) -> None:
        """
        Helper to realize an input as symmetric memory buffer if possible.
        """
        if can_realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM):
            realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM, group_name)  # type: ignore[arg-type]
        else:
            log.warning(
                "Failed to realize the input as a symmetric memory buffer for symm_mem operation; "
                "ensure the input is allocated as a symmetric memory buffer."
            )

    def _get_mutated_return_arg(schema):
        """
        For mutable ops that return an aliased tensor, find which argument is returned.

        Returns:
            Tuple of (arg_name, arg_index) for the argument that is returned, or None
            if the op doesn't return an aliased tensor.
        """
        if not schema.is_mutable:
            return None

        if len(schema.returns) == 0:
            return None

        if len(schema.returns) == 1:
            ret = schema.returns[0]
            if ret.alias_info is None:
                return None

            ret_alias_set = ret.alias_info.after_set
            for i, arg in enumerate(schema.arguments):
                if arg.alias_info is not None and arg.alias_info.is_write:
                    arg_alias_set = arg.alias_info.after_set
                    if ret_alias_set == arg_alias_set:
                        return (arg.name, i)

        return None

    def _create_symm_mem_lowering(op, symm_mem_args_set, group_arg_name="group_name"):
        """
        Create a lowering function for an operator with symm_mem args.

        Args:
            op: The operator to create lowering for
            symm_mem_args_set: Set of argument names that require symmetric memory
            group_arg_name: Name of the group_name argument (default: "group_name")

        Returns:
            Lowering function that realizes symm_mem args and calls the operator
        """
        schema = op._schema
        is_mutable = schema.is_mutable
        mutated_return = _get_mutated_return_arg(schema) if is_mutable else None

        def lowering_fn(*args, **kwargs):
            arg_names = [arg.name for arg in schema.arguments]

            all_args = {}
            for i, arg_value in enumerate(args):
                if i < len(arg_names):
                    all_args[arg_names[i]] = arg_value
            all_args.update(kwargs)

            group_name = all_args.get(group_arg_name)

            for arg_name in symm_mem_args_set:
                arg_value = all_args.get(arg_name)
                if isinstance(arg_value, ir.TensorBox):
                    if group_name is not None:
                        _maybe_realize_symm_mem(arg_value, group_name)
                elif isinstance(arg_value, (list, tuple)):
                    for item in arg_value:
                        if isinstance(item, ir.TensorBox) and group_name is not None:
                            _maybe_realize_symm_mem(item, group_name)

            if mutated_return is not None:
                # Inplace op: use _CollectiveKernel.create_inplace and return mutated arg
                arg_name, arg_idx = mutated_return
                mutated_arg = all_args.get(arg_name)
                if mutated_arg is None and arg_idx < len(args):
                    mutated_arg = args[arg_idx]

                ir._CollectiveKernel.create_inplace(op, *args, **kwargs)
                return mutated_arg
            else:
                # Non-mutating op: use FallbackKernel
                result = ir.FallbackKernel.create(op, *args, **kwargs)
                return pytree.tree_map(ir.TensorBox.create, result)

        return lowering_fn

    # Auto-register all symm_mem operations from the registry
    registered_count = 0
    for qualname, entry in singleton._data.items():
        if not qualname.startswith("symm_mem::"):
            continue

        symm_mem_args = entry.symm_mem_args.get()
        if symm_mem_args is None:
            continue

        try:
            op = torch._library.utils.lookup_op(qualname)

            # Create and register the lowering
            lowering_fn = _create_symm_mem_lowering(op, symm_mem_args)
            register_lowering(op)(lowering_fn)
            registered_count += 1
            log.debug(
                "Auto-registered lowering for %s with symm_mem args: %s",
                qualname,
                symm_mem_args,
            )

        except (AttributeError, ValueError):
            log.warning("Could not register lowering for %s", qualname)
            continue

    log.info(
        "Automatically registered %d symm_mem operation lowerings", registered_count
    )
