# mypy: allow-untyped-defs
import logging

import torch
import torch.utils._pytree as pytree

from . import config, ir
from .virtualized import V

log = logging.getLogger(__name__)
try:
    from torch.distributed._symmetric_memory import is_symm_mem_enabled_for_group

    _c10d_functional = torch.ops._c10d_functional
except (ImportError, AttributeError):
    log.info(
        "Inductor support for distributed collectives depends on building "
        "torch.distributed"
    )


def can_realize_as_reg_comm_buffer(buffer: ir.Buffer) -> bool:
    """
    Check if a buffer can be realized as a registered comm buffer.
    """
    assert isinstance(buffer, ir.Buffer)

    if buffer.get_name() in V.graph.buffer_reg_type:
        return True

    try:
        V.graph.sizevars.size_hint(buffer.get_numel())
    except Exception:
        return False

    if isinstance(
        buffer,
        (ir.InputBuffer, ir.ExternKernelAlloc, ir.MultiOutput),
    ):
        return False

    layout = buffer.get_layout()
    if isinstance(
        layout,
        (ir.MutationLayoutSHOULDREMOVE, ir.NonOwningLayout),
    ):
        return False

    return True


def realize_as_reg_comm_buffer(buffer, reg_type: str):
    """
    Realize a buffer as registered comm buffer.
    """
    buffer.realize()
    V.graph.buffer_reg_type[buffer.get_name()] = reg_type
    V.graph.never_reuse_buffers.add(buffer.get_name())
    print(V.graph.buffer_reg_type)


def _symm_mem_one_shot_all_reduce(inp, reduce_op, group_name):
    assert is_symm_mem_enabled_for_group(group_name)
    realize_as_reg_comm_buffer(inp, "symm_mem")
    return pytree.tree_map(
        ir.TensorBox.create,
        ir.FallbackKernel.create(
            torch.ops.symm_mem.one_shot_all_reduce.default,
            inp,
            reduce_op,
            group_name,
        ),
    )


ONE_SHOT_ALL_REDUCE_SIZE_THRESHOLD = 256 * 1024


def all_reduce(inp, reduce_op, group_name):
    inp.realize()
    buffer = inp.data.data
    if can_realize_as_reg_comm_buffer(buffer):
        inp_size = buffer.get_numel() * buffer.get_dtype().itemsize
        if (
            config._register_comm_buffers
            and is_symm_mem_enabled_for_group(group_name)
            and inp_size <= ONE_SHOT_ALL_REDUCE_SIZE_THRESHOLD
        ):
            return _symm_mem_one_shot_all_reduce(inp, reduce_op, group_name)
    from .lowering import clone

    # TODO: implement NCCL registered comm
    inp = clone(inp)
    ir._CollectiveKernel.create_inplace(
        _c10d_functional.all_reduce_.default, inp, reduce_op, group_name
    )
    return inp


def all_reduce_(inp, reduce_op, group_name):
    inp.realize()
    buffer = inp.data.data
    if can_realize_as_reg_comm_buffer(buffer):
        inp_size = buffer.get_numel() * buffer.get_dtype().itemsize
        if (
            config._register_comm_buffers
            and is_symm_mem_enabled_for_group(group_name)
            and inp_size <= ONE_SHOT_ALL_REDUCE_SIZE_THRESHOLD
        ):
            from .lowering import copy_

            ret = copy_(
                inp,
                _symm_mem_one_shot_all_reduce(inp, reduce_op, group_name),
            )
            ir.mark_node_as_mutating(ret, inp)
            V.graph.skip_wait.add(ret.get_name())
            return inp

    # TODO: implement NCCL registered comm
    ir._CollectiveKernel.create_inplace(
        _c10d_functional.all_reduce_.default, inp, reduce_op, group_name
    )
    return inp
