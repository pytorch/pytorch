"""Shared register, LDS, mask, and MFMA helpers for FlexAttention backward."""

import flydsl.expr as fx
from flydsl.expr import const_expr

from .flex_attn_mask import evaluate_mask_program
from .flex_attn_utils import load_scalar


def schedule_score_pipeline(*, mfma_count: int, dsrd_count: int, vmem_count: int):
    """Overlap next-tile DMA with the current tile's score MFMAs."""
    mfma_group = 2
    groups = mfma_count // mfma_group
    dsrd_preload = min(4 + dsrd_count % 2, dsrd_count)
    dsrd_remaining = dsrd_count - dsrd_preload
    dsrd_groups = dsrd_remaining // 2
    vmem_group_count = min(vmem_count, groups)
    fx.rocdl.sched_dsrd(dsrd_preload)
    for group in fx.range_constexpr(groups):
        if const_expr(group < vmem_group_count):
            fx.rocdl.sched_vmem(1)
        fx.rocdl.sched_mfma(mfma_group)
        if const_expr(group < dsrd_groups):
            fx.rocdl.sched_dsrd(2)
    fx.rocdl.sched_barrier(0)


def schedule_pack0_pipeline(*, vmem_count: int, exp_count: int, dsrd_count: int):
    """Prime the update pipeline while producing the first BF16 dS/P pack."""
    slots = 4
    vmem_per_slot = vmem_count // slots
    exp_per_slot = exp_count // slots
    dsrd_per_slot = dsrd_count // slots
    for _ in fx.range_constexpr(slots):
        if const_expr(vmem_per_slot):
            fx.rocdl.sched_group_barrier(fx.rocdl.mask_vmem_rd, vmem_per_slot, 1)
        fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, dsrd_per_slot, 1)
        fx.rocdl.sched_group_barrier(1024, exp_per_slot, 1)


def schedule_pack1_pipeline(*, mfma_count: int, exp_count: int, dsrd_count: int):
    """Consume pack 0 while building pack 1 and priming two operands."""
    exp_per_mfma = exp_count // mfma_count
    dsrd_per_operand = dsrd_count // 2
    for mfma_index in fx.range_constexpr(mfma_count):
        fx.rocdl.sched_group_barrier(fx.rocdl.mask_mfma, 1, 2)
        if const_expr(mfma_index < 2):
            fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, dsrd_per_operand, 2)
        fx.rocdl.sched_group_barrier(1024, exp_per_mfma, 2)


def schedule_update_tail(*, mfma_count: int, dsrd_count: int):
    """Consume pack 1 with a two-operand LDS-read look-ahead."""
    dsrd_per_operand = dsrd_count // 2
    for mfma_index in fx.range_constexpr(mfma_count):
        if const_expr(mfma_index < 2):
            fx.rocdl.sched_group_barrier(fx.rocdl.mask_dsrd, dsrd_per_operand, 3)
        fx.rocdl.sched_group_barrier(fx.rocdl.mask_mfma, 1, 3)
    fx.rocdl.sched_barrier(0)


def make_mask_buffers(gview, count, sizes, buffer0, buffer1, buffer2, buffer3):
    buffers = []
    if const_expr(count >= 1):
        buffers.append(gview(buffer0, None, sizes[0], 1))
    if const_expr(count >= 2):
        buffers.append(gview(buffer1, None, sizes[1], 1))
    if const_expr(count >= 3):
        buffers.append(gview(buffer2, None, sizes[2], 1))
    if const_expr(count >= 4):
        buffers.append(gview(buffer3, None, sizes[3], 1))
    return buffers


def make_mask_evaluator(program, output, strides, buffers, load_i32, batch, head):
    def evaluate(q_pos, kv_pos):
        return evaluate_mask_program(
            mask_program=program,
            mask_program_output=output,
            mask_buffer_strides=strides,
            mask_buffers=buffers,
            load_i32=load_i32,
            batch=batch,
            head=head,
            q_pos=q_pos,
            kv_pos=kv_pos,
        )

    return evaluate


def make_mfma32_ops(window_mask, vector_width):
    g128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    dma128 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    tr16 = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(16, 64), fx.BFloat16)
    f32atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    i32atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    o64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
    lds_pointer_type = fx.PointerType.get(fx.BFloat16.ir_type, 2, 16)
    atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16))

    def gload_f32(view, index):
        return fx.Float32(load_scalar(f32atom, view, index, fx.Float32))

    def gload_i32(view, index):
        return fx.Int32(load_scalar(i32atom, view, index, fx.Int32))

    def load_global_pack(view, row, column):
        fragment = fx.make_rmem_tensor(vector_width, fx.BFloat16)
        source = fx.logical_divide(fx.slice(view, (row, None)), fx.make_layout(vector_width, 1))
        fx.copy(g128, fx.slice(source, (None, column // fx.Int32(vector_width))), fragment)
        return fragment.load()

    def make_fragment(value, size, dtype):
        fragment = fx.make_rmem_tensor(size, dtype)
        fragment.store(fx.Vector(value).ir_value())
        return fragment

    def mfma(a_value, b_value, c_value):
        a_fragment = make_fragment(a_value, 8, fx.BFloat16)
        b_fragment = make_fragment(b_value, 8, fx.BFloat16)
        c_fragment = make_fragment(c_value, 16, fx.Float32)
        fx.gemm(atom, c_fragment, a_fragment, b_fragment, c_fragment)
        return c_fragment.load()

    def b_operand_column(row_group, column):
        if const_expr(window_mask):
            return column
        group = fx.Int32(row_group)
        return column ^ ((group & fx.Int32(1)) << fx.Int32(4) | (group & fx.Int32(2)) << fx.Int32(2))

    def dma_destination(base, element_offset):
        pointer = fx.inttoptr(
            lds_pointer_type,
            fx.Int32(fx.ptrtoint(fx.add_offset(base, fx.make_int_tuple(element_offset)))),
        )
        return fx.make_view(pointer, fx.make_layout(1, 1))

    return (
        dma128,
        tr16,
        o64,
        atom,
        gload_f32,
        gload_i32,
        load_global_pack,
        make_fragment,
        mfma,
        b_operand_column,
        dma_destination,
    )
