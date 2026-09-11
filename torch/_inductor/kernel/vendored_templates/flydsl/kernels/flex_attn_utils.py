# mypy: allow-untyped-defs

import flydsl.expr as fx
from flydsl.expr import const_expr

_CAUSAL_DOCUMENT_MASK_PROGRAM = (
    ("const_bool", True),
    ("ge", 2, 3),
    ("and", 4, 5),
    ("load_i32", 0, (2,)),
    ("load_i32", 1, (7,)),
    ("ge", 3, 8),
    ("and", 6, 9),
)

# FlyDSL 0.3 lacks stable grouped scheduling and fence-free barriers.
_SCHED_GROUP_MASKS = {
    "mfma": 0x008,
    "vmem_read": 0x020,
    "lds_read": 0x100,
    "transcendental": 0x400,
}


def make_global_view(tensor, coord, shape, stride):
    view = fx.make_view(fx.get_iter(tensor), fx.make_layout(shape, stride))
    if coord is not None:
        # Slice the global pointer before creating the 32-bit buffer descriptor.
        # Callers widen batch/head coordinates to i64; local strides stay static.
        view = fx.slice(view, coord)
    num_records_bytes = (
        fx.get_scalar(fx.cosize(view.layout)) * view.element_type.width + 7
    ) // 8
    if not 0 <= num_records_bytes <= 0xFFFFFFFF:
        raise ValueError("FlyDSL buffer view must fit in a 32-bit byte range")
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


def make_metadata_view(tensor, offset, local_size):
    # These compact worklists are intentionally rebased inside their 32-bit
    # descriptors; large Q/K/V tensors use make_global_view instead.
    iterator = fx.get_iter(fx.rocdl.make_buffer_tensor(tensor))
    return fx.make_view(
        fx.add_offset(iterator, fx.Int32(offset)),
        fx.make_layout(local_size, 1),
    )


def make_qk_shared_layout(rows, columns):
    layout = fx.make_layout(
        ((32, rows // 32), (32, columns // 32)),
        ((32, 32 * columns), (1, 32 * 32)),
    )
    # Swizzle eight-element packs without changing their 16-byte alignment.
    return fx.make_composed_layout(fx.static(fx.SwizzleType.get(2, 3, 5)), layout)


def make_value_shared_layout(rows, columns):
    return fx.make_layout(
        ((8, rows // 8), (32, columns // 32)),
        ((32, 8 * columns), (1, 8 * 32)),
    )


def make_shared_view(pointer, shape, stride):
    return fx.make_view(pointer, fx.make_layout(shape, stride))


def is_causal_document_mask_program(
    mask_program,
    mask_program_output,
    mask_buffer_strides,
):
    return (
        tuple(mask_program) == _CAUSAL_DOCUMENT_MASK_PROGRAM
        and int(mask_program_output) == 10
        and len(mask_buffer_strides) == 2
        and all(len(strides) == 1 for strides in mask_buffer_strides)
    )


def evaluate_mask_program(
    *,
    mask_program,
    mask_program_output,
    mask_buffer_strides,
    mask_buffers,
    load_i32,
    batch,
    head,
    q_pos,
    kv_pos,
):
    values = [fx.Int32(batch), fx.Int32(head), q_pos, kv_pos]
    for instruction in mask_program:
        op = instruction[0]
        if op == "const_i32":
            values.append(fx.Int32(instruction[1]))
        elif op == "const_bool":
            constant = fx.Int32(1 if instruction[1] else 0)
            values.append(constant == fx.Int32(1))
        elif op == "load_i32":
            buffer_index = instruction[1]
            index_ids = instruction[2]
            offset = fx.Int32(0)
            for dimension in fx.range_constexpr(len(index_ids)):
                offset = offset + values[index_ids[dimension]] * fx.Int32(
                    mask_buffer_strides[buffer_index][dimension]
                )
            values.append(load_i32(mask_buffers[buffer_index], offset))
        else:
            lhs = values[instruction[1]]
            if op == "not":
                values.append(~lhs)
            else:
                rhs = values[instruction[2]]
                if op == "add":
                    values.append(lhs + rhs)
                elif op == "sub":
                    values.append(lhs - rhs)
                elif op == "mul":
                    values.append(lhs * rhs)
                elif op == "floordiv":
                    values.append(lhs // rhs)
                elif op == "remainder":
                    remainder = lhs % rhs
                    needs_adjustment = (remainder != 0) & ((remainder < 0) != (rhs < 0))
                    values.append(needs_adjustment.select(remainder + rhs, remainder))
                elif op == "ge":
                    values.append(lhs >= rhs)
                elif op == "gt":
                    values.append(lhs > rhs)
                elif op == "le":
                    values.append(lhs <= rhs)
                elif op == "lt":
                    values.append(lhs < rhs)
                elif op == "eq":
                    values.append(lhs == rhs)
                elif op == "ne":
                    values.append(lhs != rhs)
                elif op == "and":
                    values.append(lhs & rhs)
                elif op == "or":
                    values.append(lhs | rhs)
                else:
                    raise ValueError(f"unsupported mask bytecode op {op}")
    return values[mask_program_output]


def _schedule_group(kind: str, count: int, group: int):
    fx.rocdl.sched_group_barrier(_SCHED_GROUP_MASKS[kind], count, group)


def schedule_fence():
    fx.rocdl.sched_barrier(0)


def scheduled_workgroup_barrier():
    schedule_fence()
    fx.rocdl.s_barrier()


def fast_exp2(value):
    return fx.math.exp2(fx.Float32(value), fastmath=fx.FastMathFlags.afn)


def schedule_score_pipeline(*, mfma_count: int, dsrd_count: int, vmem_count: int):
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
    schedule_fence()


def schedule_pack0_pipeline(*, vmem_count: int, exp_count: int, dsrd_count: int):
    slots = 4
    vmem_per_slot = vmem_count // slots
    exp_per_slot = exp_count // slots
    dsrd_per_slot = dsrd_count // slots
    for _ in fx.range_constexpr(slots):
        if const_expr(vmem_per_slot):
            _schedule_group("vmem_read", vmem_per_slot, 1)
        _schedule_group("lds_read", dsrd_per_slot, 1)
        _schedule_group("transcendental", exp_per_slot, 1)


def schedule_pack1_pipeline(*, mfma_count: int, exp_count: int, dsrd_count: int):
    exp_per_mfma = exp_count // mfma_count
    dsrd_per_operand = dsrd_count // 2
    for mfma_index in fx.range_constexpr(mfma_count):
        _schedule_group("mfma", 1, 2)
        if const_expr(mfma_index < 2):
            _schedule_group("lds_read", dsrd_per_operand, 2)
        _schedule_group("transcendental", exp_per_mfma, 2)


def schedule_update_tail(*, mfma_count: int, dsrd_count: int):
    dsrd_per_operand = dsrd_count // 2
    for mfma_index in fx.range_constexpr(mfma_count):
        if const_expr(mfma_index < 2):
            _schedule_group("lds_read", dsrd_per_operand, 3)
        _schedule_group("mfma", 1, 3)
    schedule_fence()


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


def make_mfma32_ops(lane, vector_width):
    g128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    dma128 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    tr16 = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(16, 64), fx.BFloat16)
    o64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16))
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((1, 1, 1), (1, 1, 1)),
    )
    thread_mma = tiled_mma.get_slice(lane)
    accumulator_coordinates = thread_mma.partition_C(
        fx.make_view(0, fx.make_layout((32, 32), (1, 0)))
    )
    row_coordinates = thread_mma.partition_C(
        fx.make_view(0, fx.make_layout((32, 32), (0, 1)))
    )
    b_coordinates = thread_mma.partition_B(
        fx.make_view(0, fx.make_layout((32, 16), (16, 1)))
    )

    def gload_f32(view, index):
        return fx.Float32(fx.get_iter(view)[index])

    def gload_i32(view, index):
        return fx.Int32(fx.get_iter(view)[index])

    def load_global_pack(view, row, column):
        fragment = fx.make_rmem_tensor(vector_width, fx.BFloat16)
        source = fx.logical_divide(
            fx.slice(view, (row, None)), fx.make_layout(vector_width, 1)
        )
        fx.copy(
            g128, fx.slice(source, (None, column // fx.Int32(vector_width))), fragment
        )
        return fragment.load()

    def make_b_fragment(value):
        fragment = fx.make_fragment_like(b_coordinates, fx.BFloat16)
        fragment.store(fx.Vector(value).ir_value())
        return fragment

    return (
        dma128,
        tr16,
        o64,
        tiled_mma,
        thread_mma,
        accumulator_coordinates,
        row_coordinates,
        gload_f32,
        gload_i32,
        load_global_pack,
        make_b_fragment,
    )
