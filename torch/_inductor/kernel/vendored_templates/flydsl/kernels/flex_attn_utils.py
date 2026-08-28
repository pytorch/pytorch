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

# FlyDSL 0.3 lacks stable grouped scheduling, fence-free barriers, and native exp2.
_SCHED_MFMA, _SCHED_VMEM_READ, _SCHED_LDS_READ, _SCHED_EXP = (0x008, 0x020, 0x100, 0x400)


def make_global_view(tensor, offset, shape, stride):
    iterator = fx.get_iter(fx.rocdl.make_buffer_tensor(tensor))
    if offset is not None:
        iterator = fx.add_offset(iterator, offset)
    return fx.make_view(iterator, fx.make_layout(shape, stride))


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
                    values.append(lhs % rhs)
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


def _schedule_group(mask: int, count: int, group: int):
    fx.rocdl.sched_group_barrier(mask, count, group)


def schedule_fence():
    fx.rocdl.sched_barrier(0)


def scheduled_workgroup_barrier():
    schedule_fence()
    fx.rocdl.s_barrier()


def fast_exp2(value):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, value.ir_value()))


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
            _schedule_group(_SCHED_VMEM_READ, vmem_per_slot, 1)
        _schedule_group(_SCHED_LDS_READ, dsrd_per_slot, 1)
        _schedule_group(_SCHED_EXP, exp_per_slot, 1)


def schedule_pack1_pipeline(*, mfma_count: int, exp_count: int, dsrd_count: int):
    exp_per_mfma = exp_count // mfma_count
    dsrd_per_operand = dsrd_count // 2
    for mfma_index in fx.range_constexpr(mfma_count):
        _schedule_group(_SCHED_MFMA, 1, 2)
        if const_expr(mfma_index < 2):
            _schedule_group(_SCHED_LDS_READ, dsrd_per_operand, 2)
        _schedule_group(_SCHED_EXP, exp_per_mfma, 2)


def schedule_update_tail(*, mfma_count: int, dsrd_count: int):
    dsrd_per_operand = dsrd_count // 2
    for mfma_index in fx.range_constexpr(mfma_count):
        if const_expr(mfma_index < 2):
            _schedule_group(_SCHED_LDS_READ, dsrd_per_operand, 3)
        _schedule_group(_SCHED_MFMA, 1, 3)
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


def make_mfma32_ops(window_mask, vector_width):
    g128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
    dma128 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
    tr16 = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(16, 64), fx.BFloat16)
    o64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
    lds_pointer_type = fx.PointerType.get(fx.BFloat16.ir_type, 2, 16)
    atom = fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16))

    def gload_f32(view, index):
        return fx.Float32(fx.get_iter(view)[index])

    def gload_i32(view, index):
        return fx.Int32(fx.get_iter(view)[index])

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
