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

# Intentional FlyDSL 0.3.1 unstable boundary: no stable API exposes nonzero
# scheduler groups, a scheduler fence, or native exp2. Keep those calls
# centralized here so the kernel otherwise uses the stable surface.


def make_global_view(tensor, offset, shape, stride):
    layout = fx.make_layout(shape, stride)
    iterator = fx.get_iter(tensor)
    if offset is None:
        return fx.rocdl.make_buffer_tensor(fx.make_view(iterator, layout))

    # AMD buffer-resource offsets are 32-bit. Rebase the raw 64-bit pointer
    # before constructing the descriptor so only this CTA-local view must fit
    # in the descriptor's addressable range.
    iterator = fx.add_offset(iterator, fx.Int64(offset))
    return fx.rocdl.make_buffer_tensor(fx.make_view(iterator, layout))


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


def _schedule_group(kind: str, count: int, group: int):
    fx.rocdl.sched_group_barrier(kind, count, group)


def schedule_fence():
    fx.rocdl.sched_barrier(0)


def fast_exp2(value):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, value.ir_value()))


def schedule_fwd_qk_pipeline(*, reduction_steps: int, vmem_count: int = 0):
    """Interleave Q/K LDS reads and optional next-tile VMEM with QK MFMAs."""
    dsrd_preload = min(6, 3 * reduction_steps)
    fx.rocdl.sched_dsrd(dsrd_preload)
    scheduled_vmem = 0
    for step in fx.range_constexpr(reduction_steps):
        target_vmem = ((step + 1) * vmem_count) // reduction_steps
        if const_expr(target_vmem > scheduled_vmem):
            fx.rocdl.sched_vmem(target_vmem - scheduled_vmem)
            scheduled_vmem = target_vmem
        fx.rocdl.sched_mfma(2)
        if const_expr(step + 2 < reduction_steps):
            fx.rocdl.sched_dsrd(3)
    schedule_fence()


def schedule_fwd_softmax_pipeline(*, vmem_count: int):
    """Spread V loads across four groups of exponentiation work."""
    slots = 4
    vmem_per_slot = vmem_count // slots
    vmem_remainder = vmem_count % slots
    for slot in fx.range_constexpr(slots):
        scheduled_vmem = vmem_per_slot + int(slot < vmem_remainder)
        if const_expr(scheduled_vmem):
            _schedule_group("vmem_read", scheduled_vmem, 1)
        _schedule_group("transcendental", 8, 1)
    schedule_fence()


def schedule_fwd_pv_pipeline(*, output_chunks: int):
    """Keep two V LDS reads ahead of each output MFMA."""
    mfma_count = 4 * output_chunks
    dsrd_count = 2 * mfma_count
    dsrd_preload = min(4, dsrd_count)
    fx.rocdl.sched_dsrd(dsrd_preload)
    remaining_reads = dsrd_count - dsrd_preload
    for mfma_index in fx.range_constexpr(mfma_count):
        if const_expr(2 * mfma_index < remaining_reads):
            fx.rocdl.sched_dsrd(min(2, remaining_reads - 2 * mfma_index))
        fx.rocdl.sched_mfma(1)
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
