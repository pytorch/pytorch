# mypy: allow-untyped-defs

import flydsl.expr as fx
from flydsl.expr import const_expr


# and_masks() inserts a True seed; the plain lambda has no seed.
_SLIDING_WINDOW_MASK_PROGRAMS = (
    (
        (
            ("const_bool", True),
            ("ge", 2, 3),
            ("and", 4, 5),
            ("sub", 2, 3),
            ("const_i32", None),
            ("lt", 7, 8),
            ("and", 6, 9),
        ),
        4,
        10,
    ),
    (
        (("ge", 2, 3), ("sub", 2, 3), ("const_i32", None), ("lt", 5, 6), ("and", 4, 7)),
        2,
        8,
    ),
)
_SCHED_GROUP_MASKS = {"vmem_read": 0x020, "transcendental": 0x400}


def make_global_view(tensor, coordinates, shape, stride):
    view = fx.make_view(fx.get_iter(tensor), fx.make_layout(shape, stride))
    if coordinates is not None:
        # Slice the global pointer before creating the 32-bit buffer descriptor.
        # Callers widen batch/head coordinates to i64; local strides stay static.
        view = fx.slice(view, coordinates)
    byte_count = (
        fx.get_scalar(fx.cosize(view.layout)) * view.element_type.width + 7
    ) // 8
    if not 0 <= byte_count <= 0xFFFFFFFF:
        raise ValueError("FlyDSL buffer view must fit in a 32-bit byte range")
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=byte_count)


def make_qk_shared_layout(rows, columns):
    layout = fx.make_layout(
        ((32, rows // 32), (32, columns // 32)), ((32, 32 * columns), (1, 32 * 32))
    )
    # Swizzle eight-element packs without changing their 16-byte alignment.
    return fx.make_composed_layout(fx.static(fx.SwizzleType.get(2, 3, 5)), layout)


def make_value_shared_layout(rows, columns):
    return fx.make_layout(
        ((8, rows // 8), (32, columns // 32)), ((32, 8 * columns), (1, 8 * 32))
    )


def is_sliding_window_mask_program(program, output, strides):
    """Recognize a causal mask bounded by a fixed lookback window.

    Such a mask gives every query block the same amount of work, so the query-block
    dispatch reversal that helps triangular masks only costs locality here.
    """
    if strides:
        return False
    program = tuple(program)
    output = int(output)
    for pattern, width_slot, pattern_output in _SLIDING_WINDOW_MASK_PROGRAMS:
        if len(program) != len(pattern) or output != pattern_output:
            continue
        width = program[width_slot]
        if len(width) != 2 or width[0] != "const_i32":
            continue
        if not isinstance(width[1], int) or isinstance(width[1], bool) or width[1] <= 0:
            continue
        if all(
            actual == expected
            for slot, (actual, expected) in enumerate(zip(program, pattern))
            if slot != width_slot
        ):
            return True
    return False


def evaluate_mask_program(
    program,
    output,
    strides,
    buffers,
    load_i32,
    batch,
    head,
    q_pos,
    kv_pos,
    cached=None,
):
    cached = {} if cached is None else cached
    values = [fx.Int32(batch), fx.Int32(head), q_pos, kv_pos]
    for index, instruction in enumerate(program):
        slot = index + 4
        if slot in cached:
            values.append(cached[slot])
            continue
        opcode = instruction[0]
        if opcode == "const_i32":
            values.append(fx.Int32(instruction[1]))
        elif opcode == "const_bool":
            constant = fx.Int32(1 if instruction[1] else 0)
            values.append(constant == fx.Int32(1))
        elif opcode == "load_i32":
            buffer = instruction[1]
            indices = instruction[2]
            offset = fx.Int32(0)
            for dimension in fx.range_constexpr(len(indices)):
                offset = offset + values[indices[dimension]] * fx.Int32(
                    strides[buffer][dimension]
                )
            values.append(load_i32(buffers[buffer], offset))
        else:
            lhs = values[instruction[1]]
            if opcode == "not":
                values.append(~lhs)
            else:
                rhs = values[instruction[2]]
                if opcode == "add":
                    values.append(lhs + rhs)
                elif opcode == "sub":
                    values.append(lhs - rhs)
                elif opcode == "mul":
                    values.append(lhs * rhs)
                elif opcode == "floordiv":
                    values.append(lhs // rhs)
                elif opcode == "remainder":
                    remainder = lhs % rhs
                    adjust = (remainder != 0) & ((remainder < 0) != (rhs < 0))
                    values.append(adjust.select(remainder + rhs, remainder))
                elif opcode == "ge":
                    values.append(lhs >= rhs)
                elif opcode == "gt":
                    values.append(lhs > rhs)
                elif opcode == "le":
                    values.append(lhs <= rhs)
                elif opcode == "lt":
                    values.append(lhs < rhs)
                elif opcode == "eq":
                    values.append(lhs == rhs)
                elif opcode == "ne":
                    values.append(lhs != rhs)
                elif opcode == "and":
                    values.append(lhs & rhs)
                elif opcode == "or":
                    values.append(lhs | rhs)
                else:
                    raise ValueError(f"unsupported mask bytecode op {opcode}")
    return values[output]


def _schedule_group(kind: str, count: int, group: int):
    fx.rocdl.sched_group_barrier(_SCHED_GROUP_MASKS[kind], count, group)


def schedule_fence():
    fx.rocdl.sched_barrier(0)


def fast_exp2(value):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, value.ir_value()))


def schedule_fwd_qk_pipeline(
    *, reduction_steps: int, vmem_count: int = 0, query_in_registers: bool = False
):
    """Interleave Q/K LDS reads and optional next-tile VMEM with QK MFMAs."""
    reads_per_step = 2 if query_in_registers else 3
    dsrd_preload = min(4 if query_in_registers else 6, reads_per_step * reduction_steps)
    fx.rocdl.sched_dsrd(dsrd_preload)
    scheduled_vmem = 0
    for step in fx.range_constexpr(reduction_steps):
        target_vmem = ((step + 1) * vmem_count) // reduction_steps
        if const_expr(target_vmem > scheduled_vmem):
            fx.rocdl.sched_vmem(target_vmem - scheduled_vmem)
            scheduled_vmem = target_vmem
        fx.rocdl.sched_mfma(2)
        if const_expr(step + 2 < reduction_steps):
            fx.rocdl.sched_dsrd(reads_per_step)
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
    tensors = (buffer0, buffer1, buffer2, buffer3)
    return [
        gview(tensors[buffer_index], None, sizes[buffer_index], 1)
        for buffer_index in fx.range_constexpr(count)
    ]


def make_mask_evaluator(program, output, strides, buffers, load_i32, batch, head):
    def evaluate(q_pos, kv_pos, cached=None):
        return evaluate_mask_program(
            program,
            output,
            strides,
            buffers,
            load_i32,
            batch,
            head,
            q_pos,
            kv_pos,
            cached,
        )

    return evaluate


def analyze_mask_access(
    mask_program,
    mask_output_slot,
    mask_buffer_shapes,
    mask_buffer_strides,
    *,
    paired,
    mask_load_width,
):
    vector_mask_loads = []
    for slot, instruction in enumerate(() if paired else mask_program, 4):
        if instruction[0] != "load_i32":
            continue
        buffer_index, indices = instruction[1:]
        if indices.count(3) != 1 or any(index not in (0, 1, 2, 3) for index in indices):
            continue
        dimension = indices.index(3)
        strides = mask_buffer_strides[buffer_index]
        if (
            strides[dimension] == 1
            and mask_buffer_shapes[buffer_index][dimension] % mask_load_width == 0
            and all(
                stride % mask_load_width == 0
                for element_index, stride in enumerate(strides)
                if element_index != dimension
            )
        ):
            vector_mask_loads.append((slot, buffer_index, indices))

    mask_interval_types = ["int"] * 4
    mask_key_dependent = [False, False, False, True]
    for instruction in mask_program:
        opcode = instruction[0]
        value_type = None
        if opcode in ("const_i32", "const_bool"):
            dependencies = ()
            value_type = "int" if opcode == "const_i32" else "bool"
        elif opcode == "load_i32":
            dependencies = instruction[2]
            if all(
                mask_interval_types[index] == "int" and not mask_key_dependent[index]
                for index in dependencies
            ):
                value_type = "int"
        else:
            dependencies = instruction[1:]
            if opcode in ("ge", "gt", "le", "lt", "eq", "ne"):
                if all(mask_interval_types[index] == "int" for index in dependencies):
                    value_type = "bool"
            elif opcode in ("and", "or", "not"):
                if all(mask_interval_types[index] == "bool" for index in dependencies):
                    value_type = "bool"
        mask_interval_types.append(value_type)
        mask_key_dependent.append(
            any(mask_key_dependent[index] for index in dependencies)
        )
    supports_mask_intervals = (
        bool(mask_program)
        and None not in mask_interval_types
        and mask_interval_types[mask_output_slot] == "bool"
    )
    flat_work_mask = is_sliding_window_mask_program(
        mask_program, mask_output_slot, mask_buffer_strides
    )
    return vector_mask_loads, supports_mask_intervals, flat_work_mask
