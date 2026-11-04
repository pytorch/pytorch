# mypy: allow-untyped-defs

import copy
import functools
import operator

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects.fly import PointerType
from flydsl.expr import const_expr
from flydsl.expr.typing import AddressSpace

from .flex_attn_bwd_utils import MAX_DIRECT_CAPTURES
from .flex_attn_target import WAVE_SIZE


# A sliding-window mask reaches the lowering in two shapes: and_masks() seeds a
# const_bool True and folds each term into it, while a plain (q >= kv) & (q - kv < W)
# lambda lowers without that prefix. Entries are (program, width_slot, output); the
# window width is a program constant, so that slot is wildcarded rather than compared.
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
        (
            ("ge", 2, 3),
            ("sub", 2, 3),
            ("const_i32", None),
            ("lt", 5, 6),
            ("and", 4, 7),
        ),
        2,
        8,
    ),
)
_SCHED_GROUP_MASKS = {
    "vmem_read": 0x020,
    "transcendental": 0x400,
}
ELEMENT_TYPES = {"bf16": fx.BFloat16, "fp16": fx.Float16}


def make_global_view(tensor, coord, shape, stride):
    view = fx.make_view(fx.get_iter(tensor), fx.make_layout(shape, stride))
    if coord is not None:
        # Slice the global pointer before creating the 32-bit buffer descriptor.
        # Callers widen batch/head coordinates to i64; local strides stay static.
        view = fx.slice(view, coord)
    return _make_buffer_view(view)


def _make_buffer_view(view):
    elements = fx.get_scalar(fx.cosize(view.layout))
    if not isinstance(elements, int):
        # A runtime extent is i32, whose bit count wraps from 256 MiB on.
        elements = fx.Int64(elements)
    # torch stores i1 (bool) elements one per byte.
    num_records_bytes = elements * max(view.element_type.width // 8, 1)
    if isinstance(num_records_bytes, int) and not 0 <= num_records_bytes <= 0xFFFFFFFF:
        raise ValueError("FlyDSL buffer view must fit in a 32-bit byte range")
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


def layout_extents(tensor):
    """The shape and stride leaves of ``tensor``: ints where its memref type is
    static, runtime values where the launch marked it layout-dynamic."""
    layout = fx.get_layout(tensor)
    return fx.get_leaves(fx.get_shape(layout)), fx.get_leaves(fx.get_stride(layout))


def optional_int(value):
    return None if value is None else int(value)


def runtime_extent(value, tensor, dim):
    """``value``, or ``tensor``'s runtime extent along ``dim`` if it is None."""
    if value is not None:
        return value
    return fx.Int32(layout_extents(tensor)[0][dim])


def runtime_strides(strides, tensor, sequence_length=None):
    """``strides`` with their runtime entries resolved: ``("seq", c)`` is ``c``
    times ``sequence_length`` and None is ``tensor``'s runtime stride."""
    if all(isinstance(stride, int) for stride in strides):
        return strides
    resolved = []
    for dim, stride in enumerate(strides):
        if stride is None:
            stride = layout_extents(tensor)[1][dim]
        elif not isinstance(stride, int):
            length = (
                sequence_length
                if isinstance(sequence_length, int)
                else fx.Int64(sequence_length)
            )
            stride = length * stride[1]
        resolved.append(stride)
    return tuple(resolved)


def runtime_cosize(tensor):
    """Elements spanned by ``tensor``'s runtime layout."""
    return fx.get_scalar(fx.cosize(fx.get_layout(tensor)))


def ceil_div(value, divisor):
    return (value + (divisor - 1)) // divisor


def tile_axes(head_slow):
    """The ``block_idx`` axes of a CTA's head and block; x dispatches fastest."""
    return ("y", "x") if head_slow else ("x", "y")


def tile_grid(heads, blocks, slices, head_slow):
    """The launch grid of a ``heads`` x ``blocks`` x ``slices`` tile space."""
    extents = dict(zip(tile_axes(head_slow), (heads, blocks)))
    return extents["x"], extents["y"], slices


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


def is_sliding_window_mask_program(
    mask_program,
    mask_program_output,
    mask_buffer_strides,
):
    """Recognize a causal mask bounded by a fixed lookback window.

    Such a mask gives every query block the same amount of work, so the query-block
    dispatch reversal that helps triangular masks only costs locality here.
    """
    return (
        sliding_window_mask_width(
            mask_program, mask_program_output, mask_buffer_strides
        )
        is not None
    )


def sliding_window_mask_width(mask_program, mask_program_output, mask_buffer_strides):
    """``W`` of a ``q >= kv and q - kv < W`` mask program, else None."""
    if mask_buffer_strides:
        return None
    program = tuple(mask_program)
    output = int(mask_program_output)
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
            return width[1]
    return None


def mask_instruction_operands(instruction):
    """Return the value ids a mask bytecode instruction reads."""
    op = instruction[0]
    if op in ("const_i32", "const_i64", "const_f32", "const_bool"):
        return ()
    if op in ("load_i32", "load"):
        return tuple(instruction[2])
    if op == "cast":
        return (instruction[1],)
    return tuple(instruction[1:])


def _as_numeric(value):
    # Kernels may pass signless positions, which shift and abs treat as unsigned.
    if isinstance(value, fx.Numeric):
        return value
    return fx.Numeric.from_ir_type(value.type)(value)


def _load_mask_element(view, index, dtype):
    iterator = fx.get_iter(view)
    if const_expr(dtype == "bool"):
        # torch.bool is one byte per element; an i1 view would load two bytes.
        iterator = fx.recast_iter(fx.Uint8, iterator)
    return mask_element_value(iterator[index], dtype)


def mask_element_value(element, dtype):
    """A loaded element of a ``load`` mask capture, as the program reads it."""
    if const_expr(dtype == "bool"):
        return fx.Int32(element) != fx.Int32(0)
    if const_expr(dtype == "uint8"):
        return fx.Int32(element) & fx.Int32(0xFF)
    if const_expr(dtype in ("int8", "int16")):
        return fx.Int32(element)
    if const_expr(dtype == "int64"):
        return fx.Int64(element)
    return fx.Float32(element)


def _cast_mask_value(value, kind):
    if const_expr(kind == "bool"):
        if const_expr(isinstance(value, fx.Boolean)):
            return value
        return value != type(value)(0)
    if const_expr(kind == "i32"):
        return fx.Int32(value)
    if const_expr(kind == "i64"):
        return fx.Int64(value)
    value = fx.Float32(value)
    if const_expr(kind == "f32"):
        return value
    return fx.Float32(ELEMENT_TYPES[kind](value))


def _minimum(lhs, rhs):
    if const_expr(isinstance(lhs, fx.Float32)):
        return lhs.minimumf(rhs)
    return (lhs < rhs).select(lhs, rhs)


def _maximum(lhs, rhs):
    if const_expr(isinstance(lhs, fx.Float32)):
        return lhs.maximumf(rhs)
    return (lhs > rhs).select(lhs, rhs)


_NUMERIC_BINARY_OPS = {
    "xor": operator.xor,
    "shl": operator.lshift,
    "shr": operator.rshift,
    "div": operator.truediv,
    "fmod": operator.mod,
    "min": _minimum,
    "max": _maximum,
}


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
    cached_values=None,
    independent_of=None,
):
    """The program's output; with ``independent_of`` 2 (``q_pos``) or 3
    (``kv_pos``), instead the value of every non-constant instruction that
    does not read that position, by slot."""
    if not mask_program:
        # The no-op mask_mod keeps every element.
        return {} if independent_of is not None else fx.Int32(1) == fx.Int32(1)
    cached_values = {} if cached_values is None else cached_values
    values = [fx.Int32(batch), fx.Int32(head), q_pos, kv_pos]
    dependent = [slot == independent_of for slot in range(4)]
    for instruction_index, instruction in enumerate(mask_program):
        slot = instruction_index + 4
        dependent.append(
            any(dependent[i] for i in mask_instruction_operands(instruction))
        )
        if slot in cached_values:
            values.append(cached_values[slot])
            continue
        if dependent[slot]:
            values.append(None)
            continue
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
        elif op == "const_i64":
            values.append(fx.Int64(instruction[1]))
        elif op == "const_f32":
            values.append(fx.Float32(instruction[1]))
        elif op == "load":
            buffer_index, index_ids, dtype = instruction[1:]
            offset = fx.Int32(0)
            for dimension in fx.range_constexpr(len(index_ids)):
                offset = offset + values[index_ids[dimension]] * fx.Int32(
                    mask_buffer_strides[buffer_index][dimension]
                )
            values.append(_load_mask_element(mask_buffers[buffer_index], offset, dtype))
        elif op == "cast":
            values.append(
                _cast_mask_value(_as_numeric(values[instruction[1]]), instruction[2])
            )
        elif op == "where":
            condition, true_value, false_value = (
                _as_numeric(values[index]) for index in instruction[1:]
            )
            values.append(condition.select(true_value, false_value))
        else:
            lhs = values[instruction[1]]
            if op == "not":
                values.append(~lhs)
            elif op == "neg":
                values.append(-_as_numeric(lhs))
            elif op == "abs":
                values.append(abs(_as_numeric(lhs)))
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
                elif op in _NUMERIC_BINARY_OPS:
                    values.append(
                        _NUMERIC_BINARY_OPS[op](_as_numeric(lhs), _as_numeric(rhs))
                    )
                else:
                    raise ValueError(f"unsupported mask bytecode op {op}")
    if independent_of is not None:
        return {
            slot: values[slot]
            for slot, instruction in enumerate(mask_program, 4)
            if values[slot] is not None and not instruction[0].startswith("const_")
        }
    return values[mask_program_output]


def _schedule_group(kind: str, count: int, group: int):
    fx.rocdl.sched_group_barrier(_SCHED_GROUP_MASKS[kind], count, group)


def schedule_fence():
    fx.rocdl.sched_barrier(0)


def fast_exp2(value):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, value.ir_value()))


def tied_nop(value, count, *, has_side_effects=True):
    """Pass ``value`` through ``s_nop count`` so every consumer follows the nop.

    gfx950 needs 1 wait state after ``v_exp_f32`` and 2 after a packed f32
    convert read by an MFMA; LLVM inserts neither and counts scalar
    instructions as wait states. Without side effects the nop is ordered by
    its operand alone and does not split the scheduling region.
    """
    raw = value.ir_value()
    return type(value)(
        llvm.inline_asm(
            raw.type, [raw], f"s_nop {count}", "=v,0", has_side_effects=has_side_effects
        )
    )


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


CAPTURE_TABLE_SLOTS = 8
_CAPTURE_ELEMENT_TYPES = {
    # torch.bool is one byte per element.
    "bool": fx.Uint8,
    "uint8": fx.Uint8,
    "int8": fx.Int8,
    "int16": fx.Int16,
    "int32": fx.Int32,
    "int64": fx.Int64,
    "float16": fx.Float16,
    "bfloat16": fx.BFloat16,
    "float32": fx.Float32,
}


def capture_table_layout(
    mask_buffer_sizes,
    mask_buffer_dtypes,
    score_mod_aux,
    score_mod_buffer_count,
    score_mod_buffer_dtypes,
):
    """``(sizes, dtypes)`` of every capture (mask first) if they need a table,
    else None."""
    if len(mask_buffer_sizes) + score_mod_buffer_count <= MAX_DIRECT_CAPTURES:
        return None
    dtypes = (*mask_buffer_dtypes, *score_mod_buffer_dtypes)
    if len(dtypes) != len(mask_buffer_sizes) + score_mod_buffer_count:
        raise ValueError("a capture table needs the dtype of every capture")
    if any(dtype not in _CAPTURE_ELEMENT_TYPES for dtype in dtypes):
        raise ValueError(f"unsupported capture dtype in {dtypes}")
    score_mod_sizes = [1] * score_mod_buffer_count
    for buffer_index, _, numel in score_mod_aux:
        score_mod_sizes[buffer_index] = max(score_mod_sizes[buffer_index], numel)
    return (*mask_buffer_sizes, *score_mod_sizes), dtypes


class CaptureTable:
    """Views of captures whose base addresses are read from an int64 table.

    Lane ``i`` of one VGPR pair per wave of captures holds address ``i``; every read
    rebuilds the descriptor from two readlanes, so it holds SGPRs only while a
    load uses it. An opaque lane move keeps the readlanes from being merged or
    hoisted out of loops.
    """

    def __init__(self, table, sizes, dtypes):
        self.sizes = sizes
        self.dtypes = dtypes
        self.first = 0
        lane = fx.Int32(fx.thread_idx.x) % fx.Int32(WAVE_SIZE)
        view = make_global_view(table, None, len(sizes), 1)
        self.entries = [
            fx.Int64(view[lane + fx.Int32(WAVE_SIZE * chunk)])
            for chunk in range(-(-len(sizes) // WAVE_SIZE))
        ]

    def skip(self, count):
        """The same table indexed from capture ``count``."""
        tail = copy.copy(self)
        tail.first = self.first + count
        return tail

    def __getitem__(self, index):
        index += self.first
        lane = llvm.inline_asm(
            fx.Int32.ir_type,
            [fx.Int32(index % WAVE_SIZE).ir_value()],
            "",
            "=s,0",
            has_side_effects=True,
        )
        address = fx.Int64(
            fx.rocdl.readlane(fx.Int64.ir_type, self.entries[index // WAVE_SIZE], lane)
        )
        element = _CAPTURE_ELEMENT_TYPES[self.dtypes[index]]
        pointer = fx.inttoptr(
            PointerType.get(element.ir_type, AddressSpace.Global, element.width // 8),
            address,
        )
        return _make_buffer_view(
            fx.make_view(pointer, fx.make_layout(self.sizes[index], 1))
        )


def make_capture_buffers(gview, mask_buffer_count, mask_buffer_sizes, layout, slots):
    """Split a kernel's four capture slots into mask and score_mod buffers;
    with a ``capture_table_layout`` slot 0 is the table and both are
    ``CaptureTable`` views."""
    if layout is None:
        masks = make_mask_buffers(gview, mask_buffer_count, mask_buffer_sizes, *slots)
        return masks, list(slots)[mask_buffer_count:]
    sizes, dtypes = layout
    table = CaptureTable(
        slots[0], (*mask_buffer_sizes, *sizes[len(mask_buffer_sizes) :]), dtypes
    )
    return table, table.skip(mask_buffer_count)


def capture_extent_count(mask_buffer_sizes, layout):
    """Capture slots a launch passes: the direct captures, or the table and
    then each table capture whose extent is read at runtime."""
    if layout is None:
        return len(mask_buffer_sizes)
    count = 1 + sum(size is None for size in mask_buffer_sizes)
    if count > MAX_DIRECT_CAPTURES:
        raise ValueError(
            f"a capture table takes at most {MAX_DIRECT_CAPTURES - 1} captures "
            "with runtime extents"
        )
    return count


def resolve_capture_extents(strides, sizes, slots, layout):
    """``strides`` and element extents of the mask captures with their runtime
    entries read from the capture slots; a table's runtime-extent captures
    follow it in slot order."""
    if layout is None:
        tensors = slots
    else:
        free_slots = iter(slots[1:])
        tensors = [None if size is not None else next(free_slots) for size in sizes]
    return (
        tuple(
            runtime_strides(stride, tensor) for stride, tensor in zip(strides, tensors)
        ),
        tuple(
            runtime_cosize(tensor) if size is None else size
            for size, tensor in zip(sizes, tensors)
        ),
    )


def build_capture_table_launch(first, count):
    """A launch ``(table, *tensors, stream)`` that writes the addresses of the
    first ``min(CAPTURE_TABLE_SLOTS, count - first)`` tensors to
    ``table[first:]``; the other slots alias ``table`` and are not read."""
    slots = min(CAPTURE_TABLE_SLOTS, count - first)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kernel(
        table: fx.Tensor,
        tensor_0: fx.Tensor,
        tensor_1: fx.Tensor,
        tensor_2: fx.Tensor,
        tensor_3: fx.Tensor,
        tensor_4: fx.Tensor,
        tensor_5: fx.Tensor,
        tensor_6: fx.Tensor,
        tensor_7: fx.Tensor,
    ):
        tensors = (
            tensor_0,
            tensor_1,
            tensor_2,
            tensor_3,
            tensor_4,
            tensor_5,
            tensor_6,
            tensor_7,
        )
        base = fx.get_iter(table)
        for slot in fx.range_constexpr(slots):
            address = fx.Int64(fx.ptrtoint(fx.get_iter(tensors[slot])))
            fx.ptr_store(address, fx.add_offset(base, first + slot))

    @flyc.jit
    def launch(
        table: fx.Tensor,
        tensor_0: fx.Tensor,
        tensor_1: fx.Tensor,
        tensor_2: fx.Tensor,
        tensor_3: fx.Tensor,
        tensor_4: fx.Tensor,
        tensor_5: fx.Tensor,
        tensor_6: fx.Tensor,
        tensor_7: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(
            table,
            tensor_0,
            tensor_1,
            tensor_2,
            tensor_3,
            tensor_4,
            tensor_5,
            tensor_6,
            tensor_7,
        ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def mask_capture_sizes(shapes, strides):
    """Element extents of the mask captures; None where one is read at runtime."""
    return tuple(
        None
        if None in shape or None in stride
        else 1 + sum((size - 1) * step for size, step in zip(shape, stride))
        for shape, stride in zip(shapes, strides)
    )


def validate_score_mod_args(score_mod, score_mod_aux, score_mod_buffer_count, mod_key):
    """Return ``(score_mod_aux, score_mod_buffer_count)`` as ints after checking them."""
    score_mod_aux = tuple(
        (int(index), tuple(int(s) for s in strides), int(numel))
        for index, strides, numel in score_mod_aux
    )
    score_mod_buffer_count = int(score_mod_buffer_count)
    if score_mod is None:
        if score_mod_aux or score_mod_buffer_count:
            raise ValueError("score_mod captures require a score_mod")
        return score_mod_aux, score_mod_buffer_count
    if mod_key is None:
        # FlyDSL's JIT cache does not see constants a mod reads from module scope.
        raise ValueError("score_mod requires an explicit mod_key")
    for buffer_index, strides, numel in score_mod_aux:
        if not 0 <= buffer_index < score_mod_buffer_count:
            raise ValueError("score_mod aux slot names a missing capture buffer")
        if len(strides) != 4 or any(stride < 0 for stride in strides) or numel <= 0:
            raise ValueError("score_mod aux slots need four non-negative strides")
    return score_mod_aux, score_mod_buffer_count


def make_score_mod_readers(gview, score_mod_aux, buffers):
    """Build ``aux[i](b, h, q_idx, kv_idx) -> f32`` readers for a score_mod.

    Slot ``i`` is ``(buffer_index, (stride_b, stride_h, stride_q, stride_kv), numel)``;
    out-of-range reads return zero through the buffer descriptor bound.
    """
    readers = []
    for buffer_index, strides, numel in score_mod_aux:
        if isinstance(buffers, CaptureTable):
            view = functools.partial(buffers.__getitem__, buffer_index)
        else:
            view = _constant(gview(buffers[buffer_index], None, numel, 1))
        readers.append(_make_score_mod_reader(view, strides))
    return readers


def make_tile_score_mod_readers(score_mod_aux, table, keys=()):
    """``make_score_mod_readers`` over a ``CaptureTable`` for one tile, loading
    once per tile per coordinate objects. ``keys`` are the tile's key positions
    in 4-aligned runs of four; a unit-stride 32-bit read at one loads the run.
    """
    width = 4
    runs = {id(key): divmod(index, width) for index, key in enumerate(keys)}
    loads = {}

    def make_reader(buffer_index, strides, numel):
        element = _CAPTURE_ELEMENT_TYPES[table.dtypes[table.first + buffer_index]]
        vector = (
            strides[3] == 1
            and element.width == 32
            and numel % width == 0
            and all(stride % width == 0 for stride in strides[:3])
        )
        scalar = _make_score_mod_reader(
            functools.partial(table.__getitem__, buffer_index), strides
        )

        def read(b, h, q_idx, kv_idx):
            coordinates = (b, h, q_idx, kv_idx)
            run, position = None, 0
            if vector and id(kv_idx) in runs:
                run, position = runs[id(kv_idx)]
            used = tuple(
                coordinate
                for dim, (stride, coordinate) in enumerate(zip(strides, coordinates))
                if stride != 0 and (dim < 3 or run is None)
            )
            # The coordinates stay referenced, so their ids are not reused.
            key = (buffer_index, run, *map(id, used))
            if key not in loads:
                if run is None:
                    value = scalar(*coordinates)
                else:
                    offset = keys[run * width]
                    for stride, coordinate in zip(strides[:3], coordinates):
                        if const_expr(stride != 0):
                            offset = offset + fx.Int32(coordinate) * fx.Int32(stride)
                    fragment = fx.make_rmem_tensor(width, element)
                    fx.copy(
                        fx.make_copy_atom(fx.rocdl.BufferCopy128b(), element),
                        fx.slice(
                            fx.logical_divide(
                                table[buffer_index], fx.make_layout(width, 1)
                            ),
                            (None, offset // fx.Int32(width)),
                        ),
                        fragment,
                    )
                    value = fragment.load()
                loads[key] = (value, used)
            value = loads[key][0]
            if run is None:
                return value
            return fx.Float32(element(value[position]))

        return read

    return [make_reader(*slot) for slot in score_mod_aux]


def _constant(value):
    return lambda: value


def _make_score_mod_reader(view, strides):
    def read(b, h, q_idx, kv_idx):
        offset = None
        for stride, coordinate in zip(strides, (b, h, q_idx, kv_idx)):
            if const_expr(stride == 0):
                continue
            term = fx.Int32(coordinate)
            if const_expr(stride != 1):
                term = term * fx.Int32(stride)
            offset = term if offset is None else offset + term
        # bf16/fp16 captures widen to f32 here.
        return fx.Float32(view()[fx.Int32(0) if offset is None else offset])

    return read


def make_mask_evaluator(program, output, strides, buffers, load_i32, batch, head):
    def evaluate(q_pos, kv_pos, cached_values=None, independent_of=None):
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
            cached_values=cached_values,
            independent_of=independent_of,
        )

    return evaluate
