# mypy: allow-untyped-defs

import math

import flydsl.expr as fx
from flydsl.expr import const_expr


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
    if mask_buffer_strides:
        return False
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
            return True
    return False


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
    only_slots=None,
):
    cached_values = {} if cached_values is None else cached_values
    values = [fx.Int32(batch), fx.Int32(head), q_pos, kv_pos]
    for instruction_index, instruction in enumerate(mask_program):
        slot = instruction_index + 4
        if slot in cached_values or (only_slots is not None and slot not in only_slots):
            values.append(cached_values.get(slot))
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
    if only_slots is not None:
        return {slot: values[slot] for slot in only_slots}
    return values[mask_program_output]


def _mask_position_dependencies(mask_program):
    dependencies = [set(), set(), {"q"}, {"kv"}]
    for instruction in mask_program:
        op = instruction[0]
        if op in ("const_i32", "const_bool"):
            operands = ()
        elif op == "load_i32":
            operands = instruction[2]
        else:
            operands = instruction[1:]
        merged = set()
        for operand in operands:
            merged |= dependencies[operand]
        dependencies.append(merged)
    return dependencies


_SCORE_LOG2E = 1.4426950408889634
_SCORE_LN2 = 0.6931471805599453
# Below this magnitude tanh's cubic Taylor term is exact in FP32 and avoids the
# cancellation in (1 - exp(-2|x|)).
_TANH_SERIES_LIMIT = 0.004
# A tangent equal to the score's own seed; it avoids multiplying by one.
_UNIT_TANGENT = "unit"
_SCORE_FLOAT_UNARY_OPS = frozenset(
    (
        "fneg",
        "fabs",
        "exp",
        "exp2",
        "log",
        "log2",
        "tanh",
        "sigmoid",
        "sqrt",
        "rsqrt",
        "recip",
        "sin",
        "cos",
    )
)


def _reciprocal(value):
    # v_rcp_f32 (1 ulp) instead of IEEE division's scale/fixup sequence; score
    # arithmetic is FP32 and feeds an exp2 anyway.
    return fx.Float32(fx.rocdl.rcp(fx.Float32.ir_type, value.ir_value()))


def _score_tanh(value):
    magnitude = fx.math.absf(value)
    decay = fast_exp2(magnitude * fx.Float32(-2.0 * _SCORE_LOG2E))
    large = (fx.Float32(1.0) - decay) * _reciprocal(fx.Float32(1.0) + decay)
    large = (value < fx.Float32(0.0)).select(-large, large)
    small = value * (fx.Float32(1.0) - value * value * fx.Float32(1.0 / 3.0))
    return (magnitude < fx.Float32(_TANH_SERIES_LIMIT)).select(small, large)


def _score_float_unary(op, value):
    if op == "fneg":
        return -value
    if op == "fabs":
        return fx.Float32(fx.math.absf(value))
    if op == "exp":
        return fast_exp2(value * fx.Float32(_SCORE_LOG2E))
    if op == "exp2":
        return fast_exp2(value)
    if op == "log":
        return fx.Float32(fx.math.log2(value)) * fx.Float32(_SCORE_LN2)
    if op == "log2":
        return fx.Float32(fx.math.log2(value))
    if op == "tanh":
        return _score_tanh(value)
    if op == "sigmoid":
        return _reciprocal(
            fx.Float32(1.0) + fast_exp2(value * fx.Float32(-_SCORE_LOG2E))
        )
    if op == "sqrt":
        return fx.Float32(fx.math.sqrt(value))
    if op == "rsqrt":
        return fx.Float32(fx.math.rsqrt(value))
    if op == "recip":
        return _reciprocal(value)
    if op == "sin":
        return fx.Float32(fx.math.sin(value))
    if op == "cos":
        return fx.Float32(fx.math.cos(value))
    raise ValueError(f"unsupported score bytecode op {op}")


# A tangent is None (zero) or a tuple with one component per differentiated
# input (the score, then each seeded load). Components are None, the unit seed,
# or an FP32 value.
def _scale_component(component, factor):
    if component is None:
        return None
    if component is _UNIT_TANGENT:
        return factor
    return component * factor


def _materialize_component(component):
    if component is _UNIT_TANGENT:
        return fx.Float32(1.0)
    return component


def _add_components(lhs, rhs, subtract):
    if rhs is None:
        return lhs
    if lhs is None:
        return -_materialize_component(rhs) if subtract else rhs
    lhs, rhs = _materialize_component(lhs), _materialize_component(rhs)
    return lhs - rhs if subtract else lhs + rhs


def _select_components(condition, lhs, rhs):
    if lhs is None and rhs is None:
        return None
    zero = fx.Float32(0.0)
    lhs = zero if lhs is None else _materialize_component(lhs)
    rhs = zero if rhs is None else _materialize_component(rhs)
    return condition.select(lhs, rhs)


def _scale_tangent(tangent, factor):
    if tangent is None:
        return None
    return tuple(_scale_component(component, factor) for component in tangent)


def _negate_tangent(tangent):
    if tangent is None:
        return None
    return tuple(
        None if component is None else -_materialize_component(component)
        for component in tangent
    )


def _add_tangents(lhs, rhs, subtract=False):
    if rhs is None:
        return lhs
    if lhs is None:
        return _negate_tangent(rhs) if subtract else rhs
    return tuple(_add_components(a, b, subtract) for a, b in zip(lhs, rhs))


def _select_tangent(condition, lhs, rhs):
    if lhs is None and rhs is None:
        return None
    width = len(lhs if lhs is not None else rhs)
    lhs = (None,) * width if lhs is None else lhs
    rhs = (None,) * width if rhs is None else rhs
    return tuple(_select_components(condition, a, b) for a, b in zip(lhs, rhs))


def _float_unary_tangent(op, operand, result, tangent):
    if tangent is None:
        return None
    if op == "fneg":
        return _negate_tangent(tangent)
    if op == "fabs":
        return _select_tangent(
            operand < fx.Float32(0.0), _negate_tangent(tangent), tangent
        )
    if op == "exp":
        factor = result
    elif op == "exp2":
        factor = result * fx.Float32(_SCORE_LN2)
    elif op == "log":
        factor = _reciprocal(operand)
    elif op == "log2":
        factor = fx.Float32(1.0 / _SCORE_LN2) * _reciprocal(operand)
    elif op == "tanh":
        factor = fx.Float32(1.0) - result * result
    elif op == "sigmoid":
        factor = result * (fx.Float32(1.0) - result)
    elif op == "sqrt":
        factor = fx.Float32(0.5) * _reciprocal(result)
    elif op == "rsqrt":
        factor = fx.Float32(-0.5) * result * result * result
    elif op == "recip":
        factor = -(result * result)
    elif op == "sin":
        factor = fx.Float32(fx.math.cos(operand))
    elif op == "cos":
        factor = -fx.Float32(fx.math.sin(operand))
    else:
        raise ValueError(f"unsupported score bytecode op {op}")
    return _scale_tangent(tangent, factor)


def evaluate_score_program(
    *,
    score_program,
    score_program_output,
    score_buffer_strides,
    score_buffers,
    load_element,
    score,
    batch,
    head,
    q_pos,
    kv_pos,
    with_gradient=False,
    seeded_loads=(),
    cached_values=None,
    only_slots=None,
):
    """Evaluate score_mod bytecode on one FP32 score.

    With ``with_gradient`` the result is ``(value, d value / d score)``,
    computed by forward-mode differentiation of the same bytecode. Loads whose
    slots are in ``seeded_loads`` are differentiated too; the result is then
    ``(value, d value / d score, d value / d loads, load offsets)``. Slots in
    ``cached_values`` are reused as score-independent values. With
    ``only_slots``, only those slots are computed and returned as a dict;
    their operands must be among them or be non-None inputs.
    """
    # Loop-carried positions may be untyped IR values; float conversion needs
    # the signed Int32 wrapper.
    values = [
        score,
        *(None if value is None else fx.Int32(value) for value in (batch, head)),
        *(None if value is None else fx.Int32(value) for value in (q_pos, kv_pos)),
    ]
    width = 1 + len(seeded_loads)
    score_tangent = (_UNIT_TANGENT,) + (None,) * (width - 1)
    tangents = [score_tangent if with_gradient else None, None, None, None, None]
    load_offsets = {}
    cached_values = {} if cached_values is None else cached_values
    for slot, instruction in enumerate(score_program, 5):
        op = instruction[0]
        tangent = None
        if slot in cached_values or (only_slots is not None and slot not in only_slots):
            values.append(cached_values.get(slot))
            tangents.append(None)
            continue
        if op == "const_f32":
            # Lowering stores float.hex() so inf survives template rendering.
            value = fx.Float32(float.fromhex(instruction[1]))
        elif op == "const_i32":
            value = fx.Int32(instruction[1])
        elif op == "const_bool":
            value = fx.Int32(1 if instruction[1] else 0) == fx.Int32(1)
        elif op in ("load_f32", "load_i32"):
            buffer_index = instruction[1]
            offset = fx.Int32(0)
            for dimension, index in enumerate(instruction[2]):
                offset = offset + values[index] * fx.Int32(
                    score_buffer_strides[buffer_index][dimension]
                )
            loaded = load_element(score_buffers[buffer_index], offset)
            value = fx.Float32(loaded) if op == "load_f32" else fx.Int32(loaded)
            if with_gradient and slot in seeded_loads:
                seed = 1 + seeded_loads.index(slot)
                tangent = tuple(
                    _UNIT_TANGENT if component == seed else None
                    for component in range(width)
                )
                load_offsets[slot] = offset
        elif op == "select":
            condition = values[instruction[1]]
            value = condition.select(values[instruction[2]], values[instruction[3]])
            tangent = _select_tangent(
                condition, tangents[instruction[2]], tangents[instruction[3]]
            )
        else:
            lhs = values[instruction[1]]
            lhs_tangent = tangents[instruction[1]]
            if op in ("i2f", "f2i"):
                value = fx.Float32(lhs) if op == "i2f" else fx.Int32(lhs)
            elif op in ("b2f", "b2i"):
                one = fx.Float32(1.0) if op == "b2f" else fx.Int32(1)
                zero = fx.Float32(0.0) if op == "b2f" else fx.Int32(0)
                value = lhs.select(one, zero)
            elif op == "f2b":
                value = lhs != fx.Float32(0.0)
            elif op == "i2b":
                value = lhs != fx.Int32(0)
            elif op == "not":
                value = ~lhs
            elif op in _SCORE_FLOAT_UNARY_OPS:
                value = _score_float_unary(op, lhs)
                tangent = _float_unary_tangent(op, lhs, value, lhs_tangent)
            else:
                rhs = values[instruction[2]]
                rhs_tangent = tangents[instruction[2]]
                if op == "fadd":
                    value = lhs + rhs
                    tangent = _add_tangents(lhs_tangent, rhs_tangent)
                elif op == "fsub":
                    value = lhs - rhs
                    tangent = _add_tangents(lhs_tangent, rhs_tangent, subtract=True)
                elif op == "fmul":
                    value = lhs * rhs
                    tangent = _add_tangents(
                        _scale_tangent(lhs_tangent, rhs),
                        _scale_tangent(rhs_tangent, lhs),
                    )
                elif op == "fdiv":
                    inverse = _reciprocal(rhs)
                    value = lhs * inverse
                    if lhs_tangent is not None or rhs_tangent is not None:
                        tangent = _add_tangents(
                            _scale_tangent(lhs_tangent, inverse),
                            _scale_tangent(rhs_tangent, value * inverse),
                            subtract=True,
                        )
                elif op in ("fmax", "fmin"):
                    value = lhs.maximumf(rhs) if op == "fmax" else lhs.minimumf(rhs)
                    if lhs_tangent is not None or rhs_tangent is not None:
                        # Match aten.maximum/minimum: ties split the gradient.
                        lhs_wins = lhs > rhs if op == "fmax" else lhs < rhs
                        tie = _add_tangents(lhs_tangent, rhs_tangent)
                        tie = _scale_tangent(tie, fx.Float32(0.5))
                        tangent = _select_tangent(
                            lhs_wins,
                            lhs_tangent,
                            _select_tangent(lhs == rhs, tie, rhs_tangent),
                        )
                elif op in ("fge", "ge"):
                    value = lhs >= rhs
                elif op in ("fgt", "gt"):
                    value = lhs > rhs
                elif op in ("fle", "le"):
                    value = lhs <= rhs
                elif op in ("flt", "lt"):
                    value = lhs < rhs
                elif op in ("feq", "eq"):
                    value = lhs == rhs
                elif op in ("fne", "ne"):
                    value = lhs != rhs
                elif op == "add":
                    value = lhs + rhs
                elif op == "sub":
                    value = lhs - rhs
                elif op == "mul":
                    value = lhs * rhs
                elif op == "max":
                    value = (lhs >= rhs).select(lhs, rhs)
                elif op == "min":
                    value = (lhs <= rhs).select(lhs, rhs)
                elif op == "floordiv":
                    value = lhs // rhs
                elif op == "remainder":
                    remainder = lhs % rhs
                    needs_adjustment = (remainder != 0) & ((remainder < 0) != (rhs < 0))
                    value = needs_adjustment.select(remainder + rhs, remainder)
                elif op == "and":
                    value = lhs & rhs
                elif op == "or":
                    value = lhs | rhs
                else:
                    raise ValueError(f"unsupported score bytecode op {op}")
        values.append(value)
        tangents.append(tangent)

    if only_slots is not None:
        return {slot: values[slot] for slot in only_slots}
    result = values[score_program_output]
    if not with_gradient:
        return result
    tangent = tangents[score_program_output] or (None,) * width
    gradients = [
        fx.Float32(0.0) if component is None else _materialize_component(component)
        for component in tangent
    ]
    if not seeded_loads:
        return result, gradients[0]
    return (
        result,
        gradients[0],
        gradients[1:],
        [load_offsets[slot] for slot in seeded_loads],
    )


def _score_operands(instruction):
    op = instruction[0]
    if op in ("const_f32", "const_i32", "const_bool"):
        return ()
    if op in ("load_f32", "load_i32"):
        return tuple(instruction[2])
    return tuple(instruction[1:])


def score_invariant_slots(score_program, varying_slots):
    """Instruction slots that depend on none of ``varying_slots``."""
    varying = [slot in varying_slots for slot in range(5)]
    for slot, instruction in enumerate(score_program, 5):
        varying.append(
            slot in varying_slots
            or any(varying[index] for index in _score_operands(instruction))
        )
    return frozenset(slot for slot in range(5, len(varying)) if not varying[slot])


def score_gradient_is_unit(score_program, score_program_output):
    """Whether d(score_mod)/d(score) is exactly 1, e.g. additive biases."""
    unit = {0}
    for slot, instruction in enumerate(score_program, 5):
        op = instruction[0]
        operands = _score_operands(instruction)
        if op in ("fadd", "fsub"):
            lhs, rhs = operands
            if lhs in unit and not _depends_on_score(score_program, rhs):
                unit.add(slot)
            elif (
                op == "fadd"
                and rhs in unit
                and not _depends_on_score(score_program, lhs)
            ):
                unit.add(slot)
    return score_program_output in unit


def _depends_on_score(score_program, value):
    if value == 0:
        return True
    if value < 5:
        return False
    return any(
        _depends_on_score(score_program, operand)
        for operand in _score_operands(score_program[value - 5])
    )


def score_program_may_be_negative_infinite(score_program):
    """Conservatively, whether score_mod can return -inf for a finite score."""

    def nonzero_constant(value):
        if value < 5:
            return False
        instruction = score_program[value - 5]
        return (
            instruction[0] == "const_f32"
            and float.fromhex(instruction[1]) != 0.0
            and math.isfinite(float.fromhex(instruction[1]))
        )

    for instruction in score_program:
        op = instruction[0]
        if op == "const_f32" and not math.isfinite(float.fromhex(instruction[1])):
            return True
        if op in ("log", "log2"):
            return True
        # Division by a nonzero constant stays finite.
        if op == "fdiv" and not nonzero_constant(instruction[2]):
            return True
        if op in ("recip", "rsqrt") and not nonzero_constant(instruction[1]):
            return True
    return False


def score_additive_bias_slot(score_program, score_program_output):
    """Slot of ``bias`` when score_mod is ``score + bias(b, h, q, kv)``, else None.

    The bias then folds into the exponent like LSE, and can be evaluated
    before the score is known.
    """
    if score_program_output < 5:
        return None
    op, *operands = score_program[score_program_output - 5]
    if op != "fadd" or 0 not in operands:
        return None
    bias = operands[1] if operands[0] == 0 else operands[0]
    if _depends_on_score(score_program, bias):
        return None
    return bias


def _score_position_dependencies(score_program):
    """Per slot, which of ``"q"``/``"kv"`` (and the score) it depends on."""
    dependencies = [{"score"}, set(), set(), {"q"}, {"kv"}]
    for instruction in score_program:
        merged = set()
        for operand in _score_operands(instruction):
            merged |= dependencies[operand]
        dependencies.append(merged)
    return dependencies


def score_bias_terms(score_program, bias_slot):
    """Split an additive bias into ``(kind, leaf, factors, sign)`` terms.

    ``kind`` is ``"row"`` (q only), ``"col"`` (kv only), ``"inv"`` (neither)
    or ``"joint"``; the bias is ``sum(sign * leaf * prod(factors))``, with
    factors depending on batch and head only. Linear position arithmetic like
    ALiBi's ``(kv - q) * slope[h]`` splits into per-row and per-column terms,
    so a tile evaluates it once per row and column instead of per element.
    """
    dependencies = _score_position_dependencies(score_program)

    def kind(slot):
        positions = dependencies[slot] & {"q", "kv"}
        if not positions:
            return "inv"
        if positions == {"q", "kv"}:
            return "joint"
        return "row" if positions == {"q"} else "col"

    def split(slot, factors, sign):
        slot_kind = kind(slot)
        if slot_kind != "joint" or slot < 5:
            return [(slot_kind, slot, factors, sign)]
        op, *operands = score_program[slot - 5]
        if op in ("fadd", "fsub"):
            lhs, rhs = operands
            return split(lhs, factors, sign) + split(
                rhs, factors, -sign if op == "fsub" else sign
            )
        if op == "fneg":
            return split(operands[0], factors, -sign)
        if op == "fmul":
            lhs, rhs = operands
            if kind(rhs) == "inv":
                return split(lhs, factors + ((rhs, False),), sign)
            if kind(lhs) == "inv":
                return split(rhs, factors + ((lhs, False),), sign)
        if op == "fdiv" and kind(operands[1]) == "inv":
            return split(operands[0], factors + ((operands[1], True),), sign)
        return [("joint", slot, factors, sign)]

    return split(bias_slot, (), 1)


def _score_slot_closure(score_program, slots):
    """Instruction slots needed to evaluate ``slots``."""
    needed = set()
    pending = [slot for slot in slots if slot >= 5]
    while pending:
        slot = pending.pop()
        if slot in needed:
            continue
        needed.add(slot)
        pending.extend(
            operand
            for operand in _score_operands(score_program[slot - 5])
            if operand >= 5
        )
    return needed


def make_score_evaluator(
    program,
    output,
    strides,
    buffers,
    load_element,
    batch,
    head,
    q_pos=None,
    seeded_loads=(),
):
    """Build a per-element evaluator, hoisting work that is invariant per call site.

    Subexpressions of batch and head, and of ``q_pos`` when it is fixed for the
    caller, are computed here once instead of for every score element. Loads in
    ``seeded_loads`` stay per element because their derivatives are needed.
    """
    common = {
        "score_program": program,
        "score_program_output": output,
        "score_buffer_strides": strides,
        "score_buffers": buffers,
        "load_element": load_element,
        "batch": batch,
        "head": head,
    }
    varying_slots = ((0, 4) if q_pos is not None else (0, 3, 4)) + tuple(seeded_loads)
    hoisted = evaluate_score_program(
        **common,
        score=None,
        q_pos=q_pos,
        kv_pos=None,
        only_slots=score_invariant_slots(program, varying_slots),
    )
    # Per-position work that does not need the score (e.g. bias-table loads).
    positional_slots = score_invariant_slots(program, (0, *seeded_loads)) - set(hoisted)

    def prepare(q_pos, kv_pos):
        """Evaluate the score-independent part early, e.g. before QK MFMAs."""
        if not positional_slots:
            return {}
        return evaluate_score_program(
            **common,
            score=None,
            q_pos=q_pos,
            kv_pos=kv_pos,
            cached_values=hoisted,
            only_slots=positional_slots,
        )

    def evaluate_slot(slot, prepared):
        """Value of a score-independent ``slot`` from ``prepare`` or hoisting."""
        if slot in prepared:
            return prepared[slot]
        if slot in hoisted:
            return hoisted[slot]
        raise ValueError(f"score slot {slot} was not evaluated early")

    def evaluate(score, q_pos, kv_pos, with_gradient=False, prepared=None):
        return evaluate_score_program(
            **common,
            score=score,
            q_pos=q_pos,
            kv_pos=kv_pos,
            with_gradient=with_gradient,
            seeded_loads=tuple(seeded_loads) if with_gradient else (),
            cached_values={**hoisted, **prepared} if prepared else hoisted,
        )

    def bias_part(bias_slot, kinds, q_pos=None, kv_pos=None):
        """Sum, in log2 units, of the bias terms whose kind is in ``kinds``.

        Row terms need only ``q_pos``, column terms only ``kv_pos``. Returns
        None when no term matches.
        """
        terms = [
            term for term in score_bias_terms(program, bias_slot) if term[0] in kinds
        ]
        if not terms:
            return None
        needed = _score_slot_closure(
            program,
            [term[1] for term in terms]
            + [factor for term in terms for factor, _ in term[2]],
        ) - set(hoisted)
        values = dict(hoisted)
        if needed:
            values.update(
                evaluate_score_program(
                    **common,
                    score=None,
                    q_pos=q_pos,
                    kv_pos=kv_pos,
                    cached_values=hoisted,
                    only_slots=needed,
                )
            )
        total = None
        for _, leaf, factors, sign in terms:
            # Invariant coefficient first, so it is shared across positions.
            coefficient = fx.Float32(_SCORE_LOG2E * sign)
            for factor, divide in factors:
                coefficient = (
                    coefficient * _reciprocal(values[factor])
                    if divide
                    else coefficient * values[factor]
                )
            term = values[leaf] * coefficient
            total = term if total is None else total + term
        return total

    evaluate.prepare = prepare
    evaluate.slot = evaluate_slot
    evaluate.bias_part = bias_part
    return evaluate


def store_element(destination, index, value):
    # Kept out of traced kernel bodies: the AST rewriter treats a subscript
    # store inside a runtime ``if`` as rebinding the container.
    destination[index] = value


def make_capture_buffers(gview, sizes, first, buffer0, buffer1, buffer2, buffer3):
    """Views for ``len(sizes)`` captured buffers starting at kernel slot ``first``."""
    slots = (buffer0, buffer1, buffer2, buffer3)
    return [
        gview(slots[first + index], None, size, 1) for index, size in enumerate(sizes)
    ]


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
    common = {
        "mask_program": program,
        "mask_program_output": output,
        "mask_buffer_strides": strides,
        "mask_buffers": buffers,
        "load_i32": load_i32,
        "batch": batch,
        "head": head,
    }
    dependencies = _mask_position_dependencies(program)
    row_slots = {
        slot for slot in range(4, len(dependencies)) if dependencies[slot] <= {"q"}
    }
    column_slots = {
        slot for slot in range(4, len(dependencies)) if dependencies[slot] == {"kv"}
    }

    def evaluate(q_pos, kv_pos, cached_values=None):
        return evaluate_mask_program(
            **common, q_pos=q_pos, kv_pos=kv_pos, cached_values=cached_values
        )

    def row(q_pos):
        """Mask values that depend on the query row only (e.g. doc_id[q])."""
        return evaluate_mask_program(
            **common, q_pos=q_pos, kv_pos=None, only_slots=row_slots
        )

    def column(kv_pos):
        """Mask values that depend on the key column only (e.g. doc_id[kv])."""
        return evaluate_mask_program(
            **common, q_pos=None, kv_pos=kv_pos, only_slots=column_slots
        )

    evaluate.row = row
    evaluate.column = column
    return evaluate
