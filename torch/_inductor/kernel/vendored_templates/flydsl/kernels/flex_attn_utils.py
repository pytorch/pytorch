# mypy: allow-untyped-defs

import flydsl.expr as fx
from flydsl.expr import const_expr

MASK_TRAVERSAL_UNMASKED = "unmasked"
MASK_TRAVERSAL_DIRECT_RANGE = "direct_range"
MASK_TRAVERSAL_BLOCK_LIST = "block_list"

DIRECT_RANGE_CAUSAL = "causal"


def _canonical_mask_expression(mask_program, mask_program_output, sequence_length=None):
    """Build a small canonical expression tree for mask-shape recognition.

    Mask bytecode value ids are an implementation detail of lowering.  Keeping
    the traversal classifier independent of those ids lets harmless changes
    such as ``True & predicate`` folding share the same fast path.
    """

    values = [
        ("var", "batch"),
        ("var", "head"),
        ("var", "query"),
        ("var", "key"),
    ]

    # Unknown loads or possible int32 overflow must retain BlockMask traversal.
    def interval(expression):
        op = expression[0]
        if op == "int":
            value = expression[1]
            return (value, value) if -(1 << 31) <= value < (1 << 31) else None
        if op == "var" and expression[1] in ("query", "key"):
            if sequence_length is not None and 0 < sequence_length <= (1 << 31):
                return (0, sequence_length - 1)
            return None
        if op not in ("add", "sub", "mul"):
            return None
        left, right = interval(expression[1]), interval(expression[2])
        if left is None or right is None:
            return None
        if op == "add":
            bounds = (left[0] + right[0], left[1] + right[1])
        elif op == "sub":
            bounds = (left[0] - right[1], left[1] - right[0])
        else:
            products = [x * y for x in left for y in right]
            bounds = (min(products), max(products))
        return bounds if -(1 << 31) <= bounds[0] <= bounds[1] < (1 << 31) else None

    def binary(op, lhs, rhs):
        if op in ("lt", "le", "gt", "ge"):
            left, right = interval(lhs), interval(rhs)
            if left is not None and right is not None:
                if op in ("gt", "ge"):
                    left, right = right, left
                strict = op in ("lt", "gt")
                always_true = left[1] < right[0] if strict else left[1] <= right[0]
                always_false = left[0] >= right[1] if strict else left[0] > right[1]
                if always_true:
                    return ("bool", True)
                if always_false:
                    return ("bool", False)
        if op == "and":
            if lhs == ("bool", True):
                return rhs
            if rhs == ("bool", True):
                return lhs
            if lhs == ("bool", False) or rhs == ("bool", False):
                return ("bool", False)
        elif op == "or":
            if lhs == ("bool", False):
                return rhs
            if rhs == ("bool", False):
                return lhs
            if lhs == ("bool", True) or rhs == ("bool", True):
                return ("bool", True)
        elif op == "add":
            if lhs == ("int", 0):
                return rhs
            if rhs == ("int", 0):
                return lhs
        elif op == "sub":
            if rhs == ("int", 0):
                return lhs
        elif op == "mul":
            if lhs == ("int", 0) or rhs == ("int", 0):
                return ("int", 0)
            if lhs == ("int", 1):
                return rhs
            if rhs == ("int", 1):
                return lhs
        if op in ("and", "or", "add", "mul", "eq", "ne") and repr(lhs) > repr(rhs):
            lhs, rhs = rhs, lhs
        return (op, lhs, rhs)

    try:
        for instruction in mask_program:
            op = instruction[0]
            if op == "const_bool":
                values.append(("bool", bool(instruction[1])))
            elif op == "const_i32":
                values.append(("int", int(instruction[1])))
            elif op == "load_i32":
                values.append(
                    (
                        "load_i32",
                        int(instruction[1]),
                        tuple(values[index] for index in instruction[2]),
                    )
                )
            elif op == "not":
                operand = values[instruction[1]]
                if operand[0] == "bool":
                    values.append(("bool", not operand[1]))
                else:
                    values.append(("not", operand))
            else:
                values.append(binary(op, values[instruction[1]], values[instruction[2]]))
        return values[int(mask_program_output)]
    except (IndexError, TypeError, ValueError):
        return None


def classify_mask_traversal(
    mask_program,
    mask_program_output,
    mask_buffer_shapes=(),
    *,
    sequence_length=None,
):
    """Return ``(traversal, direct_range_kind)`` for a lowered mask.

    This intentionally recognizes capabilities rather than Python function
    names.  Unsupported or ambiguous expressions conservatively use the
    BlockMask work-list traversal.
    """

    mask_program = tuple(mask_program)
    if not mask_program and not mask_buffer_shapes:
        return MASK_TRAVERSAL_UNMASKED, None

    expression = _canonical_mask_expression(mask_program, mask_program_output, sequence_length)
    if expression == ("bool", True):
        return MASK_TRAVERSAL_UNMASKED, None
    query = ("var", "query")
    key = ("var", "key")
    if expression in (("ge", query, key), ("le", key, query)):
        return MASK_TRAVERSAL_DIRECT_RANGE, DIRECT_RANGE_CAUSAL

    return MASK_TRAVERSAL_BLOCK_LIST, None


def make_global_view(tensor, coord, shape, stride):
    view = fx.make_view(fx.get_iter(tensor), fx.make_layout(shape, stride))
    if coord is not None:
        # Slice the global pointer before creating the 32-bit buffer descriptor.
        # Callers widen batch/head coordinates to i64; local strides stay static.
        view = fx.slice(view, coord)
    num_records_bytes = (fx.get_scalar(fx.cosize(view.layout)) * view.element_type.width + 7) // 8
    if not 0 <= num_records_bytes <= 0xFFFFFFFF:
        raise ValueError("FlyDSL buffer view must fit in a 32-bit byte range")
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


def make_value_shared_layout(rows, columns):
    layout = fx.make_layout(
        ((16, rows // 16), (32, columns // 32)),
        ((32, 16 * columns), (1, 16 * 32)),
    )
    return fx.make_composed_layout(fx.static(fx.SwizzleType.get(1, 4, 4)), layout)


def make_dq_workspace_layout(rows, columns):
    """Pack each MFMA16 C fragment into adjacent BF16 atomic pairs.

    The logical modes remain (query, dimension); the layout also defines the
    inverse scatter used to restore the caller's dQ strides.
    """
    return fx.make_layout(
        ((16, rows // 16), ((2, 2, 4), columns // 16)),
        ((2, 16 * columns), ((1, 128, 32), 256)),
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
                offset = offset + values[index_ids[dimension]] * fx.Int32(mask_buffer_strides[buffer_index][dimension])
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


def fast_exp2(value):
    return fx.math.exp2(fx.Float32(value), fastmath=fx.FastMathFlags.afn)


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
