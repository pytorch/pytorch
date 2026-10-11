# mypy: allow-untyped-defs

MASK_TRAVERSAL_UNMASKED = "unmasked"
MASK_TRAVERSAL_DIRECT_RANGE = "direct_range"
MASK_TRAVERSAL_BLOCK_LIST = "block_list"

DIRECT_RANGE_CAUSAL = "causal"

MAX_DQ_WORKSPACE_BYTES = 16 << 30


def choose_dq_partitions(
    batch_heads,
    sequence_length,
    key_rows,
    workspace_element_bytes=4,
    *,
    qk_head_dim=192,
):
    slot_bytes = batch_heads * sequence_length * qk_head_dim * workspace_element_bytes
    max_partitions = MAX_DQ_WORKSPACE_BYTES // slot_bytes
    if max_partitions == 0:
        raise ValueError("FlyDSL backward requires a dQ workspace larger than 16 GiB")
    max_partitions = 1 << (max_partitions.bit_length() - 1)
    owners = (sequence_length + key_rows - 1) // key_rows
    if key_rows == 128:
        # Bound sparse slot traffic and each buffer view for both supported dimensions.
        head_group = min(32, batch_heads & -batch_heads)
        max_slots = min(
            32,
            0xFFFFFFFF
            // (sequence_length * 192 * head_group * workspace_element_bytes),
        )
        return min(owners, max_partitions, 1 << (max_slots.bit_length() - 1))
    # Keep long owner loops distributed even when heads fill the device.
    target = max(1 if owners <= 8 else 2, (256 + batch_heads - 1) // batch_heads)
    return min(owners, max_partitions, 1 << (target - 1).bit_length())


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
                values.append(
                    binary(op, values[instruction[1]], values[instruction[2]])
                )
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

    expression = _canonical_mask_expression(
        mask_program, mask_program_output, sequence_length
    )
    if expression == ("bool", True):
        return MASK_TRAVERSAL_UNMASKED, None
    query = ("var", "query")
    key = ("var", "key")
    if expression in (("ge", query, key), ("le", key, query)):
        return MASK_TRAVERSAL_DIRECT_RANGE, DIRECT_RANGE_CAUSAL

    return MASK_TRAVERSAL_BLOCK_LIST, None


def make_bwd_shared_layout(rows, columns):
    import flydsl.expr as fx

    layout = fx.make_layout(
        ((16, rows // 16), (32, columns // 32)),
        ((32, 16 * columns), (1, 16 * 32)),
    )
    return fx.make_composed_layout(fx.static(fx.SwizzleType.get(1, 4, 4)), layout)


def bwd_exp2(value):
    import flydsl.expr as fx

    return fx.math.exp2(fx.Float32(value), fastmath=fx.FastMathFlags.afn)
