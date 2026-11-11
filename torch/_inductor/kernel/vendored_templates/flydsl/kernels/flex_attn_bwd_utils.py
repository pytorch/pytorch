# mypy: allow-untyped-defs

from torch.utils._ordered_set import OrderedSet

from . import flex_attn_target


MASK_TRAVERSAL_UNMASKED = "unmasked"
MASK_TRAVERSAL_DIRECT_RANGE = "direct_range"
MASK_TRAVERSAL_BLOCK_LIST = "block_list"

DIRECT_RANGE_CAUSAL = "causal"

MAX_DQ_WORKSPACE_BYTES = 16 << 30
MAX_BUFFER_VIEW_BYTES = 0xFFFFFFFF
# The reverse block-list builder keeps one int32 flag per Q block in LDS.
MAX_REVERSE_LIST_QUERY_BLOCKS = 16384
# Kernels take this many capture tensors directly. Each keeps a 4-SGPR buffer
# descriptor live across the attention loops, so more captures are read through
# one table of base addresses instead.
MAX_DIRECT_CAPTURES = 4
# (QK, V) head dims for both directions. The backward needs QK % 32 == 0 and
# V >= 64, and its dense LDS footprint must fit gfx950's LDS.
SUPPORTED_HEAD_DIMS = (
    (64, 64),
    (64, 128),
    (64, 256),
    (96, 96),
    (96, 128),
    (128, 64),
    (128, 128),
    (128, 256),
    (192, 128),
    (192, 192),
    (256, 128),
    (256, 256),
)
# (QK, V) head dims above 256: two backward producer waves accumulate one
# 32-column-aligned half of dK/dV each; the forward runs V in slices.
WIDE_HEAD_DIMS = tuple(
    (qk_head_dim, value_head_dim)
    for qk_head_dim in (64, 128, 256, 384, 512)
    for value_head_dim in (64, 128, 256, 384, 512)
    if max(qk_head_dim, value_head_dim) > 256
)
# A producer wave holds (QK + V) / 2 f32 dK/dV accumulators per lane; above
# this (or a wider V) two waves per SIMD spill, so the CTA runs one per SIMD.
_MAX_SHARED_SIMD_ACCUMULATORS = 160
# Rows the forward leaves empty store LSE = -inf, and a -inf score minus that
# is NaN. Applied after any ln -> log2 conversion, which would overflow it.
LSE_FLOOR = -3.0e38


def dq_workspace_head_group(batch_heads):
    return min(32, batch_heads & -batch_heads)


def backward_column_slices(qk_head_dim, value_head_dim):
    """Return how many producer waves share each 32-row key tile, one dK/dV
    column slice each."""
    return 2 if (qk_head_dim, value_head_dim) in WIDE_HEAD_DIMS else 1


def backward_single_pass(qk_head_dim, value_head_dim, target=None):
    """Whether a ``WIDE_HEAD_DIMS`` backward gives two producers 16 full-width rows."""
    # 32 K/V rows and two 32-row Q/dO stages must fit in LDS with dS and P.
    return (qk_head_dim, value_head_dim) in WIDE_HEAD_DIMS and (
        192 * (qk_head_dim + value_head_dim)
        + 2 * 32 * 40 * 2
        + 32 * 16 * 2
        + 2 * 64 * 4
        <= flex_attn_target.resolve_target(target).lds_bytes
    )


def backward_producer_waves(block_list, qk_head_dim, value_head_dim, captures=0):
    """Return the dK/dV producer waves of one backward compute CTA, whose
    score_mod and mask_mod read ``captures`` tensors."""
    if (
        max((qk_head_dim + value_head_dim) // 2, value_head_dim)
        > _MAX_SHARED_SIMD_ACCUMULATORS
    ):
        return 2
    # Above QK 128, two waves per SIMD leave too few registers for the
    # capture reads of a block list through a capture table, or of a dense or
    # causal traversal's score_mod: they spill inside the query loop.
    if qk_head_dim > 128 and (
        captures > MAX_DIRECT_CAPTURES if block_list else captures > 0
    ):
        return 2
    return 4 if block_list else 6


def backward_key_rows(block_list, qk_head_dim, value_head_dim, captures=0):
    """Return the KV rows one backward compute owner stages (16 per key group)."""
    return (
        backward_producer_waves(block_list, qk_head_dim, value_head_dim, captures)
        * 32
        // backward_column_slices(qk_head_dim, value_head_dim)
    )


def forward_value_slice(qk_head_dim, value_head_dim, target=None):
    """Return the V columns of one forward kernel launch, at most 256 and
    narrower where its double-buffered 64-row 2-byte K and V tiles would
    overflow LDS."""
    if max(qk_head_dim, value_head_dim) <= 256:
        return value_head_dim
    lds_bytes = flex_attn_target.resolve_target(target).lds_bytes
    width = 256
    while width > 128 and 2 * 64 * (qk_head_dim + width) * 2 > lds_bytes:
        width //= 2
    while value_head_dim % width:
        width //= 2
    return width


def single_pass_value_shards(qk_head_dim, value_head_dim, lean=False):
    """Waves that split one query row's output columns in the single-pass forward."""
    # One wave holding them all spills, but for (384, 384) in a ``lean`` build:
    # a static prefill with Sq and Sk multiples of 128 and no mods or captures.
    if value_head_dim == 512 or (
        value_head_dim == 384 and not (lean and qk_head_dim == 384)
    ):
        return 2
    return 1


def forward_single_pass(qk_head_dim, value_head_dim, target=None):
    """Whether one launch computes S fewer times per row than the V slices do."""
    return (qk_head_dim, value_head_dim) in WIDE_HEAD_DIMS and value_head_dim // (
        forward_value_slice(qk_head_dim, value_head_dim, target)
    ) > single_pass_value_shards(qk_head_dim, value_head_dim)


def forward_head_dims(target=None):
    """(QK, V) head dims of one forward kernel launch."""
    return SUPPORTED_HEAD_DIMS + tuple(
        sorted(
            OrderedSet(
                [(qk, forward_value_slice(qk, v, target)) for qk, v in WIDE_HEAD_DIMS]
            )
            - OrderedSet(SUPPORTED_HEAD_DIMS)
        )
    )


def split_dq_tiles(qk_head_dim, consumer_waves):
    """Split one dQ consumer wave's 16-column tiles into (phases, tiles per phase)."""
    wave_tiles = qk_head_dim // (16 * consumer_waves)
    phases = 2 if wave_tiles % 2 == 0 else 1
    return phases, wave_tiles // phases


def choose_dq_partitions(
    batch_heads,
    sequence_length,
    key_rows,
    workspace_element_bytes=4,
    *,
    qk_head_dim=192,
    key_sequence_length=None,
    compute_batch_heads=None,
    block_list=False,
    target=None,
):
    """Pick the number of private dQ slots, each holding dQ of every query head;
    occupancy counts ``compute_batch_heads`` (one compute CTA per KV head)."""
    if key_sequence_length is None:
        key_sequence_length = sequence_length
    if compute_batch_heads is None:
        compute_batch_heads = batch_heads
    slot_bytes = batch_heads * sequence_length * qk_head_dim * workspace_element_bytes
    max_partitions = MAX_DQ_WORKSPACE_BYTES // slot_bytes
    if max_partitions == 0:
        raise ValueError("FlyDSL backward requires a dQ workspace larger than 16 GiB")
    max_partitions = 1 << (max_partitions.bit_length() - 1)
    # All slots of one workspace head group are addressed through one buffer view.
    head_group = dq_workspace_head_group(batch_heads)
    view_partitions = MAX_BUFFER_VIEW_BYTES // (
        sequence_length * qk_head_dim * head_group * workspace_element_bytes
    )
    if view_partitions == 0:
        raise ValueError(
            "FlyDSL backward requires a dQ workspace view beyond the 32-bit buffer range"
        )
    max_partitions = min(max_partitions, 1 << (view_partitions.bit_length() - 1))
    owners = (key_sequence_length + key_rows - 1) // key_rows
    # Enough slots for one wave of compute CTAs: each slot costs a zeroing and
    # a reduction pass over dQ. Long owner loops stay distributed even when
    # heads fill the device.
    num_cus = flex_attn_target.resolve_target(target).num_cus
    partitions = max(1 if owners <= 8 else 2, -(-num_cus // compute_batch_heads))
    partitions = 1 << (partitions - 1).bit_length()
    if block_list:
        # Bound sparse slot traffic as for a D192 slot; view_partitions bounds wider ones.
        max_slots = max(
            1,
            min(
                32,
                MAX_BUFFER_VIEW_BYTES
                // (sequence_length * 192 * head_group * workspace_element_bytes),
            ),
        )
        # Up to two waves of owners, a slot per owner is faster than one wave
        # of shared slots (by 15% at 16 heads x 32 causal owners).
        if compute_batch_heads * owners <= 2 * num_cus:
            partitions = owners
        partitions = min(partitions, 1 << (max_slots.bit_length() - 1))
    return min(owners, max_partitions, partitions)


def _canonical_mask_expression(
    mask_program, mask_program_output, sequence_length=None, key_sequence_length=None
):
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
    if key_sequence_length is None:
        key_sequence_length = sequence_length
    lengths = {"query": sequence_length, "key": key_sequence_length}

    # Unknown loads or possible int32 overflow must retain BlockMask traversal.
    def interval(expression):
        op = expression[0]
        if op == "int":
            value = expression[1]
            return (value, value) if -(1 << 31) <= value < (1 << 31) else None
        if op == "var" and expression[1] in lengths:
            length = lengths[expression[1]]
            if length is not None and 0 < length <= (1 << 31):
                return (0, length - 1)
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
    key_sequence_length=None,
    block_mask_is_dense=False,
):
    """Return ``(traversal, direct_range_kind)`` for a lowered mask.

    This intentionally recognizes capabilities rather than Python function
    names.  Unsupported or ambiguous expressions conservatively use the
    BlockMask work-list traversal.  Mask expressions alone cannot prove that
    user-provided BlockMask metadata contains every block allowed by the
    expression, so direct traversal is safe only for metadata synthesized as
    dense by this backend.
    """

    if not block_mask_is_dense:
        return MASK_TRAVERSAL_BLOCK_LIST, None

    mask_program = tuple(mask_program)
    if not mask_program and not mask_buffer_shapes:
        return MASK_TRAVERSAL_UNMASKED, None

    expression = _canonical_mask_expression(
        mask_program, mask_program_output, sequence_length, key_sequence_length
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
