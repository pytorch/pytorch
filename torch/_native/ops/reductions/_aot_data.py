"""Torch-free metadata shared by reduction AOT declarations and builders."""

# Compute capabilities 9.x and 10.x, matching overrides._ARCH_MAJORS.
ARCHS = ("sm_90", "sm_100f")

# Welford reductions remain JIT-only: their correction overloads are unstructured,
# and no STRUCTURED=False declaration covers them yet.
# aten op, trait key, has dtype= argument, dimension spelling
CORE_OPS = (
    ("sum.dim_IntList", "sum", True, "optional_list"),
    ("mean.dim", "mean", True, "optional_list"),
    ("amax", "amax", False, "list"),
    ("amin", "amin", False, "list"),
    ("prod.dim_int", "prod", True, "int"),
)

DTYPES = ("float16", "bfloat16", "float32")
DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}

ROW_CLASSES = {
    "t32_b128": (32, 128),
    "t64_b64": (64, 64),
    "t64_b128": (64, 128),
    "t128_b128": (128, 128),
    "t256_b256": (256, 256),
}

_MAX_ROW_BYTES = 192 * 1024
_MAX_LOADS = 65
XCTA_SUBROW = 8192
_XCTA_MAX_PARTIALS = 1 << 22


def _ladder(n, itemsize):
    """kernel_rowtile.row_config's (threads per row, threads per block)."""
    if n * itemsize >= 16 * 1024:
        return 256, 256
    tpr = 32 if n <= 3072 else 64 if n <= 6144 else 128 if n <= 16384 else 256
    return tpr, max(tpr, 128 if n <= 16384 else 256)


def row_class(dtype, n, m, reduce_all=False):
    """Select the one-shot row launch class, or None to retain JIT routing."""
    itemsize = DTYPE_BYTES.get(dtype)
    if (
        itemsize is None
        or n < 1
        or m < 1
        or n >= 2**31
        or m >= 2**31
        or n % (16 // itemsize)
        or n * itemsize > _MAX_ROW_BYTES
    ):
        return None

    vec = 16 // itemsize
    chunks = n // vec
    narrow = n <= 256 and (
        (m >= 65536 and chunks <= 32)
        or (m >= 16384 and chunks <= 16)
        or (m >= 4096 and chunks <= 6)
    )
    if narrow and not reduce_all:
        return None

    threads_per_row, threads_per_block = _ladder(n, itemsize)
    if reduce_all:
        feedable = min(256, n // vec)
        wider = max(
            (threads for threads in (32, 64, 128, 256) if threads <= feedable),
            default=threads_per_row,
        )
        if wider > threads_per_row:
            threads_per_row = threads_per_block = wider

    if -(-n // (threads_per_row * vec)) > _MAX_LOADS:
        return None
    return f"t{threads_per_row}_b{threads_per_block}"


def xcta_ok(dtype, n, m):
    """Whether the shared 8K sub-row split can serve this runtime geometry."""
    itemsize = DTYPE_BYTES.get(dtype)
    if (
        itemsize is None
        or n < 2 * XCTA_SUBROW
        or n >= 2**31
        or m < 1
        or m >= 2**31
        or n % XCTA_SUBROW
    ):
        return False
    partials = n // XCTA_SUBROW
    return partials <= _XCTA_MAX_PARTIALS and m * partials < 2**31


# One runtime-width ragged row kernel per launch class serves rows whose width or pitch
# is not whole 16-byte vectors, in the JIT's B200 band
# (kernel_general._ragged_rows_ok for these traits) with row_config's ladder.
RAGGED_CLASSES = {
    "t32_b128": (32, 128),
    "t64_b128": (64, 128),
    "t128_b128": (128, 128),
    "t256_b256": (256, 256),
}
RAGGED_MIN_N = 255
RAGGED_EVEN_MIN_N = 1023
RAGGED_MAX_N = 16383


def ragged_class(dtype, n, m, pitch):
    """Select the ragged launch class for m rows of n at a pitch, or None."""
    itemsize = DTYPE_BYTES.get(dtype)
    # Single rows take the JIT's reduce_all instead.
    if itemsize is None or m < 2 or m >= 2**31:
        return None
    vec = 16 // itemsize
    if n % vec == 0 and pitch % vec == 0:
        return None  # whole vectors: the row classes serve it
    # Even widths start later where a JIT vector-class kernel serves them.
    align = min(16, (pitch * itemsize) & -(pitch * itemsize))
    late = n % 2 == 0 and align >= min(16, (n * itemsize) & -(n * itemsize))
    min_n = RAGGED_EVEN_MIN_N if late else RAGGED_MIN_N
    if not min_n <= n <= RAGGED_MAX_N:
        return None
    return "t{}_b{}".format(*_ladder(n, itemsize))
