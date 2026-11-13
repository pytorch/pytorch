"""Torch-free metadata shared by reduction AOT declarations and builders."""

# Compute capability 10.x, matching overrides._ARCH_MAJORS.
ARCHS = ("sm_100f",)

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

    if n * itemsize >= 16 * 1024:
        threads_per_row, threads_per_block = 256, 256
    else:
        if n <= 3072:
            threads_per_row = 32
        elif n <= 6144:
            threads_per_row = 64
        elif n <= 16384:
            threads_per_row = 128
        else:
            threads_per_row = 256
        threads_per_block = 128 if n <= 16384 else 256
        threads_per_block = max(threads_per_row, threads_per_block)

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
