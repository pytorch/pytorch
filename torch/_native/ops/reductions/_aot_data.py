"""Torch-free metadata shared by reduction AOT declarations and builders."""

ARCHS = ("sm_90", "sm_90a", "sm_100", "sm_100a")

# aten op, trait key, has dtype= argument, dimension spelling
CORE_OPS = (
    ("sum.dim_IntList", "sum", True, "optional_list"),
    ("mean.dim", "mean", True, "optional_list"),
    ("amax", "amax", False, "list"),
    ("amin", "amin", False, "list"),
    ("prod.dim_int", "prod", True, "int"),
)

AOT_ALIASES = {
    "sum.dim_IntList": ("sum.IntList_out",),
    "mean.dim": ("mean.out",),
    "amax": ("amax.out",),
    "amin": ("amin.out",),
    "prod.dim_int": ("prod.int_out",),
    "argmax": ("argmax.out",),
    "argmin": ("argmin.out",),
    "max.dim": ("max.dim_max",),
    "min.dim": ("min.dim_min",),
    "linalg_vector_norm": ("linalg_vector_norm.out",),
    "all.dim": ("all.out",),
    "any.dim": ("any.out",),
    "aminmax": ("aminmax.out",),
    "all.dims": ("all.dims_out",),
    "any.dims": ("any.dims_out",),
    "all": ("all.all_out",),
    "any": ("any.all_out",),
}

KERNEL_SOURCES = {
    "sum": "sum.dim_IntList",
    "mean": "mean.dim",
    "prod": "prod.dim_int",
    "max": "amax",
    "min": "amin",
    "all.dims": "all.dim",
    "all": "all.dim",
    "any.dims": "any.dim",
    "any": "any.dim",
    "count_nonzero": "count_nonzero.dim_IntList",
}

# aten op, trait key, dimension spelling, projected output dtypes, impl outputs
INDEX_OPS = (
    ("argmax", "argmax", "optional_int", ("int64",), ("out",)),
    ("argmin", "argmin", "optional_int", ("int64",), ("out",)),
    ("max.dim", "maxdim", "int", ("input", "int64"), ("max", "max_values")),
    ("min.dim", "mindim", "int", ("input", "int64"), ("min", "min_indices")),
)

# aten op, trait stem, projected output dtypes
WELFORD_OPS = (
    ("var.correction", "var", ("input",)),
    ("std.correction", "std", ("input",)),
    ("var_mean.correction", "varmean", ("input", "input")),
    ("std_mean.correction", "stdmean", ("input", "input")),
)

# aten op, trait key, dimension spelling, projected output dtypes, impl outputs
PLAIN_OPS = (
    ("all.dim", "all", "int", ("bool",), ("out",), True),
    ("any.dim", "any", "int", ("bool",), ("out",), True),
    ("count_nonzero.dim_IntList", "cnz", "list", ("int64",), (), False),
    (
        "aminmax",
        "aminmax",
        "optional_int",
        ("input", "input"),
        ("min", "max"),
        True,
    ),
    ("max", "amax", "full", ("input",), (), False),
    ("min", "amin", "full", ("input",), (), False),
    ("all.dims", "all", "optional_list", ("bool",), ("out",), True),
    ("any.dims", "any", "optional_list", ("bool",), ("out",), True),
    ("all", "all", "full", ("bool",), ("out",), False),
    ("any", "any", "full", ("bool",), ("out",), False),
    ("count_nonzero", "cnz", "optional_int", ("int64",), (), False),
)

DTYPE_OPS = (
    ("sum", "sum", "full", False),
    ("mean", "mean", "full", False),
    ("nansum", "nansum", "optional_list", True),
    ("prod", "prod", "full", False),
)

NORM_TRAITS = ("vnorm1", "vnorm2", "vnorminf", "vnormninf")
TRAIT_NFIELDS = {
    **{f"{stem}{correction}": 3 for _, stem, _ in WELFORD_OPS for correction in (0, 1)},
    "sum": 1,
    "mean": 1,
    "nansum": 1,
    "prod": 1,
    "amax": 1,
    "amin": 1,
    "vnorm1": 1,
    "vnorm2": 1,
    "vnorminf": 1,
    "vnormninf": 1,
    "all": 1,
    "any": 1,
    "cnz": 1,
    "aminmax": 2,
}

DTYPES = ("float16", "bfloat16", "float32")
DTYPE_BYTES = {
    "bool": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "int64": 8,
}
ORDER_SENSITIVE_TRAITS = frozenset(
    (
        "sum",
        "mean",
        "nansum",
        "prod",
        *NORM_TRAITS,
        *(f"{stem}{correction}" for _, stem, _ in WELFORD_OPS for correction in (0, 1)),
    )
)
INNER_TREE_LOADS = (1, 2, 4, 8)

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
_INNER_TREE_THRESHOLD = 8192
_INNER_TREE_MULTIROW_LOADS = 8
_INNER_TREE_MAX_BATCHES = 3
_INNER_TREE_WARP = 32
_INNER_TREE_TARGET_WARPS = 8


def _previous_power_of_2(n):
    return 1 << (n.bit_length() - 1)


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def inner_tree_spec(dtype, n, m):
    """Return the runtime-generic inner-tree shape and load class."""
    itemsize = DTYPE_BYTES.get(dtype)
    if itemsize is None or n < 1 or m < 1:
        return None
    vec = 16 // itemsize
    num_vecs = -(-n // vec)
    if num_vecs <= _INNER_TREE_MULTIROW_LOADS:
        return "itree_multirow", _next_power_of_2(num_vecs)

    wle = _INNER_TREE_WARP * vec
    if n > _INNER_TREE_THRESHOLD:
        estimated_batches = -(-n // _INNER_TREE_THRESHOLD)
        if estimated_batches <= _INNER_TREE_MAX_BATCHES:
            ideal_warps = sum(
                min(
                    16,
                    max(
                        1,
                        min(_INNER_TREE_THRESHOLD, n - b * _INNER_TREE_THRESHOLD)
                        // wle,
                    ),
                )
                for b in range(estimated_batches)
            )
            warps = max(1, ideal_warps // estimated_batches)
        else:
            warps = min(16, max(1, _INNER_TREE_THRESHOLD // wle))
    else:
        warps = min(16, max(1, n // wle))
    warps = _previous_power_of_2(warps)

    loads = _next_power_of_2(-(-n // (warps * wle)))
    max_loads = _previous_power_of_2(max(1, _INNER_TREE_THRESHOLD // (wle * warps)))
    loads = min(loads, max_loads)
    batch_elements = loads * wle * warps
    batches = -(-n // batch_elements)
    if batches <= _INNER_TREE_MAX_BATCHES:
        return "itree_looped", loads
    if m * batches >= 2**31:
        return None
    return "itree_split", loads


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
