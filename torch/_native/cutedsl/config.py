# Shared with declarations loaded before torch is built.
POINTWISE_ARCH_MAJORS: tuple[int, ...] = (10,)
POINTWISE_REFERENCE_LANES = 148 * 2048
POINTWISE_DENSE_MIN = 8 * 1024 * 1024
POINTWISE_PADDED_TMA_MIN = 16 * 1024 * 1024
POINTWISE_TMA_SHAPES = ((64, 96), (1024, 1024))
TMA_TILE = 32
TMA_STAGES = 6
TMA_THREADS = 128
TMA_SMEM_ALIGNMENT = 1024
RNG_BLOCK = 256
TRANSPOSE_VECTOR_MIN_BYTES = {1: 256 * 1024, 2: 256 * 1024, 4: 4 * 1024 * 1024}
TRANSPOSE_TMA_MIN_ELEMS = {1: 4 * 1024 * 1024, 2: 4 * 1024 * 1024, 4: 32 * 1024 * 1024}


def pointwise_aot_archs() -> tuple[str, ...]:
    return tuple(
        f"sm_{major * 10}{'f' if major >= 10 else ''}"
        for major in POINTWISE_ARCH_MAJORS
    )


# B200 graph measurements retain the smallest set within 1% of existing presets.
PointwisePresets = tuple[tuple[int, int, int], ...]
_M = 1024**2
_FP64_PRESETS: dict[str, PointwisePresets] = {
    name: presets
    for names, presets in (
        ("neg", ((0, 1, 128), (4 * _M, 2, 128))),
        (
            "abs round sqrt rsqrt rad2deg sinc div fmin hypot lerp",
            ((0, 1, 128), (2 * _M, 2, 128)),
        ),
        ("sign floor ceil frac angle", ((0, 1, 128), (2 * _M, 4, 128))),
        (
            "trunc reciprocal maximum heaviside",
            ((0, 1, 128), (2 * _M, 2, 128), (4 * _M, 4, 128)),
        ),
        ("exp erf", ((0, 1, 128), (8 * _M, 4, 256))),
        ("exp2", ((0, 1, 128), (8 * _M, 4, 128))),
        ("expm1 tan sigmoid cosh", ((0, 1, 128), (8 * _M, 2, 128))),
        ("log", ((0, 1, 128), (16 * _M, 2, 128))),
        (
            (
                "log2 log10 log1p sin cos asin acos tanh erfc asinh acosh atanh "
                "mish deg2rad entr atan2 logaddexp logaddexp2 xlogy pow xlog1py "
                "addcmul addcdiv"
            ),
            ((0, 1, 128),),
        ),
        ("atan silu", ((0, 1, 128), (16 * _M, 4, 128))),
        ("sinh", ((0, 1, 128), (2 * _M, 2, 128), (8 * _M, 4, 256))),
        ("add sub rsub mul minimum fmax", ((0, 1, 128), (4 * _M, 4, 128))),
    )
    for name in names.split()
}
_OP_ALIASES = {
    "roundeven": "round",
    "special_entr": "entr",
    "special_xlog1py": "xlog1py",
    "clamp_min": "maximum",
    "clamp_max": "minimum",
}


def pointwise_presets(
    op: str,
    compute_bits: int,
    storage_bits: int,
    nin: int,
    *,
    dtype: str = "",
    rowgap: bool = False,
) -> PointwisePresets:
    """Return ascending (minimum elements, work per thread, vector bits) presets."""
    name = op.removeprefix("_").split(".")[0]
    name = _OP_ALIASES.get(name, name)
    if rowgap:
        if (
            name == "add"
            and dtype == "float16"
            and compute_bits == 32
            and storage_bits < 32
        ):
            return ((0, 1, 128), (2 * 1024**2, 4, 256))
        return ((0, 1, 128),)
    if compute_bits == 64:
        if dtype == "float64" and name in _FP64_PRESETS:
            return _FP64_PRESETS[name]
        return (
            (0, 1, 128),
            (1536 * 1024, 2, 128),
            (3 * 1024**2, 4, 128),
            (8 * 1024**2, 4, 256),
        )
    if compute_bits == 32:
        if name == "sin" and storage_bits >= 32:
            return ((0, 1, 128),)
        if name == "sin" and dtype == "float16":
            return ((0, 1, 128), (4 * 1024**2, 2, 128))
        if storage_bits < 32 and nin == 1:
            return ((0, 1, 128), (POINTWISE_DENSE_MIN, 4, 256))
        return ((0, 1, 128), (POINTWISE_DENSE_MIN, 2, 128))
    return ((0, 1, 128), (384 * 1024, 2, 128), (768 * 1024, 4, 128))
