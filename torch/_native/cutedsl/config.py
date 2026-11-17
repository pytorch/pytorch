# Shared with declarations loaded before torch is built.
POINTWISE_ARCH_MAJORS: tuple[int, ...] = (10,)
POINTWISE_EPT_SCHEDULE = ((768 * 1024, 4), (384 * 1024, 2), (0, 1))
POINTWISE_VEC256_MIN = (32, 8 * 1024 * 1024)
POINTWISE_REFERENCE_LANES = 148 * 2048
POINTWISE_DENSE_MIN = 8 * 1024 * 1024
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
