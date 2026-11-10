"""Pointwise launch limits shared by JIT and torch-free AOT declarations."""

_MIN_ALIGNMENT = 16
_TMA_ALIGNMENT = 128
_VECTOR_BYTES = 16
_MAX_VECTOR_BYTES = 32
_MAX_VECTOR_ELEMENTS = 8
_DEFAULT_VECTOR_EPT = 8
_UNROLLED_EPT = 4
_VECTOR_INPUT_BYTES_PER_THREAD = 128
_SM107_MIN_IO_BYTES = 16 * 1024 * 1024
_TMA_BOX_SIDE = 32
_THREADS_PER_BLOCK = 128
_ELEMENTS_PER_THREAD = 8
_TILES_PER_BLOCK = 6
_SMEM_ALIGNMENT = 1024
_BARRIER_BYTES = 8
_MAX_GRID_YZ = 65535
_MAX_INDEX = (1 << 31) - 1
_WARP_SIZE = 32
_SWIZZLE_BASE_LOG2 = 4
_SWIZZLE_SHIFT = 3
_SWIZZLE_ATOM_BYTES = 128
_FP32_LANES_PER_GROUP = 8
_FP16_LANES_PER_GROUP = 16
_FP16_ELEMENTS_PER_BANK = 2

# (tile side, threads, pipeline stages), selected by runtime compute capability.
_TMA_SM100_TRANSPOSE = (64, 256, 1)
_TMA_SM107_TRANSPOSE = (64, 128, 2)
_TMA_SM100_POINTWISE = (32, 128, 1)
_TMA_SM100_LARGE_POINTWISE = (32, 128, 8)
_TMA_SM107_POINTWISE = (64, 128, 1)
_SM100_VECTOR_THREADS = 256
_SM100_VECTOR_BYTES_PER_THREAD = 32
_TMA_BF16_MIN_STAGES = 2
_TMA_LARGE_INPUT_BYTES = 256 * 1024 * 1024
_TMA_SM107_LARGE_BOX = 128
_TMA_LARGE_BYTES_PER_THREAD = 256


def _tma_configs(
    transposed: tuple[bool | None, ...],
    cc: tuple[int, int],
    element_bytes: int,
    matrix_elements: int | None = None,
) -> tuple[tuple[int, int, int], ...]:
    # A 32-element fallback retains layouts that cannot fit the preferred tile.
    transpose = all(value for value in transposed if value is not None)
    if cc == (10, 7):
        config = _TMA_SM107_TRANSPOSE if transpose else _TMA_SM107_POINTWISE
    else:
        config = _TMA_SM100_TRANSPOSE if transpose else _TMA_SM100_POINTWISE
    if element_bytes == 2:
        config = (config[0], config[1], max(config[2], _TMA_BF16_MIN_STAGES))
    fallback = (_TMA_BOX_SIDE, _THREADS_PER_BLOCK, config[2])
    configs = (config,) if config == fallback else (config, fallback)
    large_input = (
        matrix_elements is None
        or matrix_elements * element_bytes >= _TMA_LARGE_INPUT_BYTES
    )
    if cc != (10, 7) and not transpose and large_input:
        return (_TMA_SM100_LARGE_POINTWISE, *configs)
    if cc == (10, 7) and transpose and (large_input):
        # Large transpose tiles amortize transfers with a fixed byte budget per thread.
        box = _TMA_SM107_LARGE_BOX
        large = (box, box * box * element_bytes // _TMA_LARGE_BYTES_PER_THREAD, 1)
        return (large, *configs)
    return configs
