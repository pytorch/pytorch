"""Lane geometry shared by the FlyDSL row scatter-add kernel and its conds.

Kept free of FlyDSL imports so the dispatch conds can size a row without
loading FlyDSL, and so the conds and the kernel cannot disagree about it.

Each lane moves one 4-byte word per step (two halves or one fp32), lanes
taking consecutive words. A wave-wide atomic instruction therefore updates
contiguous addresses; with 16-byte vectors per lane, each atomic instruction
touched only every fourth word and v1 ran 2.5-3.9x slower than ATen on
gfx950.
"""

LANE_BYTES = 4
# Unrolled steps per lane; rows needing more are declined by the cond. At
# wave64 this is 32 KiB per row (bf16/fp16 N <= 16384, fp32 N <= 8192).
MAX_STEPS_PER_LANE = 128


def lane_elems(itemsize: int) -> int:
    """Elements one lane moves per step."""
    return LANE_BYTES // itemsize


def steps_per_lane(n: int, itemsize: int, wave: int) -> int:
    """Steps a lane takes to cover a row of ``n`` elements (``n`` whole words)."""
    words = n // lane_elems(itemsize)
    return (words + wave - 1) // wave
