"""Launch policy shared by the RMSNorm JIT adapter and AOT code generation."""

from functools import cache
from typing import NamedTuple


NORMALIZED_SIZES = (128, 512, 1024, 2048, 4096, 8192)

# Weight-only calls need enough work to pay for the extra output and reduction.
_WEIGHT_GRAD_MIN_ROWS = {
    128: (8192, 8192),
    512: (8192, 8192),
    1024: (8192, 16384),
    2048: (16384, 16384),
}


def weight_grad_min_rows(n: int, element_size: int) -> int | None:
    rows = _WEIGHT_GRAD_MIN_ROWS.get(n)
    return rows[element_size == 4] if rows is not None else None


class BackwardLaunch(NamedTuple):
    rows_per_block: int
    blocks_per_sm: int
    large_rows: int
    large_blocks_per_sm: int

    def blocks(self, rows: int, sm_count: int) -> int:
        multiplier = (
            self.large_blocks_per_sm if rows >= self.large_rows else self.blocks_per_sm
        )
        return min(
            sm_count * multiplier,
            (rows + self.rows_per_block - 1) // self.rows_per_block,
        )


@cache
def backward_launch(n: int, compute_dw: bool, element_size: int = 2) -> BackwardLaunch:
    rows_per_block = 8 if n == 128 else 2 if n == 512 else 1
    if not compute_dw:
        multiplier = 16 if n <= 256 else 8 if n <= 1024 else 4 if n <= 2048 else 2
        return BackwardLaunch(rows_per_block, multiplier, 0, multiplier)
    # Bound partial-reduction work while giving longer rows enough parallelism.
    multiplier = 2 if n == 128 else 4 if n <= 2048 else 1
    large_multiplier = 4 if n <= 512 else 8 if n == 1024 else 4 if n == 2048 else 2
    large_rows = 1024 if n >= 4096 and (n == 4096 or element_size == 2) else 8192
    return BackwardLaunch(rows_per_block, multiplier, large_rows, large_multiplier)
