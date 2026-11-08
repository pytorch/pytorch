"""Launch policy shared by the RMSNorm JIT adapter and AOT code generation."""

from functools import cache
from typing import NamedTuple


NORMALIZED_SIZES = (128, 512, 1024, 2048, 4096, 8192)

# Compute capabilities with a measured backward launch and weight-only crossover.
TUNED_ARCHS = ((9, 0), (10, 0), (10, 3))

# Weight-only calls need enough work to pay for the extra output and reduction.
# Rows from which the native kernel is faster, as (fp16/bf16, fp32); None means
# ATen was faster at every row count measured.
_WEIGHT_GRAD_MIN_ROWS: dict[int, tuple[int | None, int | None]] = {
    128: (8192, 8192),
    512: (8192, 8192),
    1024: (8192, 16384),
    2048: (16384, 16384),
}
# Measured on B300 (sm_103), where the native kernel is faster from fewer rows.
_WEIGHT_GRAD_MIN_ROWS_SM103: dict[int, tuple[int | None, int | None]] = {
    128: (1024, 1024),
    512: (2048, 2048),
    1024: (4096, 4096),
    2048: (16384, 8192),
    8192: (8192, None),
}


def weight_grad_min_rows(n: int, element_size: int, cc: tuple[int, int]) -> int | None:
    table = _WEIGHT_GRAD_MIN_ROWS_SM103 if cc == (10, 3) else _WEIGHT_GRAD_MIN_ROWS
    rows = table.get(n)
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
