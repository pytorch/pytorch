"""Launch policy shared by the RMSNorm JIT adapter and AOT code generation."""

from typing import NamedTuple


NORMALIZED_SIZES = (128, 512, 1024, 2048, 4096, 8192)


class BackwardLaunch(NamedTuple):
    rows_per_block: int
    blocks_per_sm: int

    def blocks(self, rows: int, sm_count: int) -> int:
        return min(
            sm_count * self.blocks_per_sm,
            (rows + self.rows_per_block - 1) // self.rows_per_block,
        )


def backward_launch(n: int, compute_dw: bool) -> BackwardLaunch:
    rows_per_block = 8 if n == 128 else 2 if n == 512 else 1
    blocks_per_sm = (
        1
        if compute_dw
        else 16
        if n <= 256
        else 8
        if n <= 1024
        else 4
        if n <= 2048
        else 2
    )
    return BackwardLaunch(rows_per_block, blocks_per_sm)
