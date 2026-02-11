import threading


__all__ = ["LinearBlockSparsePattern"]


def _is_valid_linear_block_sparse_pattern(
    row_block_size: int, col_block_size: int
) -> bool:
    return (row_block_size == 1 and col_block_size == 4) or (
        row_block_size == 8 and col_block_size == 1
    )


# This is a stop-gap measure as current flow does not allow module
# specific block sparse pattern.
# In fact there is no way to convey sparse pattern via module config
# of quantization flow. Thus using the global context to convey
# sparsity pattern.
# Once the flow supports it, this should be removed.
class LinearBlockSparsePattern:
    rlock = threading.RLock()
    row_block_size: int = 1
    col_block_size: int = 4
    prev_row_block_size: int = 1
    prev_col_block_size: int = 4

    def __init__(self, row_block_size: int = 1, col_block_size: int = 4):
        if not _is_valid_linear_block_sparse_pattern(row_block_size, col_block_size):
            raise AssertionError(
                f"Invalid linear block sparse pattern: "
                f"row_block_size={row_block_size}, col_block_size={col_block_size}"
            )
        LinearBlockSparsePattern.rlock.acquire()
        LinearBlockSparsePattern.prev_row_block_size = (
            LinearBlockSparsePattern.row_block_size
        )
        LinearBlockSparsePattern.prev_col_block_size = (
            LinearBlockSparsePattern.col_block_size
        )
        LinearBlockSparsePattern.row_block_size = row_block_size
        LinearBlockSparsePattern.col_block_size = col_block_size

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        backtrace: object | None,
    ) -> None:
        LinearBlockSparsePattern.row_block_size = (
            LinearBlockSparsePattern.prev_row_block_size
        )
        LinearBlockSparsePattern.col_block_size = (
            LinearBlockSparsePattern.prev_col_block_size
        )
        LinearBlockSparsePattern.rlock.release()

    @staticmethod
    def block_size() -> tuple[int, int]:
        return (
            LinearBlockSparsePattern.row_block_size,
            LinearBlockSparsePattern.col_block_size,
        )
