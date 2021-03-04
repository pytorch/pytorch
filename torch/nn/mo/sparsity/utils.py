import threading

def is_valid_linear_block_sparse_pattern(row_block_size, col_block_size):
    return (row_block_size == 1 and col_block_size == 4) or \
           (row_block_size == 8 and col_block_size == 1)

# This is a stop-gap measure as current flow does not allow module
# specific block sparse pattern.
# Infact there is no way to convey sparse pattern via module config
# of quantization flow. Thus using the global context to convey
# sparsity pattern.
# Once the flow supports it, this should be removed.
class QNNPACKLinearBlockSparsePattern:
    rlock = threading.RLock()
    row_block_size = 1
    col_block_size = 4
    prev_row_block_size = 1
    prev_col_block_size = 4

    def __init__(self, row_block_size=1, col_block_size=4):
        assert(is_valid_linear_block_sparse_pattern(row_block_size, col_block_size))
        QNNPACKLinearBlockSparsePattern.rlock.acquire()
        QNNPACKLinearBlockSparsePattern.prev_row_block_size = QNNPACKLinearBlockSparsePattern.row_block_size
        QNNPACKLinearBlockSparsePattern.prev_col_block_size = QNNPACKLinearBlockSparsePattern.col_block_size
        QNNPACKLinearBlockSparsePattern.row_block_size = row_block_size
        QNNPACKLinearBlockSparsePattern.col_block_size = col_block_size

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, backtrace):
        QNNPACKLinearBlockSparsePattern.row_block_size = QNNPACKLinearBlockSparsePattern.prev_row_block_size
        QNNPACKLinearBlockSparsePattern.col_block_size = QNNPACKLinearBlockSparsePattern.prev_col_block_size
        QNNPACKLinearBlockSparsePattern.rlock.release()

    @staticmethod
    def block_size():
        return QNNPACKLinearBlockSparsePattern.row_block_size, QNNPACKLinearBlockSparsePattern.col_block_size
