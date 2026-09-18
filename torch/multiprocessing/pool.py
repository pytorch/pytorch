from multiprocessing.pool import Pool as _Pool


class Pool(_Pool):
    """Multiprocessing pool that passes tensors through shared memory."""


__all__ = ["Pool"]
