from multiprocessing.queues import Queue as _Queue, SimpleQueue as _SimpleQueue


class Queue(_Queue):
    """Multiprocessing queue that passes tensors through shared memory."""


class SimpleQueue(_SimpleQueue):
    """Multiprocessing simple queue that passes tensors through shared memory."""


__all__ = ["Queue", "SimpleQueue"]
