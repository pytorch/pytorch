import multiprocessing
import multiprocessing.pool
import multiprocessing.util as util

from .queue import SimpleQueue


def clean_worker(*args, **kwargs):
    import gc
    multiprocessing.pool.worker(*args, **kwargs)
    # Regular multiprocessing workers don't fully clean up after themselves,
    # so we have to explicitly trigger garbage collection to make sure that all
    # destructors are called...
    gc.collect()


class Pool(multiprocessing.pool.Pool):
    """Pool implementation which uses our version of SimpleQueue.
    This lets us pass tensors in shared memory across processes instead of
    serializing the underlying data."""

    def _setup_queues(self):
        self._inqueue = SimpleQueue()
        self._outqueue = SimpleQueue()
        self._quick_put = self._inqueue._writer.send
        self._quick_get = self._outqueue._reader.recv

    def _repopulate_pool(self):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        for i in range(self._processes - len(self._pool)):
            # changed worker -> clean_worker
            args = (self._inqueue, self._outqueue,
                    self._initializer,
                    self._initargs, self._maxtasksperchild)
            if hasattr(self, '_wrap_exception'):
                args += (self._wrap_exception,)
            w = self.Process(target=clean_worker, args=args)
            self._pool.append(w)
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            util.debug('added worker')
