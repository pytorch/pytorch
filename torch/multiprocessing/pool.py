import multiprocessing
import multiprocessing.pool
import multiprocessing.util as util

from . import Queue


def clean_worker(*args, **kwargs):
    import gc
    multiprocessing.pool.worker(*args, **kwargs)
    # Regular multiprocessing workers don't fully clean up after themselves,
    # so we have to explicitly trigger garbage collection to make sure that all
    # destructors are called...
    gc.collect()


class Pool(multiprocessing.pool.Pool):
    """Pool implementation with customizable pickling reducers.
    This is useful to control how data is shipped between processes
    and makes it possible to use shared memory without useless
    copies induces by the default pickling methods of the original
    objects passed as arguments to dispatch.
    `forward_reducers` and `backward_reducers` are expected to be
    dictionaries with key/values being `(type, callable)` pairs where
    `callable` is a function that, given an instance of `type`, will return a
    tuple `(constructor, tuple_of_objects)` to rebuild an instance out of the
    pickled `tuple_of_objects` as would return a `__reduce__` method.
    See the standard library documentation about pickling for more details.
    """

    def __init__(self, processes=None, forward_reducers=None,
                 backward_reducers=None, **kwargs):
        if forward_reducers is None:
            forward_reducers = dict()
        if backward_reducers is None:
            backward_reducers = dict()

        self._forward_reducers = forward_reducers
        self._backward_reducers = backward_reducers

        poolargs = dict(processes=processes)
        poolargs.update(kwargs)
        super(Pool, self).__init__(**poolargs)

    def _setup_queues(self):
        context = getattr(self, '_ctx', multiprocessing)
        self._inqueue = Queue(context, self._forward_reducers)
        self._outqueue = Queue(context, self._backward_reducers)
        self._quick_put = self._inqueue._send
        self._quick_get = self._outqueue._recv

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
