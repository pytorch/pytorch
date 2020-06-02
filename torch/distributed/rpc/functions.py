import functools

def async_execution(fn):
    r"""
    A decorator for a function indicating that the return value of the function
    is guaranteed to be a ``torch.futures.Future`` object and this function can
    run asynchronously on the RPC callee. This is useful when this function's
    execution needs to pause and resume due to, e.g., containing
    :meth:`~torch.distributed.rpc.rpc_async` or waiting for other signals.

    Example::
        The returned ``torch.futures.Future`` object can come from
        ``rpc.rpc_async``, ``Future.then(cb)``, or ``torch.futures.Future``
        constructor. The example below shows directly using the ``Future``
        returned by ``Future.then(cb)``.

        >>> from torch.distributed import rpc
        >>>
        >>> # omit setting up and shutdown RPC
        >>>
        >>> # On worker0
        >>> @rpc.functions.async_execution
        >>> def async_add_chained(to, x, y, z):
        >>>     # This function runs on "worker1" and returns immediately when the
        >>>     # the callback is installed through the `then(cb)` API. In the mean
        >>>     # time, the `rpc_async` to "worker2" can run concurrently. When the
        >>>     # return value of that `rpc_async` arrives at "worker1", "worker1"
        >>>     # will run the lambda function accordinly and set the value for the
        >>>     # previously returned `Future`, which will then trigger RPC to send
        >>>     # the result back to "worker0".
        >>>     return rpc.rpc_async(to, torch.add, args=(x, y)).then(
        >>>         lambda fut: fut.wait() + z
        >>>     )
        >>>
        >>> ret = pc.rpc_sync(
        >>>     "worker1",
        >>>     async_add_chained,
        >>>     args=("worker2", torch.ones(2), 1, 1)
        >>> )
        >>> print(ret)  # prints tensor([3., 3.])
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._wrapped_async_rpc_function = fn
    return wrapper
