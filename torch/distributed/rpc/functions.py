import functools


def async_execution(fn):
    r"""
    A decorator for a function indicating that the return value of the function
    is guaranteed to be a :class:`~torch.futures.Future` object and this
    function can run asynchronously on the RPC callee. More specifically, the
    callee extracts the :class:`~torch.futures.Future` returned by the wrapped
    function and installs subsequent processing steps as a callback to that
    :class:`~torch.futures.Future`. The installed callback will read the value
    from the :class:`~torch.futures.Future` when completed and send the
    value back as the RPC response. That also means the returned
    :class:`~torch.futures.Future` only exists on the callee side and is never
    sent through RPC. This decorator is useful when the wrapped function's
    (``fn``) execution needs to pause and resume due to, e.g., containing
    :meth:`~torch.distributed.rpc.rpc_async` or waiting for other signals.

    .. note:: This decorator must be the outmost one when combined with other
        decorators. Otherwise, RPC will not be able to detect the attributes
        installed by this decorator.

    Example::
        The returned :class:`~torch.futures.Future` object can come from
        ``rpc.rpc_async``, ``Future.then(cb)``, or :class:`~torch.futures.Future`
        constructor. The example below shows directly using the
        :class:`~torch.futures.Future` returned by ``Future.then(cb)``.

        >>> from torch.distributed import rpc
        >>>
        >>> # omitting setup and shutdown RPC
        >>>
        >>> # On worker0
        >>> @rpc.functions.async_execution
        >>> def async_add_chained(to, x, y, z):
        >>>     # This function runs on "worker1" and returns immediately when
        >>>     # the callback is installed through the `then(cb)` API. In the
        >>>     # mean time, the `rpc_async` to "worker2" can run concurrently.
        >>>     # When the return value of that `rpc_async` arrives at
        >>>     # "worker1", "worker1" will run the lambda function accordinly
        >>>     # and set the value for the previously returned `Future`, which
        >>>     # will then trigger RPC to send the result back to "worker0".
        >>>     return rpc.rpc_async(to, torch.add, args=(x, y)).then(
        >>>         lambda fut: fut.wait() + z
        >>>     )
        >>>
        >>> ret = rpc.rpc_sync(
        >>>     "worker1",
        >>>     async_add_chained,
        >>>     args=("worker2", torch.ones(2), 1, 1)
        >>> )
        >>> print(ret)  # prints tensor([3., 3.])

        When combined with TorchScript decorators (or any other decorators),
        this decorator must be the outmost one.

        >>> from torch.distributed import rpc
        >>>
        >>> # omitting setup and shutdown RPC
        >>>
        >>> # On worker0
        >>> @torch.jit.script
        >>> def script_add(x, y):
        >>>     # type: (Tensor, Tensor) -> Tensor
        >>>     return x + y
        >>>
        >>> @rpc.functions.async_execution
        >>> @torch.jit.script
        >>> def async_add(to, x, y):
        >>>     # type: (str, Tensor, Tensor) -> Future[Tensor]
        >>>     return rpc.rpc_async(to, script_add, (x, y))
        >>>
        >>> ret = rpc.rpc_sync(
        >>>     "worker1",
        >>>     async_add,
        >>>     args=("worker2", torch.ones(2), 1)
        >>> )
        >>> print(ret)  # prints tensor([2., 2.])
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._wrapped_async_rpc_function = fn
    return wrapper
