# mypy: allow-untyped-defs
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

    .. note:: To enable asynchronous execution, applications must pass the
        function object returned by this decorator to RPC APIs. If RPC detected
        attributes installed by this decorator, it knows that this function
        returns a ``Future`` object and will handle that accordingly.
        However, this does not mean this decorator has to be outmost one when
        defining a function. For example, when combined with ``@staticmethod``
        or ``@classmethod``, ``@rpc.functions.async_execution`` needs to be the
        inner decorator to allow the target function be recognized as a static
        or class function. This target function can still execute asynchronously
        because, when accessed, the static or class method preserves attributes
        installed by ``@rpc.functions.async_execution``.


    Example::
        The returned :class:`~torch.futures.Future` object can come from
        :meth:`~torch.distributed.rpc.rpc_async`,
        :meth:`~torch.futures.Future.then`, or :class:`~torch.futures.Future`
        constructor. The example below shows directly using the
        :class:`~torch.futures.Future` returned by
        :meth:`~torch.futures.Future.then`.

        >>> from torch.distributed import rpc
        >>>
        >>> # omitting setup and shutdown RPC
        >>>
        >>> # On all workers
        >>> @rpc.functions.async_execution
        >>> def async_add_chained(to, x, y, z):
        >>>     # This function runs on "worker1" and returns immediately when
        >>>     # the callback is installed through the `then(cb)` API. In the
        >>>     # mean time, the `rpc_async` to "worker2" can run concurrently.
        >>>     # When the return value of that `rpc_async` arrives at
        >>>     # "worker1", "worker1" will run the lambda function accordingly
        >>>     # and set the value for the previously returned `Future`, which
        >>>     # will then trigger RPC to send the result back to "worker0".
        >>>     return rpc.rpc_async(to, torch.add, args=(x, y)).then(
        >>>         lambda fut: fut.wait() + z
        >>>     )
        >>>
        >>> # On worker0
        >>> # xdoctest: +SKIP
        >>> ret = rpc.rpc_sync(
        >>>     "worker1",
        >>>     async_add_chained,
        >>>     args=("worker2", torch.ones(2), 1, 1)
        >>> )
        >>> print(ret)  # prints tensor([3., 3.])

        When combined with TorchScript decorators, this decorator must be the
        outmost one.

        >>> from torch import Tensor
        >>> from torch.futures import Future
        >>> from torch.distributed import rpc
        >>>
        >>> # omitting setup and shutdown RPC
        >>>
        >>> # On all workers
        >>> @torch.jit.script
        >>> def script_add(x: Tensor, y: Tensor) -> Tensor:
        >>>     return x + y
        >>>
        >>> @rpc.functions.async_execution
        >>> @torch.jit.script
        >>> def async_add(to: str, x: Tensor, y: Tensor) -> Future[Tensor]:
        >>>     return rpc.rpc_async(to, script_add, (x, y))
        >>>
        >>> # On worker0
        >>> ret = rpc.rpc_sync(
        >>>     "worker1",
        >>>     async_add,
        >>>     args=("worker2", torch.ones(2), 1)
        >>> )
        >>> print(ret)  # prints tensor([2., 2.])

        When combined with static or class method, this decorator must be the
        inner one.

        >>> from torch.distributed import rpc
        >>>
        >>> # omitting setup and shutdown RPC
        >>>
        >>> # On all workers
        >>> class AsyncExecutionClass:
        >>>
        >>>     @staticmethod
        >>>     @rpc.functions.async_execution
        >>>     def static_async_add(to, x, y, z):
        >>>         return rpc.rpc_async(to, torch.add, args=(x, y)).then(
        >>>             lambda fut: fut.wait() + z
        >>>         )
        >>>
        >>>     @classmethod
        >>>     @rpc.functions.async_execution
        >>>     def class_async_add(cls, to, x, y, z):
        >>>         ret_fut = torch.futures.Future()
        >>>         rpc.rpc_async(to, torch.add, args=(x, y)).then(
        >>>             lambda fut: ret_fut.set_result(fut.wait() + z)
        >>>         )
        >>>         return ret_fut
        >>>
        >>>     @rpc.functions.async_execution
        >>>     def bound_async_add(self, to, x, y, z):
        >>>         return rpc.rpc_async(to, torch.add, args=(x, y)).then(
        >>>             lambda fut: fut.wait() + z
        >>>         )
        >>>
        >>> # On worker0
        >>> ret = rpc.rpc_sync(
        >>>     "worker1",
        >>>     AsyncExecutionClass.static_async_add,
        >>>     args=("worker2", torch.ones(2), 1, 2)
        >>> )
        >>> print(ret)  # prints tensor([4., 4.])
        >>>
        >>> ret = rpc.rpc_sync(
        >>>     "worker1",
        >>>     AsyncExecutionClass.class_async_add,
        >>>     args=("worker2", torch.ones(2), 1, 2)
        >>> )
        >>> print(ret)  # prints tensor([4., 4.])

        This decorator also works with RRef helpers, i.e., .
        :meth:`torch.distributed.rpc.RRef.rpc_sync`,
        :meth:`torch.distributed.rpc.RRef.rpc_async`, and
        :meth:`torch.distributed.rpc.RRef.remote`.

        >>> from torch.distributed import rpc
        >>>
        >>> # reuse the AsyncExecutionClass class above
        >>> rref = rpc.remote("worker1", AsyncExecutionClass)
        >>> ret = rref.rpc_sync().static_async_add("worker2", torch.ones(2), 1, 2)
        >>> print(ret)  # prints tensor([4., 4.])
        >>>
        >>> rref = rpc.remote("worker1", AsyncExecutionClass)
        >>> ret = rref.rpc_async().static_async_add("worker2", torch.ones(2), 1, 2).wait()
        >>> print(ret)  # prints tensor([4., 4.])
        >>>
        >>> rref = rpc.remote("worker1", AsyncExecutionClass)
        >>> ret = rref.remote().static_async_add("worker2", torch.ones(2), 1, 2).to_here()
        >>> print(ret)  # prints tensor([4., 4.])
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    # Can't declare and use attributes of function objects (mypy#2087)
    wrapper._wrapped_async_rpc_function = fn  # type: ignore[attr-defined]
    return wrapper
