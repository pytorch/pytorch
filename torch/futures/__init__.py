import torch


class Future(torch._C.Future):
    r"""
    Wrapper around a ``torch._C.Future`` which encapsulates an asynchronous
    execution of a callable, e.g. :meth:`~torch.distributed.rpc.rpc_async`. It
    also exposes a set of APIs to add callback functions and set results.
    """
    def __new__(cls):
        return super(Future, cls).__new__(cls)

    def wait(self):
        r"""
        Block until the value of this ``Future`` is ready.

        Returns:
            The value held by this ``Future``. If the function (callback or RPC)
            creating the value has thrown an error, this ``wait`` method will
            also throw an error.
        """
        return super(Future, self).wait()

    def then(self, callback):
        r"""
        Append the given callback function to this ``Future``, which will be run
        when the ``Future`` is completed.  Multiple callbacks can be added to
        the same ``Future``, and will be invoked in the same order as they were
        added. The callback must take one argument, which is the reference to
        this ``Future``. The callback function can use the ``Future.wait()`` API
        to get the value.

        Arguments:
            callback(``Callable``): a ``Callable`` that takes this ``Future`` as
                                    the only argument.

        Returns:
            A new ``Future`` object that holds the return value of the
            ``callback`` and will be marked as completed when the given
            ``callback`` finishes.

        Example::
            >>> import torch
            >>>
            >>> def callback(fut):
            >>>     print(f"RPC return value is {fut.wait()}.")
            >>>
            >>> fut = torch.futures.Future()
            >>> # The inserted callback will print the return value when
            >>> # receiving the response from "worker1"
            >>> cb_fut = fut.then(callback)
            >>> chain_cb_fut = cb_fut.then(
            >>>     lambda x : print(f"Chained cb done. {x.wait()}")
            >>> )
            >>> fut.set_result(5)
            >>>
            >>> # Outputs are:
            >>> # RPC return value is 5.
            >>> # Chained cb done. None
        """
        return super(Future, self).then(callback)

    def set_result(self, result):
        r"""
        Set the result for this ``Future``, which will mark this ``Future`` as
        completed and trigger all attached callbacks. Note that a ``Future``
        cannot be marked completed twice.

        Arguments:
            result (object): the result object of this ``Future``.

        Example::
            >>> import threading
            >>> import time
            >>> import torch
            >>>
            >>> def slow_set_future(fut, value):
            >>>     time.sleep(0.5)
            >>>     fut.set_result(value)
            >>>
            >>> fut = torch.futures.Future()
            >>> t = threading.Thread(
            >>>     target=slow_set_future,
            >>>     args=(fut, torch.ones(2) * 3)
            >>> )
            >>> t.start()
            >>>
            >>> print(fut.wait())  # tensor([3., 3.])
            >>> t.join()
        """
        super(Future, self).set_result(result)


def collect_all(futures):
    r"""
    Collects the provided :class:`~torch.futures.Future` objects into a single
    combined :class:`~torch.futures.Future` that is completed when all of the
    sub-futures are completed.

    Arguments:
        futures (list): a list of :class:`~torch.futures.Future` objects.

    Returns:
        Returns a :class:`~torch.futures.Future` object to a list of the passed
        in Futures.

    Example::
        >>> import torch
        >>>
        >>> fut0 = torch.futures.Future()
        >>> fut1 = torch.futures.Future()
        >>>
        >>> fut = torch.futures.collect_all([fut0, fut1])
        >>>
        >>> fut0.set_result(0)
        >>> fut1.set_result(1)
        >>>
        >>> fut_list = fut.wait()
        >>> print(f"fut0 result = {fut_list[0].wait()}")
        >>> print(f"fut1 result = {fut_list[1].wait()}")
        >>> # outputs:
        >>> # fut0 result = 0
        >>> # fut1 result = 1
    """
    return torch._C._collect_all(futures)

def wait_all(futures):
    r"""
    Waits for all provided futures to be complete, and returns
    the list of completed values.

    Arguments:
        futures (list): a list of :class:`~torch.futures.Future` object.

    Returns:
        A list of the completed :class:`~torch.futures.Future` results. This
        method will throw an error if ``wait`` on any
        :class:`~torch.futures.Future` throws.
    """
    return [fut.wait() for fut in torch._C._collect_all(futures).wait()]
