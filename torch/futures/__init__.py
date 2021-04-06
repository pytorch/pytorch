from typing import cast, Callable, Generic, List, Type, TypeVar

import torch
from torch._six import PY37

T = TypeVar("T")
S = TypeVar("S")

if not PY37:
    # Workaround for https://github.com/python/typing/issues/449 in Python 3.6
    from typing import GenericMeta

    class _PyFutureMeta(type(torch._C.Future), GenericMeta):   # type: ignore[misc]
        pass
else:
    class _PyFutureMeta(type(torch._C.Future), type(Generic)):  # type: ignore[misc, no-redef]
        pass

class Future(torch._C.Future, Generic[T], metaclass=_PyFutureMeta):
    r"""
    Wrapper around a ``torch._C.Future`` which encapsulates an asynchronous
    execution of a callable, e.g. :meth:`~torch.distributed.rpc.rpc_async`. It
    also exposes a set of APIs to add callback functions and set results.
    """

    def done(self) -> bool:
        r"""
        Return ``True`` if this ``Future`` is done. A ``Future`` is done if it
        has a result or an exception.
        """
        return super().done()

    def wait(self) -> T:
        r"""
        Block until the value of this ``Future`` is ready.

        Returns:
            The value held by this ``Future``. If the function (callback or RPC)
            creating the value has thrown an error, this ``wait`` method will
            also throw an error.
        """
        return super().wait()

    # Have to use string annotations because  PEP-0563 is not available in 3.6
    def then(self, callback):  # type: (Callable[[Future[T]], S]) -> Future[S]
        r"""
        Append the given callback function to this ``Future``, which will be run
        when the ``Future`` is completed.  Multiple callbacks can be added to
        the same ``Future``, and will be invoked in the same order as they were
        added. The callback must take one argument, which is the reference to
        this ``Future``. The callback function can use the ``Future.wait()`` API
        to get the value. Note that if this ``Future`` is already completed, the
        given callback will be run immediately inline.

        Args:
            callback(``Callable``): a ``Callable`` that takes this ``Future`` as
                                    the only argument.

        Returns:
            A new ``Future`` object that holds the return value of the
            ``callback`` and will be marked as completed when the given
            ``callback`` finishes.

        .. note:: Note that if the callback function throws, either
            through the original future being completed with an exception and
            calling ``fut.wait()``, or through other code in the callback, the
            future returned by ``then`` will be marked appropriately with the
            encountered error. However, if this callback later completes
            additional futures, those futures are not marked as completed with
            an error and the user is responsible for handling completion/waiting
            on those futures independently.

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
        return cast(Future[S], super().then(callback))

    # Have to use string annotations because  PEP-0563 is not available in 3.6
    def _add_done_callback(self, callback):  # type: (Callable[[Future[T]], None]) -> None
        r"""
        Append the given callback function to this ``Future``, which will be run
        when the ``Future`` is completed.  Multiple callbacks can be added to
        the same ``Future``, and will be invoked in the same order as they were
        added. The callback must take one argument, which is the reference to
        this ``Future``. The callback function can use the ``Future.wait()`` API
        to get the value. Note that if this ``Future`` is already completed, the
        given callback will be run inline.

        We recommend that you use the ``then`` API as it provides a way to synchronize
        after your callback has completed. ``add_done_callback`` can be cheaper if your
        callback does not return anything. But both ``then`` and ``add_done_callback``
        use the same callback registration API under the hood, and thus the order of
        their callbacks will be maintained even if their calls are interleaved.

        Args:
            callback(``Future``): a ``Callable`` that takes in one argument,
            which is the reference to this ``Future``.

        .. note:: Note that if the callback function throws, either
            through the original future being completed with an exception and
            calling ``fut.wait()``, or through other code in the callback,
            error handling must be carefully taken care of. For example, if
            this callback later completes additional futures, those futures are
            not marked as completed with an error and the user is responsible
            for handling completion/waiting on those futures independently.

        Example::
            >>> import torch
            >>>
            >>> def callback(fut):
            >>>     print(f"This will run after the future has finished.")
            >>>     print(fut.wait())
            >>>
            >>> fut = torch.futures.Future()
            >>> fut.add_done_callback(callback)
            >>> fut.set_result(5)
            >>>
            >>> # Outputs are:
            >>> This will run after the future has finished.
            >>> 5
        """
        super().add_done_callback(callback)

    def set_result(self, result: T) -> None:
        r"""
        Set the result for this ``Future``, which will mark this ``Future`` as
        completed and trigger all attached callbacks. Note that a ``Future``
        cannot be marked completed twice.

        Args:
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
        super().set_result(result)

    def set_exception(self, result: T) -> None:
        r"""
        Set an exception for this ``Future``, which will mark this ``Future`` as
        completed with an error and trigger all attached callbacks. Note that
        when calling wait()/value() on this ``Future``, the exception set here
        will be raised inline.

        Args:
            result (BaseException): the exception for this ``Future``.

        Example::
            >>> import torch
            >>>
            >>> fut = torch.futures.Future()
            >>> fut.set_exception(ValueError("foo"))
            >>> fut.wait()
            >>>
            >>> # Output:
            >>> # This will run after the future has finished.
            >>> ValueError: foo
        """
        assert isinstance(result, Exception), f"{result} is of type {type(result)}, not an Exception."

        def raise_error(fut_result):
            raise fut_result

        super()._set_unwrap_func(raise_error)
        self.set_result(result)  # type: ignore


def collect_all(futures: List[Future]) -> Future[List[Future]]:
    r"""
    Collects the provided :class:`~torch.futures.Future` objects into a single
    combined :class:`~torch.futures.Future` that is completed when all of the
    sub-futures are completed.

    Args:
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
    return cast(Future[List[Future]], torch._C._collect_all(cast(List[torch._C.Future], futures)))


def wait_all(futures: List[Future]) -> List:
    r"""
    Waits for all provided futures to be complete, and returns
    the list of completed values.

    Args:
        futures (list): a list of :class:`~torch.futures.Future` object.

    Returns:
        A list of the completed :class:`~torch.futures.Future` results. This
        method will throw an error if ``wait`` on any
        :class:`~torch.futures.Future` throws.
    """
    return [fut.wait() for fut in torch._C._collect_all(cast(List[torch._C.Future], futures)).wait()]
