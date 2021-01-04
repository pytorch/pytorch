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

    .. warning:: GPU support is a beta feature, subject to changes.
    """

    def done(self) -> bool:
        r"""
        Return ``True`` if this ``Future`` is done. A ``Future`` is done if it
        has a result or an exception.

        If the value contains tensors that reside on GPUs, ``Future.done()``
        will return ``True`` even if the asynchronous kernels that are
        populating those tensors haven't yet completed running on the device,
        because at such stage the result is already usable, provided one
        performs the appropriate synchronizations (see :meth:`wait`).
        """
        return super().done()

    def wait(self, non_blocking: bool = False) -> T:
        r"""
        Block until the value of this ``Future`` is ready.

        If the value contains tensors that reside on GPUs, then an additional
        synchronization is performed with the kernels (executing on the device)
        which may be asynchronously populating those tensors. Such sync can be:

        - *Blocking* (the default), in which case the calling thread is held up
          until those asynchronous kernels effectively terminate running on the
          device. In such a mode, when ``wait()`` returns it is safe to
          immediately use the result on any stream without further
          synchronization. On the other hand, a blocking wait delays the moment
          at which new operations can be sent to the GPU, possibly degrading
          performance.
        - *Non-blocking*, in which case ``wait()`` inserts the necessary
          instructions in the current streams to ensure that further operations
          enqueued on those streams will be properly scheduled after the async
          kernels. Once that is done, ``wait()`` will return, even if
          those kernels are still running. This mode allows for better
          pipelining and performance, and should be safe as long as one doesn't
          change streams between the call to ``wait()`` and the time at
          which the result is used (but it requires explicit manual
          synchronization when switching streams).

        Args:
            non_blocking (bool): whether the sync with the GPU kernels should be
                                 non-blocking.

        Returns:
            The value held by this ``Future``. If the function (callback or RPC)
            creating the value has thrown an error, this ``wait`` method will
            also throw an error.
        """
        return super().wait(non_blocking)

    def value(self) -> T:
        r"""
        Obtain the value of an already-completed future.

        This method should only be called after a call to :meth:`wait` has
        completed, or inside a callback function passed to :meth:`then`. In
        other cases this ``Future`` may not yet hold a value and calling
        ``value()`` could fail.

        If the value contains tensors that reside on GPUs, then this method will
        *not* perform any additional synchronization. This should be done
        beforehand, separately, through a call to :meth:`wait` (except within
        callbacks, for which it's already being taken care of by :meth:`then`).

        Returns:
            The value held by this ``Future``. If the function (callback or RPC)
            creating the value has thrown an error, this ``value()`` method will
            also throw an error.
        """
        return super().value()

    # Have to use string annotations because  PEP-0563 is not available in 3.6
    def then(self, callback):  # type: (Callable[[Future[T]], S]) -> Future[S]
        r"""
        Append the given callback function to this ``Future``, which will be run
        when the ``Future`` is completed.  Multiple callbacks can be added to
        the same ``Future``, but the order in which they will be executed cannot
        be guaranteed (to enforce a certain order consider chaining:
        ``fut.then(cb1).then(cb2)``). The callback must take one argument, which
        is the reference to this ``Future``. The callback function can use the
        :meth:`value` method to get the value.

        If the ``Future``'s value contains tensors that reside on GPUs, the
        callback might be invoked while the async kernels that are populating
        those tensors haven't yet finished executing on the device. However, the
        callback will be invoked with some dedicated streams set as current
        (fetched from a global pool) which will be synchronized with those
        kernels so that any new operation enqueued on those streams will be
        scheduled on the device after the kernels complete. In other words, as
        long as the callback doesn't switch streams, it can safely manipulate
        the result without any additional synchronization. This is similar to
        the non-blocking mode of :meth:`wait`.

        Args:
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
        return cast(Future[S], super().then(callback))

    # Have to use string annotations because  PEP-0563 is not available in 3.6
    def add_done_callback(self, callback):  # type: (Callable[[Future[T]], None]) -> None
        r"""
        Append the given callback function to this ``Future``, which will be run
        when the ``Future`` is completed.  Multiple callbacks can be added to
        the same ``Future``, but the order in which they will be executed cannot
        be guaranteed. The callback must take one argument, which is the
        reference to this ``Future``. The callback function can use the
        :meth:`value` method to get the value.

        We recommend that you use the :meth:`then` method as it provides a way to synchronize
        after your callback has completed. ``add_done_callback`` can be cheaper if your
        callback does not return anything. But both :meth:`then` and ``add_done_callback``
        use the same callback registration API under the hood.

        Args:
            callback(``None``): a ``Callable`` that takes in no arguments

        Example::
            >>> import torch
            >>>
            >>> def callback():
            >>>     print(f"This will run after the future has finished.")
            >>>
            >>> fut = torch.futures.Future()
            >>> fut.add_done_callback(callback)
            >>> fut.set_result(5)
            >>>
            >>> # Outputs are:
            >>> # This will run after the future has finished.
        """
        super().add_done_callback(callback)

    def set_result(self, result: T) -> None:
        r"""
        Set the result for this ``Future``, which will mark this ``Future`` as
        completed and trigger all attached callbacks. Note that a ``Future``
        cannot be marked completed twice.

        If the result contains tensors that reside on GPUs, this method can be
        called even if the asynchronous kernels that are populating those
        tensors haven't yet completed running on the device, provided that the
        streams on which those kernels were enqueued are set as the current ones
        when this method is called. Put simply, it's safe to call this method
        immediately after launching those kernels, without any additional
        synchronization, as long as one doesn't change streams in between. This
        method will record events on all the relevant current streams and will
        use them to ensure proper scheduling for all the consumers of this
        ``Future``.

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


def wait_all(futures: List[Future], non_blocking: bool = False) -> List:
    r"""
    Waits for all provided futures to be complete, and returns
    the list of completed values.

    Args:
        futures (list): a list of :class:`~torch.futures.Future` object.
        non_blocking (bool): whether the sync with the GPU kernels should be
                             non-blocking.

    Returns:
        A list of the completed :class:`~torch.futures.Future` results. This
        method will throw an error if ``wait`` on any
        :class:`~torch.futures.Future` throws.
    """
    return [
        fut.wait(non_blocking)
        for fut in torch._C._collect_all(cast(List[torch._C.Future], futures)).wait()
    ]
