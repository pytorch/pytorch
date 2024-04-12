from __future__ import annotations

from typing import cast, Callable, Generic, List, Optional, Type, TypeVar, Union

import torch

__all__ = ['Future', 'collect_all', 'wait_all']

T = TypeVar("T")
S = TypeVar("S")


class _PyFutureMeta(type(torch._C.Future), type(Generic)):  # type: ignore[misc, no-redef]
    pass


class Future(torch._C.Future, Generic[T], metaclass=_PyFutureMeta):
    r"""
    Wrapper around a ``torch._C.Future`` which encapsulates an asynchronous
    execution of a callable, e.g. :meth:`~torch.distributed.rpc.rpc_async`. It
    also exposes a set of APIs to add callback functions and set results.

    .. warning:: GPU support is a beta feature, subject to changes.
    """

    def __init__(self, *, devices: Optional[List[Union[int, str, torch.device]]] = None):
        r"""
        Create an empty unset ``Future``. If the future is intended to hold
        values containing CUDA tensors, (a superset of) their CUDA devices must
        be specified at construction. (This is only supported if
        ``torch.cuda.is_available()`` returns ``True``). This is needed to
        ensure proper CUDA stream synchronization. The child futures, returned
        by the ``then`` method, will inherit these devices.

        Args:
            devices(``List[Union[int, str, torch.device]]``, optional): the set
                of devices on which tensors contained in this future's value are
                allowed to reside and on which callbacks are allowed to operate.
        """
        if devices is None:
            devices = []
        super().__init__([torch.device(d) for d in devices])

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

    def wait(self) -> T:
        r"""
        Block until the value of this ``Future`` is ready.

        If the value contains tensors that reside on GPUs, then an additional
        synchronization is performed with the kernels (executing on the device)
        which may be asynchronously populating those tensors. Such sync is
        non-blocking, which means that ``wait()`` will insert the necessary
        instructions in the current streams to ensure that further operations
        enqueued on those streams will be properly scheduled after the async
        kernels but, once that is done, ``wait()`` will return, even if those
        kernels are still running. No further synchronization is required when
        accessing and using the values, as long as one doesn't change streams.

        Returns:
            The value held by this ``Future``. If the function (callback or RPC)
            creating the value has thrown an error, this ``wait`` method will
            also throw an error.
        """
        return super().wait()

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

    def then(self, callback: Callable[[Future[T]], S]) -> Future[S]:
        r"""
        Append the given callback function to this ``Future``, which will be run
        when the ``Future`` is completed.  Multiple callbacks can be added to
        the same ``Future``, but the order in which they will be executed cannot
        be guaranteed (to enforce a certain order consider chaining:
        ``fut.then(cb1).then(cb2)``). The callback must take one argument, which
        is the reference to this ``Future``. The callback function can use the
        :meth:`value` method to get the value. Note that if this ``Future`` is
        already completed, the given callback will be run immediately inline.

        If the ``Future``'s value contains tensors that reside on GPUs, the
        callback might be invoked while the async kernels that are populating
        those tensors haven't yet finished executing on the device. However, the
        callback will be invoked with some dedicated streams set as current
        (fetched from a global pool) which will be synchronized with those
        kernels. Hence any operation performed by the callback on these tensors
        will be scheduled on the device after the kernels complete. In other
        words, as long as the callback doesn't switch streams, it can safely
        manipulate the result without any additional synchronization. This is
        similar to the non-blocking behavior of :meth:`wait`.

        Similarly, if the callback returns a value that contains tensors that
        reside on a GPU, it can do so even if the kernels that are producing
        these tensors are still running on the device, as long as the callback
        didn't change streams during its execution. If one wants to change
        streams, one must be careful to re-synchronize them with the original
        streams, that is, those that were current when the callback was invoked.

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
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
            >>> def callback(fut):
            ...     print(f"RPC return value is {fut.wait()}.")
            >>> fut = torch.futures.Future()
            >>> # The inserted callback will print the return value when
            >>> # receiving the response from "worker1"
            >>> cb_fut = fut.then(callback)
            >>> chain_cb_fut = cb_fut.then(
            ...     lambda x : print(f"Chained cb done. {x.wait()}")
            ... )
            >>> fut.set_result(5)
            RPC return value is 5.
            Chained cb done. None
        """
        return cast(Future[S], super().then(callback))

    def add_done_callback(self, callback: Callable[[Future[T]], None]) -> None:
        r"""
        Append the given callback function to this ``Future``, which will be run
        when the ``Future`` is completed.  Multiple callbacks can be added to
        the same ``Future``, but the order in which they will be executed cannot
        be guaranteed. The callback must take one argument, which is the
        reference to this ``Future``. The callback function can use the
        :meth:`value` method to get the value. Note that if this ``Future`` is
        already completed, the given callback will be run inline.

        We recommend that you use the :meth:`then` method as it provides a way
        to synchronize after your callback has completed. ``add_done_callback``
        can be cheaper if your callback does not return anything. But both
        :meth:`then` and ``add_done_callback`` use the same callback
        registration API under the hood.

        With respect to GPU tensors, this method behaves in the same way as
        :meth:`then`.

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
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
            >>> def callback(fut):
            ...     print("This will run after the future has finished.")
            ...     print(fut.wait())
            >>> fut = torch.futures.Future()
            >>> fut.add_done_callback(callback)
            >>> fut.set_result(5)
            This will run after the future has finished.
            5
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
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
            >>> import threading
            >>> import time
            >>> def slow_set_future(fut, value):
            ...     time.sleep(0.5)
            ...     fut.set_result(value)
            >>> fut = torch.futures.Future()
            >>> t = threading.Thread(
            ...     target=slow_set_future,
            ...     args=(fut, torch.ones(2) * 3)
            ... )
            >>> t.start()
            >>> print(fut.wait())
            tensor([3., 3.])
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
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
            >>> fut = torch.futures.Future()
            >>> fut.set_exception(ValueError("foo"))
            >>> fut.wait()
            Traceback (most recent call last):
            ...
            ValueError: foo
        """
        assert isinstance(result, Exception), f"{result} is of type {type(result)}, not an Exception."

        def raise_error(fut_result):
            raise fut_result

        super()._set_unwrap_func(raise_error)
        self.set_result(result)  # type: ignore[arg-type]


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
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
        >>> fut0 = torch.futures.Future()
        >>> fut1 = torch.futures.Future()
        >>> fut = torch.futures.collect_all([fut0, fut1])
        >>> fut0.set_result(0)
        >>> fut1.set_result(1)
        >>> fut_list = fut.wait()
        >>> print(f"fut0 result = {fut_list[0].wait()}")
        fut0 result = 0
        >>> print(f"fut1 result = {fut_list[1].wait()}")
        fut1 result = 1
    """
    return cast(Future[List[Future]], torch._C._collect_all(cast(List[torch._C.Future], futures)))


def wait_all(futures: List[Future]) -> List:
    r"""
    Waits for all provided futures to be complete, and returns
    the list of completed values. If any of the futures encounters an error,
    the method will exit early and report the error not waiting for other
    futures to complete.

    Args:
        futures (list): a list of :class:`~torch.futures.Future` object.

    Returns:
        A list of the completed :class:`~torch.futures.Future` results. This
        method will throw an error if ``wait`` on any
        :class:`~torch.futures.Future` throws.
    """
    return [fut.wait() for fut in torch._C._collect_all(cast(List[torch._C.Future], futures)).wait()]
