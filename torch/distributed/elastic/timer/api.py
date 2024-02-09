# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Optional, Set

__all__ = ['TimerRequest', 'TimerClient', 'RequestQueue', 'TimerServer', 'configure', 'expires']

log = logging.getLogger(__name__)

class TimerRequest:
    """
    Data object representing a countdown timer acquisition and release
    that is used between the ``TimerClient`` and ``TimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.

    .. note:: the type of ``worker_id`` is implementation specific.
              It is whatever the TimerServer and TimerClient implementations
              have on to uniquely identify a worker.
    """

    __slots__ = ["worker_id", "scope_id", "expiration_time"]

    def __init__(self, worker_id: Any, scope_id: str, expiration_time: float):
        self.worker_id = worker_id
        self.scope_id = scope_id
        self.expiration_time = expiration_time

    def __eq__(self, other):
        if isinstance(other, TimerRequest):
            return (
                self.worker_id == other.worker_id
                and self.scope_id == other.scope_id
                and self.expiration_time == other.expiration_time
            )
        return False


class TimerClient(abc.ABC):
    """
    Client library to acquire and release countdown timers by communicating
    with the TimerServer.
    """

    @abc.abstractmethod
    def acquire(self, scope_id: str, expiration_time: float) -> None:
        """
        Acquires a timer for the worker that holds this client object
        given the scope_id and expiration_time. Typically registers
        the timer with the TimerServer.
        """
        pass

    @abc.abstractmethod
    def release(self, scope_id: str):
        """
        Releases the timer for the ``scope_id`` on the worker this
        client represents. After this method is
        called, the countdown timer on the scope is no longer in effect.
        """
        pass


class RequestQueue(abc.ABC):
    """
    Consumer queue holding timer acquisition/release requests
    """

    @abc.abstractmethod
    def size(self) -> int:
        """
        Returns the size of the queue at the time this method is called.
        Note that by the time ``get`` is called the size of the queue
        may have increased. The size of the queue should not decrease
        until the ``get`` method is called. That is, the following assertion
        should hold:

        size = q.size()
        res = q.get(size, timeout=0)
        assert size == len(res)

        -- or --

        size = q.size()
        res = q.get(size * 2, timeout=1)
        assert size <= len(res) <= size * 2
        """
        pass

    @abc.abstractmethod
    def get(self, size: int, timeout: float) -> List[TimerRequest]:
        """
        Gets up to ``size`` number of timer requests in a blocking fashion
        (no more than ``timeout`` seconds).
        """
        pass


class TimerServer(abc.ABC):
    """
    Entity that monitors active timers and expires them
    in a timely fashion. This server is responsible for
    reaping workers that have expired timers.
    """

    def __init__(
        self, request_queue: RequestQueue, max_interval: float, daemon: bool = True
    ):
        """
        :param request_queue: Consumer ``RequestQueue``
        :param max_interval: max time (in seconds) to wait
                             for an item in the request_queue
        :param daemon: whether to run the watchdog thread as a daemon
        """
        super().__init__()
        self._request_queue = request_queue
        self._max_interval = max_interval
        self._daemon = daemon
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_signaled = False

    @abc.abstractmethod
    def register_timers(self, timer_requests: List[TimerRequest]) -> None:
        """
        Processes the incoming timer requests and registers them with the server.
        The timer request can either be a acquire-timer or release-timer request.
        Timer requests with a negative expiration_time should be interpreted
        as a release-timer request.
        """
        pass

    @abc.abstractmethod
    def clear_timers(self, worker_ids: Set[Any]) -> None:
        """
        Clears all timers for the given ``worker_ids``.
        """
        pass

    @abc.abstractmethod
    def get_expired_timers(self, deadline: float) -> Dict[str, List[TimerRequest]]:
        """
        Returns all expired timers for each worker_id. An expired timer
        is a timer for which the expiration_time is less than or equal to
        the provided deadline.
        """
        pass

    @abc.abstractmethod
    def _reap_worker(self, worker_id: Any) -> bool:
        """
        Reaps the given worker. Returns True if the worker has been
        successfully reaped, False otherwise. If any uncaught exception
        is thrown from this method, the worker is considered reaped
        and all associated timers will be removed.
        """

    def _reap_worker_no_throw(self, worker_id: Any) -> bool:
        """
        Wraps ``_reap_worker(worker_id)``, if an uncaught exception is
        thrown, then it considers the worker as reaped.
        """
        try:
            return self._reap_worker(worker_id)
        except Exception:
            log.exception(
                "Uncaught exception thrown from _reap_worker(), "
                "check that the implementation correctly catches exceptions",
            )
            return True

    def _watchdog_loop(self):
        while not self._stop_signaled:
            try:
                self._run_watchdog()
            except Exception:
                log.exception("Error running watchdog")

    def _run_watchdog(self):
        batch_size = max(1, self._request_queue.size())
        timer_requests = self._request_queue.get(batch_size, self._max_interval)
        self.register_timers(timer_requests)
        now = time.time()
        reaped_worker_ids = set()
        for worker_id, expired_timers in self.get_expired_timers(now).items():
            log.info(
                "Reaping worker_id=[%s]."
                " Expired timers: %s",
                worker_id, self._get_scopes(expired_timers)
            )
            if self._reap_worker_no_throw(worker_id):
                log.info("Successfully reaped worker=[%s]", worker_id)
                reaped_worker_ids.add(worker_id)
            else:
                log.error(
                    "Error reaping worker=[%s]. Will retry on next watchdog.", worker_id
                )
        self.clear_timers(reaped_worker_ids)

    def _get_scopes(self, timer_requests):
        return [r.scope_id for r in timer_requests]

    def start(self) -> None:
        log.info(
            "Starting %s..."
            " max_interval=%s,"
            " daemon=%s",
            type(self).__name__, self._max_interval, self._daemon
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=self._daemon
        )
        log.info("Starting watchdog thread...")
        self._watchdog_thread.start()

    def stop(self) -> None:
        log.info("Stopping %s", type(self).__name__)
        self._stop_signaled = True
        if self._watchdog_thread:
            log.info("Stopping watchdog thread...")
            self._watchdog_thread.join(self._max_interval)
            self._watchdog_thread = None
        else:
            log.info("No watchdog thread running, doing nothing")


_timer_client: Optional[TimerClient] = None


def configure(timer_client: TimerClient):
    """
    Configures a timer client. Must be called before using ``expires``.
    """
    global _timer_client
    _timer_client = timer_client
    log.info("Timer client configured to: %s", type(_timer_client).__name__)


@contextmanager
def expires(
    after: float, scope: Optional[str] = None, client: Optional[TimerClient] = None
):
    """
    Acquires a countdown timer that expires in ``after`` seconds from now,
    unless the code-block that it wraps is finished within the timeframe.
    When the timer expires, this worker is eligible to be reaped. The
    exact meaning of "reaped" depends on the client implementation. In
    most cases, reaping means to terminate the worker process.
    Note that the worker is NOT guaranteed to be reaped at exactly
    ``time.now() + after``, but rather the worker is "eligible" for being
    reaped and the ``TimerServer`` that the client talks to will ultimately
    make the decision when and how to reap the workers with expired timers.

    Usage::

        torch.distributed.elastic.timer.configure(LocalTimerClient())
        with expires(after=10):
            torch.distributed.all_reduce(...)
    """
    if client is None:
        if _timer_client is None:
            raise RuntimeError("Configure timer client before using countdown timers.")
        client = _timer_client
    if scope is None:
        # grab the caller file + lineno
        caller = getframeinfo(stack()[1][0])
        scope = f"{caller.filename}#{caller.lineno}"
    expiration = time.time() + after
    client.acquire(scope, expiration)
    try:
        yield
    finally:
        client.release(scope)
