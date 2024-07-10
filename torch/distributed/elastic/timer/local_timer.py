# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import multiprocessing as mp
import os
import signal
import time
from queue import Empty
from typing import Any, Dict, List, Set, Tuple

from .api import RequestQueue, TimerClient, TimerRequest, TimerServer


__all__ = ["LocalTimerClient", "MultiprocessingRequestQueue", "LocalTimerServer"]

logger = logging.getLogger(__name__)


class LocalTimerClient(TimerClient):
    """
    Client side of ``LocalTimerServer``. This client is meant to be used
    on the same host that the ``LocalTimerServer`` is running on and uses
    pid to uniquely identify a worker. This is particularly useful in situations
    where one spawns a subprocess (trainer) per GPU on a host with multiple
    GPU devices.
    """

    def __init__(self, mp_queue):
        super().__init__()
        self._mp_queue = mp_queue

    def acquire(self, scope_id, expiration_time):
        pid = os.getpid()
        acquire_request = TimerRequest(pid, scope_id, expiration_time)
        self._mp_queue.put(acquire_request)

    def release(self, scope_id):
        pid = os.getpid()
        release_request = TimerRequest(pid, scope_id, -1)
        self._mp_queue.put(release_request)


class MultiprocessingRequestQueue(RequestQueue):
    """
    A ``RequestQueue`` backed by python ``multiprocessing.Queue``
    """

    def __init__(self, mp_queue: mp.Queue):
        super().__init__()
        self._mp_queue = mp_queue

    def size(self) -> int:
        return self._mp_queue.qsize()

    def get(self, size, timeout: float) -> List[TimerRequest]:
        requests = []
        wait = timeout
        for _ in range(0, size):
            start = time.time()

            try:
                r = self._mp_queue.get(block=True, timeout=wait)
            except Empty:
                break

            requests.append(r)
            wait = wait - (time.time() - start)
            if wait <= 0:
                break

        return requests


class LocalTimerServer(TimerServer):
    """
    Server that works with ``LocalTimerClient``. Clients are expected to be
    subprocesses to the parent process that is running this server. Each host
    in the job is expected to start its own timer server locally and each
    server instance manages timers for local workers (running on processes
    on the same host).
    """

    def __init__(
        self, mp_queue: mp.Queue, max_interval: float = 60, daemon: bool = True
    ):
        super().__init__(MultiprocessingRequestQueue(mp_queue), max_interval, daemon)
        self._timers: Dict[Tuple[Any, str], TimerRequest] = {}

    def register_timers(self, timer_requests: List[TimerRequest]) -> None:
        for request in timer_requests:
            pid = request.worker_id
            scope_id = request.scope_id
            expiration_time = request.expiration_time

            # negative expiration is a proxy for a release call
            if expiration_time < 0:
                self._timers.pop((pid, scope_id), None)
            else:
                self._timers[(pid, scope_id)] = request

    def clear_timers(self, worker_ids: Set[int]) -> None:
        for pid, scope_id in list(self._timers.keys()):
            if pid in worker_ids:
                self._timers.pop((pid, scope_id))

    def get_expired_timers(self, deadline: float) -> Dict[Any, List[TimerRequest]]:
        # pid -> [timer_requests...]
        expired_timers: Dict[Any, List[TimerRequest]] = {}
        for request in self._timers.values():
            if request.expiration_time <= deadline:
                expired_scopes = expired_timers.setdefault(request.worker_id, [])
                expired_scopes.append(request)
        return expired_timers

    def _reap_worker(self, worker_id: int) -> bool:
        try:
            os.kill(worker_id, signal.SIGKILL)
            return True
        except ProcessLookupError:
            logger.info("Process with pid=%s does not exist. Skipping", worker_id)
            return True
        except Exception:
            logger.exception("Error terminating pid=%s", worker_id)
        return False
