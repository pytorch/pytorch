# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import json
import os
import select
import shutil
import signal
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from torch.distributed.elastic.timer.api import TimerClient, TimerRequest
from torch.distributed.elastic.utils.logging import get_logger

__all__ = ["FileTimerClient", "FileTimerRequest", "FileTimerServer"]

log = get_logger(__name__)

class FileTimerRequest(TimerRequest):
    """
    Data object representing a countdown timer acquisition and release
    that is used between the ``FileTimerClient`` and ``FileTimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.
    ``signal`` is the signal to reap the worker process from the server
    process.
    """

    __slots__ = ["worker_pid", "scope_id", "expiration_time", "signal"]

    def __init__(self, worker_pid: int, scope_id: str, expiration_time: float, signal: int = 0) -> None:
        self.worker_pid = worker_pid
        self.scope_id = scope_id
        self.expiration_time = expiration_time
        self.signal = signal

    def __eq__(self, other) -> bool:
        if isinstance(other, FileTimerRequest):
            return (
                self.worker_pid == other.worker_pid
                and self.scope_id == other.scope_id
                and self.expiration_time == other.expiration_time
                and self.signal == other.signal
            )
        return False

    def to_json(self) -> str:
        return json.dumps(
            {
                "pid": self.worker_pid,
                "scope_id": self.scope_id,
                "expiration_time": self.expiration_time,
                "signal": self.signal
            },
        )


class FileTimerClient(TimerClient):
    """
    Client side of ``FileTimerServer``. This client is meant to be used
    on the same host that the ``FileTimerServer`` is running on and uses
    pid to uniquely identify a worker.
    This client uses a temp file to send timer requests to the
    ``FileTimerServer``. This client is a producer while the
    ``FileTimerServer`` is a consumer. Multiple clients can work with
    the same ``FileTimerServer``.

    Args:

        file_path: str, the path of a watchdog file.

        timeout_sec: int, watchdog timeout.

        scope_id: str, scope ID of worker

        signal: signal, the signal to use to kill the process. Using a
                        negative or zero signal will not kill the process.
    """
    def __init__(self,
        file_path: str,
        timeout_sec: int,
        scope_id: str,
        signal=(signal.SIGKILL if sys.platform != "win32" else
            signal.CTRL_C_EVENT),
    ) -> None:  # type: ignore[attr-defined]
        super().__init__()
        self._file_path = file_path
        self._timeout_sec = timeout_sec
        self.signal = signal
        log.info(
            "Starting %s..."
            " file_path=%s,"
            " timeout=%s,"
            " signal=%s",
            type(self).__name__,
            self._file_path,
            self._timeout_sec,
            self.signal,
        )
        with open(self._file_path, "w") as file:
            # FileTimerRequest at init is used to write client level
            # metadata once to watchdog file, later acquire will not
            # write duplicate data
            request = FileTimerRequest(
                worker_pid=os.getpid(),
                scope_id=scope_id,
                expiration_time=timeout_sec,
                signal=signal,
            )
            json_request = request.to_json()
            file.write(json_request + "\n")

    def acquire(self, scope_id: str, expiration_time: float) -> None:
        if not os.path.isfile(self._file_path):
            raise FileNotFoundError("Could not send the FileTimerRequest because FileTimerServer is not available.")
        # updating watchdog file modified time to current time as heartbeat
        # avoid writing duplicate FileTimerRequest, expiration time is
        # calcuated using 'modified time + timeout'
        os.utime(self._file_path)

    def release(self, scope_id: str) -> None:
        # release is no-op as timer are alive until end of server shutdown
        pass


class FileTimerServer:
    """
    Server that works with ``FileTimerClient``. Clients are expected to be
    running on the same host as the process that is running this server.
    Each host in the job is expected to start its own timer server locally
    and each server instance manages timers for local workers (running on
    processes on the same host).

    Args:

        dir_path: str, the path of directory where watchdog files will be
        created.

        num_clients: int, number of worker clients.

        max_interval: float, max interval in seconds for each watchdog loop.

        daemon: bool, running the watchdog thread in daemon mode or not.
                      A daemon thread will not block a process to stop.
        log_event: Callable[[Dict[str, str]], None], an optional callback for
                logging the events in JSON format.
    """

    def __init__(
        self,
        dir_path: str,
        num_clients: int,
        max_interval: float = 10,
        daemon: bool = True,
        log_event: Optional[Callable[[str, Optional[FileTimerRequest]], None]] = None
    ) -> None:
        self._dir_path = dir_path
        self._num_clients = num_clients
        self._max_interval = max_interval
        self._daemon = daemon
        self._timers: Dict[str, FileTimerRequest] = {}
        self._stop_signaled = False
        self._watchdog_thread: Optional[threading.Thread] = None
        # For test only. Process all requests and stop the server.
        self._run_once = False
        self._log_event = log_event if log_event is not None else lambda name, request: None
        self._clients_metadata: Dict[str, Optional[FileTimerRequest]] = {}


    def start(self) -> None:
        log.info(
            "Starting %s..."
            " max_interval=%s,"
            " daemon=%s",
            type(self).__name__, self._max_interval, self._daemon
        )
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=self._daemon)
        log.info("Starting watchdog thread...")
        self._watchdog_thread.start()
        self._log_event("watchdog started", None)

    def stop(self) -> None:
        log.info("Stopping %s", type(self).__name__)
        self._stop_signaled = True
        if self._watchdog_thread:
            log.info("Stopping watchdog thread...")
            self._watchdog_thread.join(self._max_interval)
            self._watchdog_thread = None
        else:
            log.info("No watchdog thread running, doing nothing")
        if os.path.exists(self._dir_path):
            shutil.rmtree(self._dir_path)
        self._log_event("watchdog stopped", None)

    def run_once(self) -> None:
        self._run_once = True
        if self._watchdog_thread:
            log.info("Stopping watchdog thread...")
            self._watchdog_thread.join()
            self._watchdog_thread = None
        else:
            log.info("No watchdog thread running, doing nothing")

    def _watchdog_loop(self) -> None:
        while not self._stop_signaled:
            try:
                run_once = self._run_once
                self._load_clients()
                self._run_watchdog()
                if run_once:
                    break
            except Exception:
                log.exception("Error running watchdog")

    def _load_clients(self) -> None:
        if len(self._clients_metadata) == self._num_clients:
            return

        client_files = os.listdir(self._dir_path)
        for file in client_files:
            if file not in self._clients_metadata:
                file_name = f"{self._dir_path}/{file}"
                with open(file_name, "r") as fd:
                    request = json.loads(fd.readline())
                    self._clients_metadata[file] = FileTimerRequest(
                        worker_pid=request["pid"],
                        scope_id=request["scope_id"],
                        expiration_time=request["expiration_time"],
                        signal=request["signal"],
                    )

    def _run_watchdog(self) -> None:
        timer_requests = self._get_requests(self._max_interval)
        self.register_timers(timer_requests)
        now = time.time()
        reaped_workers = set()
        for etimer in self.get_expired_timers(now):
            log.info("Reaping worker_pid=[%s]. Expired timer: %s", etimer.worker_pid, etimer.scope_id)
            reaped_workers.add(etimer.scope_id)
            self._log_event("timer expired", etimer)
            if etimer.signal <= 0:
                log.info("No signal specified with worker=[%s]. Do not reap it.", etimer.worker_pid)
                continue
            # clear metadata for reaped worker to avoid keep reaping in future
            self._clients_metadata[etimer.scope_id] = None
            if self._reap_worker(etimer.worker_pid, etimer.signal):
                log.info("Successfully reaped worker=[%s] with signal=%s", etimer.worker_pid, etimer.signal)
                self._log_event("kill worker process", etimer)
            else:
                log.error("Error reaping worker=[%s]. Will retry on next watchdog.", etimer.worker_pid)
        self.clear_timers(reaped_workers)

    def _get_requests(self, max_interval: float) -> List[FileTimerRequest]:
        start = time.time()
        requests = []
        while not self._stop_signaled or self._run_once:
            for metadata in self._clients_metadata.values():
                if not metadata:
                    continue
                last_mtime = os.path.getmtime(f"{self._dir_path}/{metadata.scope_id}")
                requests.append(
                    FileTimerRequest(
                        worker_pid=metadata.worker_pid,
                        scope_id=metadata.scope_id,
                        expiration_time=last_mtime + metadata.expiration_time,
                        signal=metadata.signal,
                    )
                )
            time.sleep(min(max_interval, 10))
            now = time.time()
            if now - start > max_interval:
                break
        return requests

    def register_timers(self, timer_requests: List[FileTimerRequest]) -> None:
        for request in timer_requests:
            scope_id = request.scope_id
            expiration_time = request.expiration_time

            # negative expiration is a proxy for a release call
            if expiration_time < 0:
                if scope_id in self._timers:
                    del self._timers[scope_id]
            else:
                self._timers[scope_id] = request

    def clear_timers(self, workers: Set[str]) -> None:
        for worker in workers:
            if worker in self._timers:
                del self._timers[worker]

    def get_expired_timers(self, deadline: float) -> List[FileTimerRequest]:
        return [timer for timer in self._timers.values() if timer.expiration_time <= deadline]

    def _reap_worker(self, worker_pid: int, signal: int) -> bool:
        try:
            os.kill(worker_pid, signal)
            return True
        except ProcessLookupError:
            log.info("Process with pid=%s does not exist. Skipping", worker_pid)
            return True
        except Exception:
            log.exception("Error terminating pid=%s", worker_pid)
        return False
