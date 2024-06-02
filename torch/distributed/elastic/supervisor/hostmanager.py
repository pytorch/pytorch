# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import io
import logging
import os
import pickle
import signal
import socket
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from random import random
from typing import Callable, Dict, List, Mapping, Optional

import zmq  # type: ignore[import-not-found]

from torch.distributed.elastic.supervisor import (
    HEARTBEAT_INTERVAL,
    HEARTBEAT_LIVENESS,
    ProcessFailedToStart,
)
from torch.distributed.elastic.utils.logging import get_logger

logger: logging.Logger = get_logger(__name__)

ABORT_INTERVAL = 5
__NR_pidfd_open = 434
libc = ctypes.CDLL(None)


# older libc do not have this syscall
def _pidfd_open(pid: int) -> int:
    return libc.syscall(__NR_pidfd_open, pid, 0)


# *******************************************
# objects in this file represent Host/Process
# on the host machine itself.
#
# _Hostmanager::run_event_loop_forever is the
# main event loop that is responsible for
# communicating with the supervisor and
# responding to commands
# *******************************************


# main package has Host/Process used by
# the supervisor.
class _Process:
    def __init__(
        self,
        name: str,
        logfilename: Optional[str],
        proc_comm: zmq.Socket,
        proc_id: int,
        rank: int,
        processes_per_host: int,
        world_size: int,
        popen: Mapping[str, object],
        proc_addr: str,
    ) -> None:
        self.proc_id = proc_id
        self.proc_comm = proc_comm
        environ = dict(os.environ)
        if popen["env"] is not None:
            # pyre-ignore
            environ.update(popen["env"])  # type: ignore[call-overload]
        environ["RANK"] = str(rank)
        environ["GROUP_RANK"] = "DUMMY"  # TODO: introduce group ranking
        environ["ROLE_RANK"] = str(rank)  # TODO: add support for multi-role job
        environ["ROLE_NAME"] = "trainer"
        environ["ROLE_WORLD_SIZE"] = str(world_size)
        environ["WORLD_SIZE"] = str(world_size)
        environ["LOCAL_RANK"] = str(rank % processes_per_host)
        environ["LOCAL_WORLD_SIZE"] = str(processes_per_host)
        environ["SUPERVISOR_PIPE"] = proc_addr
        environ["SUPERVISOR_IDENT"] = str(proc_id)
        popen = {**popen, "env": environ}
        try:
            if logfilename is None:
                logcontext = nullcontext()
            else:
                try:
                    logcontext = open(logfilename, "a")  # type: ignore[assignment]
                except FileNotFoundError:
                    Path(logfilename).parent.mkdir(exist_ok=True, parents=True)
                    logcontext = open(logfilename, "a")  # type: ignore[assignment]
            with logcontext as logfile:
                logger.debug("Staring process on host: %s", popen)
                self.subprocess: subprocess.Popen[str] = subprocess.Popen(
                    # pyre-ignore
                    **popen,
                    start_new_session=True,
                    stdout=logfile,
                    stderr=logfile,
                )  # type: ignore[call-overload]
        except Exception as e:
            s = io.StringIO()
            traceback.print_exc(file=s)
            logger.warning("Process failed to start: %s\n", s.getvalue())
            raise ProcessFailedToStart(s.getvalue()) from e
        self.proc_id_bytes: bytes = proc_id.to_bytes(8, byteorder="little")
        self.deferred_sends: Optional[List[bytes]] = []

    def _send(self, _msg: object) -> None:
        msg = pickle.dumps(_msg)

        if self.deferred_sends is not None:
            self.deferred_sends.append(msg)
        else:
            self.proc_comm.send_multipart([self.proc_id_bytes, msg])

    def _notify_connected(self) -> None:
        deferred_sends = self.deferred_sends
        if deferred_sends is not None:
            for msg in deferred_sends:
                self.proc_comm.send_multipart([self.proc_id_bytes, msg])
            self.deferred_sends = None

    def __repr__(self) -> str:
        return f"_Process[{self.proc_id}]"


class _Hostmanager:
    """
    Represents a host that can be supervised.
    Starts an event loop listening for commands from the supervisor, including launching/killing processes.
    """

    def __init__(self, supervisor_addr: str) -> None:
        self.context: zmq.Context = zmq.Context(1)
        self.supervisor_comm: zmq.Socket = self.context.socket(zmq.DEALER)
        self.supervisor_comm.setsockopt(zmq.IPV6, True)

        logger.debug("Connecting to %s", supervisor_addr)
        self.supervisor_comm.connect(supervisor_addr)

        # tell the supervisor we exist, and provide
        # hostname for debugging.
        self.supervisor_comm.send(
            pickle.dumps(("_cmd_hostname", None, socket.gethostname()))
        )

        self.poller = zmq.Poller()
        self.poller.register(self.supervisor_comm, zmq.POLLIN)

        # optional way to send messages to processes.
        # all processes on this host will use the same
        # socket.
        self.proc_comm: zmq.Socket = self.context.socket(zmq.ROUTER)
        self.proc_addr = f"ipc:///tmp/proc_{os.getpid()}"
        self.proc_comm.bind(self.proc_addr)
        self.poller.register(self.proc_comm, zmq.POLLIN)

        self.process_table: Dict[bytes, _Process] = {}
        self.fd_to_on_exit: Dict[int, Callable[[], None]] = {}
        self._launches = 0
        self.has_shutdown = False

    def heartbeat(self) -> None:
        self.supervisor_comm.send(b"")

    def cmd_launch(
        self,
        proc_id: int,
        rank: int,
        processes_per_host: int,
        world_size: int,
        popen: Mapping[str, object],
        name: str,
        simulate: bool,
        log_file: Optional[str],
    ) -> None:
        logger.debug("cmd_launch: %s on rank %s", popen, rank)
        self._launches += 1
        if simulate:
            self.supervisor_comm.send(pickle.dumps(("_cmd_started", proc_id, 2)))
            self.supervisor_comm.send(pickle.dumps(("_cmd_exited", proc_id, 0)))
            return
        try:
            process = _Process(
                name,
                log_file,
                self.proc_comm,
                proc_id,
                rank,
                processes_per_host,
                world_size,
                popen,
                self.proc_addr,
            )
            self.process_table[process.proc_id_bytes] = process
            self.on_subprocess_exit(
                process.subprocess, lambda: self.process_exit(process)
            )
            reply = process.subprocess.pid
        except ProcessFailedToStart as e:
            reply = str(e)  # type: ignore[assignment]
        self.supervisor_comm.send(pickle.dumps(("_cmd_started", proc_id, reply)))

    def process_exit(self, process: _Process) -> None:
        logger.debug("%s exited", process)
        self.process_table.pop(process.proc_id_bytes)
        # we do not allow descendents to outlive the parent
        # if any remain this kill will clean them up
        os.killpg(process.subprocess.pid, signal.SIGKILL)
        returncode = process.subprocess.wait()
        if not self.has_shutdown:
            self.supervisor_comm.send(
                pickle.dumps(("_cmd_exited", process.proc_id, returncode))
            )

    def on_subprocess_exit(
        self, subprocess: subprocess.Popen, on_exit: Callable[[], None]
    ) -> None:
        fd: int = _pidfd_open(subprocess.pid)
        self.fd_to_on_exit[fd] = on_exit
        self.poller.register(fd, zmq.POLLIN)

    def cmd_send(self, _proc_id: int, msg: bytes) -> None:
        proc_id = _proc_id.to_bytes(8, byteorder="little")
        if proc_id in self.process_table:
            process = self.process_table[proc_id]
            process._send(msg)

    def cmd_signal(self, _proc_id: int, sig: int, group: bool) -> None:
        logger.debug("cmd_signal: signal %s on proc %s", sig, _proc_id)
        proc_id = _proc_id.to_bytes(8, byteorder="little")
        if proc_id in self.process_table:
            process = self.process_table[proc_id]
            if group:
                os.killpg(process.subprocess.pid, sig)
            else:
                process.subprocess.send_signal(sig)

    def _fd_exit(self, fd: int) -> None:
        on_exit = self.fd_to_on_exit.pop(fd)
        self.poller.unregister(fd)
        os.close(fd)
        on_exit()

    def shutdown(self) -> None:
        if self.has_shutdown:
            return
        self.has_shutdown = True
        for proc in self.process_table.values():
            os.killpg(proc.subprocess.pid, signal.SIGTERM)
        expiry = time.time() + ABORT_INTERVAL
        ttl: float = ABORT_INTERVAL
        while ttl > 0 and self.process_table:
            for s, _ in self.poller.poll(timeout=int(1000 * ttl)):
                if isinstance(s, int):
                    self._fd_exit(s)
            ttl = time.time() - expiry
        if self.process_table:
            for proc in self.process_table.values():
                os.killpg(proc.subprocess.pid, signal.SIGKILL)

    def abort(self, with_error: Optional[str] = None) -> None:
        self.shutdown()
        if with_error:
            logger.error("Host manager exiting with error: %s", with_error)
            raise ConnectionAbortedError(with_error)
        else:
            logger.debug("Host manager exiting")
            sys.exit(0)

    def run_event_loop_forever(self) -> None:
        expiry = None
        heartbeat_at: float = 0

        while True:
            timeout = (
                None
                if expiry is None
                else int(max(1000 * (heartbeat_at - time.time()) + 1, 0))
            )
            for sock, _ in self.poller.poll(timeout=timeout):
                if isinstance(sock, int):
                    # we register a file descriptor to the poller, which is a raw
                    # int file description that becomes ready when the a subprocess exits
                    # see pidfd_open.
                    self._fd_exit(sock)
                elif sock is self.supervisor_comm:
                    if expiry is None:
                        logger.debug("Connected to supervisor")
                        # first heartbeat is set to between 0 to HEARTBEAT_INTERVAL
                        # to spread out the heartbeats from hosts that all start
                        # at the same time.
                        heartbeat_at = time.time() + HEARTBEAT_INTERVAL * random()

                    expiry = time.time() + HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS
                    msg = self.supervisor_comm.recv()
                    if msg:
                        cmd, *args = pickle.loads(msg)
                        getattr(self, cmd)(*args)
                elif sock is self.proc_comm:
                    proc_id_bytes, msg = self.proc_comm.recv_multipart()
                    process = self.process_table.get(proc_id_bytes)
                    # it is possible for the process to have already exited before
                    # we get its messages, so process_table will be empty
                    if process is not None:
                        process._notify_connected()
                    if len(msg):
                        proc_id = int.from_bytes(proc_id_bytes, byteorder="little")
                        self.supervisor_comm.send(
                            pickle.dumps(("_cmd_response", proc_id, msg))
                        )
            if expiry is not None:
                t = time.time()
                if t > heartbeat_at:
                    heartbeat_at = t + HEARTBEAT_INTERVAL
                    self.heartbeat()
                if t > expiry:
                    duration = HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS
                    self.abort(
                        f"No messages from supervisor for {duration} seconds, aborting."
                    )


def main(addr: str) -> None:
    logger.info("Hostmanager: supervisor address %s", addr)
    manager: _Hostmanager = _Hostmanager(addr)

    def handler(signal: int, frame: object) -> None:
        manager.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    try:
        manager.run_event_loop_forever()
    except Exception as e:
        logger.exception("Failed while running eventloop")
        raise e
    finally:
        manager.shutdown()


if __name__ == "__main__":
    logger.info("starting hostnamager")
    (addr,) = sys.argv[1:]
    main(addr)


__all__ = ["main"]
