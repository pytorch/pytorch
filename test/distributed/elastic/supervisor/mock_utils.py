#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest

from unittest.mock import patch

import torch.distributed.elastic.supervisor.hostmanager as hostmanager

try:
    import zmq  # noqa: F401
except ImportError:
    raise unittest.SkipTest("zmq not installed in test harness") from None

import os
import signal
import subprocess
import threading
from contextlib import contextmanager


@contextmanager
def mock_process_handling():
    lock = threading.Lock()
    all_processes = []

    def killpg(pid, return_code):
        with lock:
            p = all_processes[pid]
            if p.immortal and return_code != signal.SIGKILL:
                return
            if not hasattr(p, "returncode"):
                p.returncode = return_code
                with os.fdopen(p._done_w, "w") as f:
                    f.write("done")

    class MockPopen:
        def __init__(self, *args, fail=False, immortal=False, **kwargs):
            if fail:
                raise RuntimeError("process fail")
            with lock:
                self.args = args
                self.kwargs = kwargs
                self.immortal = immortal
                self.pid = len(all_processes)
                all_processes.append(self)
                self.signals_sent = []
                self._done_r, self._done_w = os.pipe()

        def send_signal(self, sig):
            killpg(self.pid, sig)

        def wait(self):
            return self.returncode

    def mock_pidfdopen(pid):
        with lock:
            return all_processes[pid]._done_r

    with (
        patch.object(subprocess, "Popen", MockPopen),
        patch.object(hostmanager, "pidfd_open", mock_pidfdopen),
        patch.object(os, "killpg", killpg),
    ):
        yield killpg


@contextmanager
def connected_hostmanager():
    context: zmq.Context = zmq.Context(1)
    backend = context.socket(zmq.ROUTER)
    backend.setsockopt(zmq.IPV6, True)
    backend.bind("tcp://*:55555")
    exited = None
    host = hostmanager._Hostmanager("tcp://127.0.0.1:55555")

    def run_host():
        nonlocal exited
        try:
            host.run_event_loop_forever()
        except ConnectionAbortedError as e:
            exited = e
        except SystemExit:
            exited = True

    thread = threading.Thread(target=run_host, daemon=True)
    thread.start()
    try:
        yield backend, host
    finally:
        backend.close()
        context.term()
        thread.join(timeout=1)
        if thread.is_alive():
            raise RuntimeError("thread did not terminate")
    host.context.destroy(linger=500)
    if exited != True:  # noqa: E712
        raise exited
