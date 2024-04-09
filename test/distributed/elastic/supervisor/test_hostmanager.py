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
from torch.distributed.elastic.supervisor import get_message_queue

try:
    import zmq  # noqa: F401
except ImportError:
    raise unittest.SkipTest("zmq not installed in test harness") from None

import pickle
import time
from socket import gethostname

import torch.testing._internal.common_utils as testing_common

from .mock_utils import connected_hostmanager, mock_process_handling


class HostmanagerUnitTests(testing_common.TestCase):
    def _launch(
        self,
        dest: (zmq.Socket, bytes),
        proc_id,
        rank=0,
        processes_per_host=1,
        world_size=1,
        popen={"env": None},  # noqa: B006
        name="fake",
        simulate=False,
        log_file=None,
    ):
        msg = (
            "cmd_launch",
            proc_id,
            rank,
            processes_per_host,
            world_size,
            popen,
            name,
            simulate,
            log_file,
        )
        dest[0].send_multipart([dest[1], pickle.dumps(msg)])

    def test_host_manager(self):
        with (
            mock_process_handling() as kill,
            connected_hostmanager() as (sup_sock, host),
            patch.object(hostmanager, "ABORT_INTERVAL", 0.01),
        ):
            idenity, msg = sup_sock.recv_multipart()
            dest = (sup_sock, idenity)

            def send(msg):
                sup_sock.send_multipart([idenity, pickle.dumps(msg)])

            def recv():
                return pickle.loads(sup_sock.recv_multipart()[1])

            _cmd, _, hostname = pickle.loads(msg)

            self.assertEqual(_cmd, "_cmd_hostname")
            self.assertEqual(hostname, gethostname())

            self._launch(dest, proc_id=1)
            self.assertEqual(recv(), ("_cmd_started", 1, 0))
            kill(0, 4)
            self.assertEqual(recv(), ("_cmd_exited", 1, 4))

            self._launch(dest, proc_id=2)
            self.assertEqual(recv(), ("_cmd_started", 2, 1))
            send(("cmd_send", 2, "a message"))
            msg_queue = get_message_queue(2, host.proc_addr)
            self.assertEqual(pickle.loads(msg_queue.recv()), "a message")
            send(("cmd_send", 2, "another message"))
            self.assertEqual(pickle.loads(msg_queue.recv()), "another message")
            msg_queue.send(b"a reply")
            msg_queue.close()
            msg_queue.context.term()
            self.assertEqual(recv(), ("_cmd_response", 2, b"a reply"))
            send(("cmd_signal", 2, 8, True))
            self.assertEqual(recv(), ("_cmd_exited", 2, 8))
            self._launch(dest, proc_id=3, popen={"env": {"foo": "3"}})
            self.assertEqual(recv(), ("_cmd_started", 3, 2))
            send(("cmd_signal", 3, 9, False))
            self.assertEqual(recv(), ("_cmd_exited", 3, 9))
            self._launch(dest, proc_id=4, popen={"fail": True, "env": None})
            _started, _, msg = recv()
            self.assertEqual(_started, "_cmd_started")
            self.assertIn("process fail", msg)
            self._launch(dest, proc_id=5, simulate=True)
            self.assertEqual(recv(), ("_cmd_started", 5, 2))
            self.assertEqual(recv(), ("_cmd_exited", 5, 0))
            self._launch(dest, proc_id=6)  # leave something open
            self._launch(dest, proc_id=7, popen={"immortal": True, "env": None})
            send(("abort", None))
        # test double shutdown
        host.shutdown()

        with self.assertRaises(ConnectionAbortedError):
            with connected_hostmanager() as (sup_sock, _):
                f, msg = sup_sock.recv_multipart()
                sup_sock.send_multipart([f, pickle.dumps(("abort", "An error"))])

    def test_host_timeout_and_heartbeat(self):
        with self.assertRaises(ConnectionAbortedError):
            with patch.object(
                hostmanager, "HEARTBEAT_INTERVAL", 0.01
            ), connected_hostmanager() as (socket, host):
                f, msg = socket.recv_multipart()
                socket.send_multipart([f, b""])
                time.sleep(0.1)
                f, msg = socket.recv_multipart()
                self.assertEqual(msg, b"")


if __name__ == "__main__":
    testing_common.run_tests()
