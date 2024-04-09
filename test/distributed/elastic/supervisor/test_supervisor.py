#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

try:
    import zmq  # noqa: F401
except ImportError:
    raise unittest.SkipTest("zmq not installed in test harness") from None
import os
import pickle
import signal
import sys

import tempfile
import time
from contextlib import contextmanager
from unittest.mock import Mock, patch

import torch.testing._internal.common_utils as internal

from torch.distributed.elastic import supervisor
from torch.distributed.elastic.supervisor import as_completed, Context, Future


@contextmanager
def context(*args, **kwargs):
    try:
        ctx = Context(*args, **kwargs)
        yield ctx
    finally:
        ctx.shutdown()


@contextmanager
def host_sockets(N: int):
    context: zmq.Context = zmq.Context(1)

    def create_socket():
        backend = context.socket(zmq.DEALER)
        backend.setsockopt(zmq.IPV6, True)
        backend.connect("tcp://127.0.0.1:55555")
        return backend

    sockets = [create_socket() for i in range(N)]
    try:
        yield sockets
    finally:
        context.destroy(linger=500)


class SupervisorUnitTests(internal.TestCase):
    def test_future(self):
        with context() as ctx:

            def future(hostname=None):
                return Future(ctx, "test", hostname)

            f = future()
            self.assertFalse(f.done())
            fstr = str(f)
            self.assertIn("incomplete", fstr)
            ctx._schedule(lambda: f.set_result("finished"))
            self.assertEqual(f.result(timeout=1), "finished")
            fstr = str(f)
            self.assertIn("[complete", fstr)
            f2 = future(hostname=f)
            fstr = str(f2)
            self.assertIn("finished", fstr)
            with self.assertRaises(TimeoutError):
                f2.result(timeout=0.001)
            with self.assertRaises(TimeoutError):
                f2.exception(timeout=0.001)
            ctx._schedule(lambda: f2.set_exception(ValueError("foo")))
            self.assertIsInstance(f2.exception(timeout=1), ValueError)
            with self.assertRaises(ValueError):
                f2.result()

            m = Mock()
            with patch.object(supervisor.logger, "exception", m):
                f3 = future()
                l = []
                f3.add_done_callback(lambda f: l.append(True))

                def err(v):
                    raise ValueError(v)

                f3.add_done_callback(lambda f: err("done0"))
                self.assertFalse(l)
                ctx._schedule(lambda: f3.set_result("finished"))
                f3.result(timeout=1)
                self.assertEqual(l[0], True)
                f3.add_done_callback(lambda f: l.append(4))
                f3.add_done_callback(lambda f: err("done1"))
                self.assertEqual(l[1], 4)
            self.assertEqual(len(m.call_args_list), 2)
            f4 = future(hostname=future())
            self.assertIn("unconnected", str(f4))
            ctx._schedule(lambda: f4.set_result("finished"))
            time.sleep(0.1)
            self.assertEqual(f4.result(timeout=0), "finished")
            f5 = future()
            ctx._schedule(lambda: f5.set_result("finished"))
            self.assertEqual(f5.result(), "finished")

    def test_as_completed(self):
        with context() as ctx:
            futures = [Future(ctx, f"future_{i}", None) for i in range(10)]
            ctx._schedule(lambda: futures[1].set_result(1))
            for f in as_completed(futures, timeout=1):
                self.assertTrue(f is futures[1] and f.result() == 1)
                break
            ctx._schedule(lambda: futures[2].set_result(2))
            a = as_completed(futures, timeout=1)
            for f in a:
                if f is futures[1]:
                    pass
                elif f is futures[2]:
                    self.assertTrue(f.result() == 2)
                    nf = Future(ctx, "new_future", None)
                    a.add(nf)
                    ctx._schedule(lambda: nf.set_result(3))
                else:
                    self.assertIs(f, nf)
                    break
            with self.assertRaises(TimeoutError):
                for x in as_completed(futures[3:], timeout=0.001):
                    self.fail("should have timed out")
            m = Mock()
            with patch.object(
                Context._process_futures, "__defaults__", (0,)
            ), patch.object(supervisor.logger, "info", m):
                with self.assertRaises(TimeoutError):
                    for x in as_completed(futures[3:], timeout=0.001):
                        pass
            # self.assertIn("Waiting for {Future[", m.call_args[0][0])

            for _ in as_completed([]):
                pass

            seen = False
            for x in as_completed([futures[1]]):
                seen = True
            self.assertTrue(seen)

            x = supervisor.wait(futures, return_when=supervisor.FIRST_COMPLETED)
            self.assertTrue(len(x.done))
            self.assertTrue(len(x.not_done))

            x = supervisor.wait(futures[1:3], supervisor.ALL_COMPLETED)
            self.assertEqual(len(x.done), 2)
            self.assertEqual(len(x.not_done), 0)
            ctx._schedule(lambda: futures[4].set_exception(ValueError()))
            x = supervisor.wait(futures[4:6], return_when=supervisor.FIRST_EXCEPTION)
            self.assertTrue(futures[4] in x.done)

            self.assertEqual(len(as_completed(futures)), 10)

    def test_supervisor_api(self):
        with context() as ctx:
            h0, h1 = ctx.request_hosts(2)
            proc = ctx.create_process_group([h0], args=["test"])[0]
            proc.send("hello")
            pm = proc.recv(filter=lambda x: x == "world")
            with host_sockets(1) as (socket,):
                socket.send_pyobj(("_cmd_hostname", None, "host0"))
                self.assertEqual(h0.hostname().result(timeout=1), "host0")
                self.assertIn("host0", str(h0))
                self.assertEqual(socket.recv(), b"")
                expected = (
                    "cmd_launch",
                    0,
                    0,
                    1,
                    1,
                    {"args": ["test"], "env": None, "cwd": None},
                    "pg0",
                    False,
                    None,
                )
                self.assertEqual(socket.recv_pyobj(), expected)
                self.assertEqual(socket.recv_pyobj(), ("cmd_send", 0, "hello"))
                self.assertFalse(pm.done())
                socket.send_pyobj(("_cmd_started", 0, 7))
                socket.send_pyobj(("_cmd_response", 0, pickle.dumps("nope")))
                socket.send_pyobj(("_cmd_response", 0, pickle.dumps("world")))
                self.assertEqual(proc.pid().result(timeout=1), 7)
                self.assertEqual(pm.result(timeout=1), "world")
                self.assertEqual(proc.recv().result(timeout=1), "nope")
                pm1 = proc.recv()
                ctx.return_hosts([h0])
                self.assertTrue(proc.returncode().exception(timeout=1) is not None)
                self.assertTrue(proc.recv().exception(timeout=1) is not None)
                proc = None
                self.assertTrue(h0.connection_lost())
                self.assertTrue(pm1.exception(timeout=1) is not None)
                (p,) = ctx.create_process_group([h0], args=["test3"])
                self.assertTrue(p.pid().exception(timeout=1) is not None)

            with host_sockets(1) as (socket,):
                socket.send_pyobj(("_cmd_hostname", None, "host1"))
                self.assertEqual(socket.recv(), b"")
                self.assertEqual(h1.hostname().result(timeout=1), "host1")
                proc = ctx.create_process_group([h1], args=["test2"])[0]
                self.assertIn("rank=0", str(proc))
                expected = (
                    "cmd_launch",
                    2,
                    0,
                    1,
                    1,
                    {"args": ["test2"], "env": None, "cwd": None},
                    "pg2",
                    False,
                    None,
                )
                self.assertEqual(socket.recv_pyobj(), expected)
                proc = None
                # test sending a message after a proc timeout
                socket.send_pyobj(("_cmd_response", 2, pickle.dumps("old response")))
                socket.send(b"")
                self.assertEqual(socket.recv(), b"")

        # now try with host connecting before the host object exists
        with host_sockets(1) as (socket,):
            socket.send_pyobj(("_cmd_hostname", None, "host0"))
            with context() as ctx:
                self.assertEqual(socket.recv(), b"")
                (h,) = ctx.request_hosts(1)
                self.assertEqual(h.hostname().result(timeout=1), "host0")

    def test_bad_host_managers(self):
        with context() as ctx:
            with host_sockets(5) as (socket0, socket1, socket2, socket3, socket4):
                socket0.send(b"somegarbage")
                self.assertEqual(
                    socket0.recv_pyobj(),
                    ("abort", "Connection did not start with a hostname"),
                )
                socket1.send_pyobj(("_cmd_hostname", None, 7))
                self.assertEqual(
                    socket1.recv_pyobj(),
                    ("abort", "Connection did not start with a hostname"),
                )
                socket2.send_pyobj(("_cmd_hostname", None, "host0"))
                self.assertEqual(socket2.recv(), b"")
                socket2.send_pyobj(("_cmd_started", 0, 7))
                self.assertEqual(
                    socket2.recv_pyobj(),
                    ("abort", "Host manager sent messages before attached."),
                )
                (h,) = ctx.request_hosts(1)
                socket3.send_pyobj(("_cmd_hostname", None, "host3"))
                self.assertEqual(h.hostname().result(timeout=1), "host3")
                socket4.send(b"")
                self.assertEqual(
                    socket4.recv_pyobj(),
                    ("abort", "Connection did not start with a hostname"),
                )

    def test_host_manager_no_heartbeat(self):
        with patch.object(
            supervisor, "HEARTBEAT_INTERVAL", 0.01
        ), context() as ctx, host_sockets(1) as (socket,):
            socket.send_pyobj(("_cmd_hostname", None, "host0"))
            self.assertEqual(socket.recv(), b"")
            socket.send(b"")
            self.assertEqual(socket.recv(), b"")
            (h,) = ctx.request_hosts(1)
            self.assertEqual(socket.recv_pyobj(), ("abort", "Host did not heartbeat"))
            socket.send(b"")
            self.assertEqual(
                socket.recv_pyobj(), ("abort", "Supervisor thought host timed out")
            )
            h.time_connection_lost().result(timeout=1)

    def test_proc_creation(self):
        with context() as ctx, host_sockets(2) as (socket0, socket1):
            h0, h1 = ctx.request_hosts(2)
            socket0.send_pyobj(("_cmd_hostname", None, "host0"))
            self.assertEqual(socket0.recv(), b"")
            socket1.send_pyobj(("_cmd_hostname", None, "host1"))
            self.assertEqual(socket1.recv(), b"")
            pg = ctx.create_process_group([h0, h1], args=["test"], processes_per_host=3)
            self.assertEqual(len(pg), 6)
            pg[0].signal(signal.SIGTERM)
            for i in range(3):
                socket0.recv_pyobj()  # launches
            self.assertEqual(
                socket0.recv_pyobj(), ("cmd_signal", 0, signal.SIGTERM, True)
            )
            socket0.send_pyobj(("_cmd_response", 0, pickle.dumps("hello")))
            socket0.send_pyobj(("_cmd_response", 0, pickle.dumps("world")))
            self.assertEqual(
                "world", pg[0].recv(lambda x: x == "world").result(timeout=1)
            )
            self.assertEqual("hello", pg[0].recv().result(timeout=1))
            socket0.send_pyobj(("_cmd_started", 1, 8))
            socket0.send_pyobj(("_cmd_exited", 1, 7))
            self.assertEqual(7, pg[1].returncode().result(timeout=1))
            socket0.send_pyobj(("_cmd_started", 2, "Failed"))
            self.assertTrue(pg[2].pid().exception(timeout=1) is not None)

    def test_log_redirect(self):
        m = Mock()
        with tempfile.NamedTemporaryFile(delete=True) as f, patch.object(
            os, "dup2", m
        ), patch.object(sys, "stdout", m), patch.object(sys, "stderr", m), context(
            log_format=f.name
        ):
            pass
        m.assert_called()

    def test_host_lost_first(self):
        with context() as ctx, host_sockets(1) as (socket0,):
            (h0,) = ctx.request_hosts(1)
            (h1,) = ctx.replace_hosts([h0])
            self.assertTrue(h0.hostname().exception(timeout=1) is not None)
            socket0.send_pyobj(("_cmd_hostname", None, "host0"))
            self.assertEqual(socket0.recv(), b"")
            self.assertEqual("host0", h1.hostname().result(timeout=1))

    def test_host_replace(self):
        with context() as ctx:
            b = ctx.request_hosts(2)
            nh = ctx.replace_hosts(x for x in b)
            self.assertEqual(len(b), len(nh))


if __name__ == "__main__":
    internal.run_tests()
