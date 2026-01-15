#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import json
import os
import pickle
import socket
import tempfile
import unittest
from contextlib import contextmanager

from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool

from torch.distributed.elastic.control_plane import (
    TORCH_WORKER_SERVER_SOCKET,
    worker_main,
)
from torch.monitor import _WaitCounter
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    MI200_ARCH,
    requires_cuda,
    run_tests,
    skipIfRocmArch,
    TestCase,
)


class UnixHTTPConnection(HTTPConnection):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")

        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


class UnixHTTPConnectionPool(HTTPConnectionPool):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")

        self.socket_path = socket_path

    def _new_conn(self):
        return UnixHTTPConnection(self.socket_path)


@contextmanager
def local_worker_server() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        socket_path = os.path.join(tmpdir, "socket.sock")
        os.environ[TORCH_WORKER_SERVER_SOCKET] = socket_path

        with worker_main():
            pool = UnixHTTPConnectionPool(socket_path)
            yield pool


class WorkerServerTest(TestCase):
    def test_worker_server(self) -> None:
        with local_worker_server() as pool:
            resp = pool.request("GET", "/")
            self.assertEqual(resp.status, 200)
            self.assertEqual(
                resp.data,
                b"<h1>torch.distributed.WorkerServer</h1>\n"
                b'<a href="'
                b"/handler/"
                b'">Handler names</a>\n',
            )

            resp = pool.request("POST", "/handler/ping")
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.data, b"pong")

            resp = pool.request("GET", "/handler/")
            self.assertEqual(resp.status, 200)
            self.assertIn("ping", json.loads(resp.data))

            resp = pool.request("POST", "/handler/nonexistent")
            self.assertEqual(resp.status, 404)
            self.assertIn(b"Handler nonexistent not found:", resp.data)

    @requires_cuda
    def test_dump_nccl_trace_pickle(self) -> None:
        with local_worker_server() as pool:
            resp = pool.request("POST", "/handler/dump_nccl_trace_pickle")
            self.assertEqual(resp.status, 200)
            out = pickle.loads(resp.data)
            self.assertIsInstance(out, dict)
            self.assertIn("version", out)

    @requires_cuda
    def test_dump_nccl_trace_pickle_with_params(self) -> None:
        with local_worker_server() as pool:
            # bad key - not lower case
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includeCollectives=true"
            )
            self.assertEqual(resp.status, 400)
            # unknown key
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?unknownkey=true"
            )
            self.assertEqual(resp.status, 400)
            # bad value - not a bool
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includecollectives=notabool"
            )
            self.assertEqual(resp.status, 400)
            # bad value - value not lowercase
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includecollectives=True"
            )
            self.assertEqual(resp.status, 400)
            # good key and value
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_pickle?includecollectives=true"
            )
            self.assertEqual(resp.status, 200)
            # multiple good keys and values
            resp = pool.request(
                "POST",
                "/handler/dump_nccl_trace_pickle?includecollectives=true&includestacktraces=false&onlyactive=true",
            )
            self.assertEqual(resp.status, 200)

    @requires_cuda
    def test_dump_nccl_trace_pickle_with_json(self) -> None:
        with local_worker_server() as pool:
            # bad key - not lower case
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_json?includeCollectives=true"
            )
            self.assertEqual(resp.status, 400)
            # unknown key
            resp = pool.request("POST", "/handler/dump_nccl_trace_json?unknownkey=true")
            self.assertEqual(resp.status, 400)
            # bad value - not a bool
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_json?includecollectives=notabool"
            )
            self.assertEqual(resp.status, 400)
            # bad value - value not lowercase
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_json?includecollectives=True"
            )
            self.assertEqual(resp.status, 400)
            # good key and value
            resp = pool.request(
                "POST", "/handler/dump_nccl_trace_json?includecollectives=true"
            )
            self.assertEqual(resp.status, 200)
            # multiple good keys and values
            resp = pool.request(
                "POST",
                "/handler/dump_nccl_trace_json?includecollectives=true&onlyactive=true",
            )
            self.assertEqual(resp.status, 200)

    @skipIfRocmArch(MI200_ARCH)
    def test_tcp(self) -> None:
        import requests

        from torch._C._distributed_c10d import _WorkerServer

        server = _WorkerServer("", 1234)
        out = requests.get("http://localhost:1234/handler/")
        self.assertEqual(out.status_code, 200)

        server.shutdown()

    def test_dump_traceback(self) -> None:
        with local_worker_server() as pool:
            resp = pool.request("POST", "/handler/dump_traceback")
            self.assertEqual(resp.status, 200)
            self.assertIn(b"in test_dump_traceback\n", resp.data)

    def test_run_handler(self) -> None:
        from torch._C._distributed_c10d import _get_handler, _Request, _Response

        handler = _get_handler("ping")

        class Request(_Request):
            def __init__(self) -> None:
                _Request.__init__(self)

            def body(self) -> bytes:
                return b"dummy"

            def params(self) -> dict[str, str]:
                return {}

        class Response(_Response):
            def __init__(self) -> None:
                _Response.__init__(self)

            def set_content(self, content: str, content_type: str) -> None:
                self.content = content
                self.content_type = content_type

            def set_status(self, status: int) -> None:
                self.status = status

        req = Request()
        resp = Response()

        handler(req, resp)

        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.content, "pong")
        self.assertEqual(resp.content_type, "text/plain")

    def test_get_handler_nonexistant(self) -> None:
        from torch._C._distributed_c10d import _get_handler

        with self.assertRaisesRegex(ValueError, "Failed to find handler nonexistent"):
            _get_handler("nonexistent")

    def test_get_handler_names(self) -> None:
        from torch._C._distributed_c10d import _get_handler_names

        names = _get_handler_names()
        self.assertIn("ping", names)

    @unittest.skipIf(IS_FBCODE, "disabled in FBCODE")
    def test_wait_counter_values(self) -> None:
        """
        Test that WaitCounter values are properly tracked and returned by the handler.

        Note: This test may trigger an ASAN heap-use-after-free error during process
        shutdown due to static destruction order issues with boost regex in the logging
        framework. The test assertions pass successfully before this shutdown error occurs.
        """
        with local_worker_server() as pool:
            # Create and use a WaitCounter with a specific name
            counter_name = "test_counter"
            counter = _WaitCounter(counter_name)

            # Use the counter multiple times to generate metrics
            # Note: Using minimal/no sleep to avoid timing issues
            for i in range(3):
                with counter.guard():
                    pass  # Minimal work

            # Query the wait counter values
            resp = pool.request("POST", "/handler/wait_counter_values")
            self.assertEqual(resp.status, 200)

            # Parse the JSON response
            data = json.loads(resp.data)
            # Should be a dictionary
            self.assertIsInstance(data, dict)

            # Verify our test counter appears in the response
            self.assertIn(
                counter_name,
                data,
                f"Counter '{counter_name}' not found in response. Available counters: {list(data.keys())}",
            )

            # Verify the counter has expected metrics
            counter_data = data[counter_name]
            self.assertIn("active_count", counter_data)
            self.assertIn("total_calls", counter_data)
            self.assertIn("total_time_us", counter_data)
            self.assertIn("max_time_us", counter_data)

            # Verify the counter was called 3 times
            self.assertEqual(
                counter_data["total_calls"],
                3,
                f"Expected 3 calls, got {counter_data['total_calls']}",
            )

            # Verify active_count is 0 (no active waiters)
            self.assertEqual(
                counter_data["active_count"],
                0,
                f"Expected 0 active, got {counter_data['active_count']}",
            )

            # total_time_us and max_time_us may be 0 or very small for fast operations
            # Just verify they exist and are non-negative
            self.assertGreaterEqual(counter_data["total_time_us"], 0)
            self.assertGreaterEqual(counter_data["max_time_us"], 0)


if __name__ == "__main__":
    run_tests()
