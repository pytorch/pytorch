#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import json
import os
import pickle
import socket
import tempfile
from contextlib import contextmanager

from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool

from torch.distributed.elastic.control_plane import (
    TORCH_WORKER_SERVER_SOCKET,
    worker_main,
)
from torch.testing._internal.common_utils import requires_cuda, run_tests, TestCase


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
                b"""<h1>torch.distributed.WorkerServer</h1>
<a href="/handler/">Handler names</a>
""",
            )

            resp = pool.request("POST", "/handler/ping")
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.data, b"pong")

            resp = pool.request("GET", "/handler/")
            self.assertEqual(resp.status, 200)
            self.assertIn("ping", json.loads(resp.data))

            resp = pool.request("POST", "/handler/nonexistant")
            self.assertEqual(resp.status, 404)
            self.assertIn(b"Handler nonexistant not found:", resp.data)

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


if __name__ == "__main__":
    run_tests()
