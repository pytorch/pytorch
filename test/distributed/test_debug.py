# Owner(s): ["oncall: distributed"]

import os
import shutil
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
import torch.distributed as dist
import torch.distributed.debug as debug_module
from torch.distributed.debug import start_debug_server, stop_debug_server
from torch.testing._internal.common_utils import run_tests, TestCase


session = requests.Session()
retry_strategy = Retry(total=5, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)


def _reset_debug_server_state() -> None:
    debug_module._WORKER_SERVER = None
    debug_module._DEBUG_SERVER_PROC = None


class TestDebug(TestCase):
    def setUp(self) -> None:
        super().setUp()
        _reset_debug_server_state()

    def tearDown(self) -> None:
        super().tearDown()
        if (
            debug_module._DEBUG_SERVER_PROC is not None
            or debug_module._WORKER_SERVER is not None
        ):
            try:
                stop_debug_server()
            except Exception:
                pass
        _reset_debug_server_state()

    def test_all(self) -> None:
        store = dist.TCPStore("localhost", 0, 1, is_master=True, wait_for_workers=False)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(store.port)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        port = 25998

        def fetch(path: str) -> str:
            resp = session.get(f"http://localhost:{port}{path}")
            resp.raise_for_status()
            return resp.text

        start_debug_server(port=port)

        with self.subTest("index"):
            self.assertIn("torch profiler", fetch("/"))

        with self.subTest("profile"):
            self.assertIn("View 0", fetch("/profile?duration=0.01"))

        with self.subTest("stacks"):
            self.assertIn("test_all", fetch("/stacks"))

        with self.subTest("wait_counters"):
            self.assertIn("Rank 0", fetch("/wait_counters"))

        with self.subTest("fr_trace"):
            self.assertIn("Memberships", fetch("/fr_trace"))
            self.assertIn("pg_status", fetch("/fr_trace_json"))

            if torch.cuda.is_available():
                self.assertIn("Memberships", fetch("/fr_trace_nccl"))
                self.assertIn("pg_status", fetch("/fr_trace_nccl_json"))

        with self.subTest("error codes"):
            resp = session.get(f"http://localhost:{port}/blah")
            self.assertEqual(resp.status_code, 404)
            self.assertIn("Handler not found: /blah", resp.text)

        with self.subTest("tcpstore"):
            store.set("test", "value")
            store.set("test2", "a" * 1000)
            out = fetch("/tcpstore")
            self.assertIn("test: b'value'", out)
            self.assertIn("test2: b'" + "a" * 95 + "...", out)

        with self.subTest("pyspy"):
            if shutil.which("py-spy"):
                self.assertIn("test_all", fetch("/pyspy_dump"))
                self.assertIn("_frontend", fetch("/pyspy_dump?subprocesses=1"))
                self.assertIn("libc.so", fetch("/pyspy_dump?native=1"))

        stop_debug_server()

    def test_start_method_spawn(self) -> None:
        store = dist.TCPStore("localhost", 0, 1, is_master=True, wait_for_workers=False)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(store.port)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        port = 25997

        start_debug_server(port=port, start_method="spawn")
        time.sleep(1)

        self.assertIsNotNone(debug_module._DEBUG_SERVER_PROC)
        self.assertIsNotNone(debug_module._WORKER_SERVER)

        stop_debug_server()

    def test_start_method_forkserver(self) -> None:
        store = dist.TCPStore("localhost", 0, 1, is_master=True, wait_for_workers=False)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(store.port)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        port = 25996

        start_debug_server(port=port, start_method="forkserver")
        time.sleep(1)

        self.assertIsNotNone(debug_module._DEBUG_SERVER_PROC)
        self.assertIsNotNone(debug_module._WORKER_SERVER)

        stop_debug_server()


if __name__ == "__main__":
    run_tests()
