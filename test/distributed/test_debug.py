# Owner(s): ["oncall: distributed"]

import os
import shutil
import tempfile
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
import torch.distributed as dist
import torch.distributed.debug as debug_module
from torch.distributed.debug import start_debug_server, stop_debug_server
from torch.distributed.debug._frontend import (
    DebugHandler,
    NavLink,
    PeriodicDumper,
    Route,
)
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


class _StubHandler(DebugHandler):
    """Minimal handler for testing PeriodicDumper without network access."""

    def __init__(self, name: str, content: str | None) -> None:
        self._name = name
        self._content = content

    def routes(self) -> list[Route]:
        return []

    def nav_links(self) -> list[NavLink]:
        return []

    def dump(self) -> str | None:
        return self._content

    def dump_filename(self) -> str:
        return self._name


class _ErrorHandler(DebugHandler):
    """Handler whose dump() always raises."""

    def routes(self) -> list[Route]:
        return []

    def nav_links(self) -> list[NavLink]:
        return []

    def dump(self) -> str | None:
        raise RuntimeError("boom")

    def dump_filename(self) -> str:
        return "error_handler"


class TestPeriodicDumper(TestCase):
    def test_writes_dump_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h = _StubHandler("mystub", "hello world")
            dumper = PeriodicDumper([h], tmp, interval_seconds=0.1)
            dumper.start()
            time.sleep(0.35)
            dumper.stop()

            files = os.listdir(tmp)
            self.assertGreater(len(files), 0)
            self.assertTrue(all(f.startswith("mystub_") for f in files))
            with open(os.path.join(tmp, files[0])) as f:
                self.assertEqual(f.read(), "hello world")

    def test_skips_none_dump(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h = _StubHandler("nodump", None)
            dumper = PeriodicDumper([h], tmp, interval_seconds=0.1)
            dumper.start()
            time.sleep(0.25)
            dumper.stop()

            self.assertEqual(os.listdir(tmp), [])

    def test_enabled_dumps_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h1 = _StubHandler("included", "yes")
            h2 = _StubHandler("excluded", "no")
            enabled = {"included"}
            filtered = [h for h in [h1, h2] if h.dump_filename() in enabled]
            dumper = PeriodicDumper(filtered, tmp, interval_seconds=0.1)
            dumper.start()
            time.sleep(0.25)
            dumper.stop()

            files = os.listdir(tmp)
            self.assertGreater(len(files), 0)
            self.assertTrue(all(f.startswith("included_") for f in files))

    def test_enabled_dumps_none_runs_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h1 = _StubHandler("alpha", "a")
            h2 = _StubHandler("beta", "b")
            dumper = PeriodicDumper([h1, h2], tmp, interval_seconds=0.1)
            dumper.start()
            time.sleep(0.25)
            dumper.stop()

            files = os.listdir(tmp)
            alpha_files = [f for f in files if f.startswith("alpha_")]
            beta_files = [f for f in files if f.startswith("beta_")]
            self.assertGreater(len(alpha_files), 0)
            self.assertGreater(len(beta_files), 0)

    def test_survives_handler_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            err = _ErrorHandler()
            ok = _StubHandler("ok", "survived")
            dumper = PeriodicDumper([err, ok], tmp, interval_seconds=0.1)
            dumper.start()
            time.sleep(0.25)
            dumper.stop()

            files = os.listdir(tmp)
            ok_files = [f for f in files if f.startswith("ok_")]
            self.assertGreater(len(ok_files), 0)
            err_files = [f for f in files if f.startswith("error_handler_")]
            self.assertEqual(len(err_files), 0)

    def test_stop_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h = _StubHandler("idem", "data")
            dumper = PeriodicDumper([h], tmp, interval_seconds=60.0)
            dumper.start()
            dumper.stop()
            dumper.stop()


if __name__ == "__main__":
    run_tests()
