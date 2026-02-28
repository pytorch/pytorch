# Owner(s): ["oncall: distributed"]

import os
import shutil
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
import torch.distributed as dist
import torch.distributed.debug as debug_module
from torch.distributed.debug import start_debug_server, stop_debug_server
from torch.distributed.debug._frontend import (
    DebugHandler,
    fetch_thread_pool,
    format_fetch_summary,
    NavLink,
    PeriodicDumper,
    Response,
    Route,
)
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    from torch.distributed.debug._frontend import fetch_aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


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


class _CountingHandler(DebugHandler):
    """Handler that counts dump() calls and signals after N calls."""

    def __init__(
        self,
        name: str = "counting",
        content: str | None = "data",
        *,
        notify_after: int = 1,
    ) -> None:
        self.dump_count = 0
        self._name = name
        self._content = content
        self._notify_after = notify_after
        self.ready = threading.Event()

    def routes(self) -> list[Route]:
        return []

    def nav_links(self) -> list[NavLink]:
        return []

    def dump(self) -> str | None:
        self.dump_count += 1
        if self.dump_count >= self._notify_after:
            self.ready.set()
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
            h = _CountingHandler("mystub", "hello world")
            dumper = PeriodicDumper([h], tmp, interval_seconds=60.0)
            dumper.start()
            h.ready.wait(timeout=5)
            dumper.stop()

            files = os.listdir(tmp)
            self.assertGreater(len(files), 0)
            self.assertTrue(all(f.startswith("mystub_") for f in files))
            with open(os.path.join(tmp, files[0])) as f:
                self.assertEqual(f.read(), "hello world")

    def test_skips_none_dump(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h = _CountingHandler("nodump", None)
            dumper = PeriodicDumper([h], tmp, interval_seconds=60.0)
            dumper.start()
            h.ready.wait(timeout=5)
            dumper.stop()

            self.assertEqual(os.listdir(tmp), [])

    def test_enabled_dumps_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h1 = _CountingHandler("included", "yes")
            h2 = _CountingHandler("excluded", "no")
            enabled = {"included"}
            filtered = [h for h in [h1, h2] if h.dump_filename() in enabled]
            dumper = PeriodicDumper(filtered, tmp, interval_seconds=60.0)
            dumper.start()
            h1.ready.wait(timeout=5)
            dumper.stop()

            files = os.listdir(tmp)
            self.assertGreater(len(files), 0)
            self.assertTrue(all(f.startswith("included_") for f in files))

    def test_enabled_dumps_none_runs_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h1 = _CountingHandler("alpha", "a")
            h2 = _CountingHandler("beta", "b")
            dumper = PeriodicDumper([h1, h2], tmp, interval_seconds=60.0)
            dumper.start()
            # Both handlers are called in the same cycle, so waiting on either
            # guarantees the cycle completed.
            h2.ready.wait(timeout=5)
            dumper.stop()

            files = os.listdir(tmp)
            alpha_files = [f for f in files if f.startswith("alpha_")]
            beta_files = [f for f in files if f.startswith("beta_")]
            self.assertGreater(len(alpha_files), 0)
            self.assertGreater(len(beta_files), 0)

    def test_survives_handler_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            err = _ErrorHandler()
            ok = _CountingHandler("ok", "survived")
            dumper = PeriodicDumper([err, ok], tmp, interval_seconds=60.0)
            dumper.start()
            ok.ready.wait(timeout=5)
            dumper.stop()

            files = os.listdir(tmp)
            ok_files = [f for f in files if f.startswith("ok_")]
            self.assertGreater(len(ok_files), 0)
            err_files = [f for f in files if f.startswith("error_handler_")]
            self.assertEqual(len(err_files), 0)

    def test_max_dumps_cleans_old_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h = _CountingHandler(notify_after=5)
            dumper = PeriodicDumper([h], tmp, interval_seconds=0, max_dumps=2)
            dumper.start()
            h.ready.wait(timeout=5)
            dumper.stop()
            files = [f for f in os.listdir(tmp) if f.startswith("counting_")]
            self.assertEqual(len(files), 2)
            self.assertGreaterEqual(h.dump_count, 5)

    def test_max_dumps_none_keeps_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h = _CountingHandler(notify_after=5)
            dumper = PeriodicDumper([h], tmp, interval_seconds=0, max_dumps=None)
            dumper.start()
            h.ready.wait(timeout=5)
            dumper.stop()
            files = [f for f in os.listdir(tmp) if f.startswith("counting_")]
            self.assertGreaterEqual(len(files), 5)

    def test_max_dumps_per_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h1 = _CountingHandler("aaa", "data1", notify_after=5)
            h2 = _CountingHandler("bbb", "data2", notify_after=5)
            dumper = PeriodicDumper([h1, h2], tmp, interval_seconds=0, max_dumps=2)
            dumper.start()
            h2.ready.wait(timeout=5)
            dumper.stop()
            aaa_files = [f for f in os.listdir(tmp) if f.startswith("aaa_")]
            bbb_files = [f for f in os.listdir(tmp) if f.startswith("bbb_")]
            self.assertEqual(len(aaa_files), 2)
            self.assertEqual(len(bbb_files), 2)

    def test_stop_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h = _StubHandler("idem", "data")
            dumper = PeriodicDumper([h], tmp, interval_seconds=60.0)
            dumper.start()
            dumper.stop()
            dumper.stop()


class TestFetchUnavailableWorkers(TestCase):
    @staticmethod
    def _get_refused_port() -> int:
        """Return a port that will refuse connections."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def test_fetch_thread_pool_connection_refused(self) -> None:
        port = self._get_refused_port()
        resps = fetch_thread_pool(
            [f"http://localhost:{port}/handler/ping"], timeout=1.0
        )
        self.assertEqual(len(resps), 1)
        self.assertEqual(resps[0].status_code, 503)
        self.assertIn("ConnectionError:", resps[0].text)

    def test_fetch_thread_pool_timeout(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            resps = fetch_thread_pool(
                [f"http://localhost:{port}/handler/ping"], timeout=0.5
            )
            self.assertEqual(len(resps), 1)
            self.assertEqual(resps[0].status_code, 408)
            self.assertIn("Timeout:", resps[0].text)
        finally:
            server.close()

    @unittest.skipUnless(HAS_AIOHTTP, "aiohttp not installed")
    def test_fetch_aiohttp_connection_refused(self) -> None:
        port = self._get_refused_port()
        resps = fetch_aiohttp([f"http://localhost:{port}/handler/ping"], timeout=1.0)
        self.assertEqual(len(resps), 1)
        self.assertEqual(resps[0].status_code, 503)
        self.assertIn("Error:", resps[0].text)

    @unittest.skipUnless(HAS_AIOHTTP, "aiohttp not installed")
    def test_fetch_aiohttp_timeout(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            resps = fetch_aiohttp(
                [f"http://localhost:{port}/handler/ping"], timeout=0.5
            )
            self.assertEqual(len(resps), 1)
            # aiohttp may report timeout as 408 or wrap it in a ClientError (503)
            self.assertIn(resps[0].status_code, (408, 503))
        finally:
            server.close()

    def test_mixed_available_and_unavailable(self) -> None:
        class _OKHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"pong")

            def log_message(self, *args):
                pass

        http_server = HTTPServer(("localhost", 0), _OKHandler)
        good_port = http_server.server_address[1]
        t = threading.Thread(target=http_server.serve_forever, daemon=True)
        t.start()

        bad_port = self._get_refused_port()
        try:
            urls = [
                f"http://localhost:{good_port}/handler/ping",
                f"http://localhost:{bad_port}/handler/ping",
            ]
            resps = fetch_thread_pool(urls, timeout=1.0)
            self.assertEqual(len(resps), 2)
            self.assertEqual(resps[0].status_code, 200)
            self.assertEqual(resps[0].text, "pong")
            self.assertNotEqual(resps[1].status_code, 200)
        finally:
            http_server.shutdown()


class TestFormatFetchSummary(TestCase):
    def test_all_success_returns_none(self) -> None:
        addrs = ["http://host0:1", "http://host1:1"]  # @lint-ignore
        resps = [Response(200, "ok"), Response(200, "ok")]
        self.assertIsNone(format_fetch_summary(addrs, resps))

    def test_partial_failure(self) -> None:
        addrs = ["http://h0:1", "http://h1:1", "http://h2:1"]  # @lint-ignore
        resps = [
            Response(200, "ok"),
            Response(503, "Worker unavailable"),
            Response(200, "ok"),
        ]
        summary = format_fetch_summary(addrs, resps)
        self.assertIsNotNone(summary)
        self.assertIn("PARTIAL DATA", summary)
        self.assertIn("2/3", summary)
        self.assertIn("Rank 1", summary)

    def test_all_failed(self) -> None:
        addrs = ["http://h0:1", "http://h1:1"]  # @lint-ignore
        resps = [Response(503, "unavailable"), Response(408, "timeout")]
        summary = format_fetch_summary(addrs, resps)
        self.assertIsNotNone(summary)
        self.assertIn("0/2", summary)

    def test_timeout_message_included(self) -> None:
        addrs = ["http://h0:1"]  # @lint-ignore
        resps = [Response(408, "Request timed out after 10.0s")]
        summary = format_fetch_summary(addrs, resps)
        self.assertIn("timed out", summary)


class TestFetchTimeout(TestCase):
    def test_handler_default_fetch_timeout(self) -> None:
        from torch.distributed.debug._frontend import _DEFAULT_FETCH_TIMEOUT

        handler = _StubHandler("test", "data")
        self.assertEqual(handler.fetch_timeout, _DEFAULT_FETCH_TIMEOUT)

    def test_handler_fetch_timeout_override(self) -> None:
        handler = _StubHandler("test", "data")
        handler.fetch_timeout = 5.0
        self.assertEqual(handler.fetch_timeout, 5.0)

    def test_fetch_all_requires_timeout(self) -> None:
        from torch.distributed.debug._frontend import fetch_all

        with self.assertRaises(TypeError):
            # fetch_all requires timeout as a keyword-only argument
            fetch_all("test_endpoint")  # type: ignore[call-arg]


class TestHandlerPartialDumps(TestCase):
    @patch("torch.distributed.debug._debug_handlers.fetch_all")
    def test_stacks_handler_partial_dump(self, mock_fetch_all) -> None:
        from torch.distributed.debug._debug_handlers import StacksHandler

        mock_fetch_all.return_value = (
            [
                "http://h0:1/handler/dump_traceback?",  # @lint-ignore
                "http://h1:1/handler/dump_traceback?",  # @lint-ignore
            ],
            [Response(200, "stack0"), Response(503, "Worker unavailable")],
        )
        handler = StacksHandler()
        content = handler.dump()
        self.assertIn("PARTIAL DATA", content)
        self.assertIn("1/2", content)
        self.assertIn("stack0", content)
        self.assertIn("Error: 503", content)

    @patch("torch.distributed.debug._debug_handlers.fetch_all")
    def test_stacks_handler_all_success(self, mock_fetch_all) -> None:
        from torch.distributed.debug._debug_handlers import StacksHandler

        mock_fetch_all.return_value = (
            [
                "http://h0:1/handler/dump_traceback?",  # @lint-ignore
                "http://h1:1/handler/dump_traceback?",  # @lint-ignore
            ],
            [Response(200, "stack0"), Response(200, "stack1")],
        )
        handler = StacksHandler()
        content = handler.dump()
        self.assertNotIn("PARTIAL", content)
        self.assertIn("stack0", content)
        self.assertIn("stack1", content)

    def test_periodic_dumper_writes_partial_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            partial_content = (
                "PARTIAL DATA: 1/2 workers responded\n"
                "  Rank 1: Worker unavailable\n"
                "\n"
                "=== Rank 0 ===\ndata0\n"
                "=== Rank 1 ===\nError: 503"
            )
            h = _CountingHandler("partial_stacks", partial_content)
            dumper = PeriodicDumper([h], tmp, interval_seconds=60.0)
            dumper.start()
            h.ready.wait(timeout=5)
            dumper.stop()

            files = os.listdir(tmp)
            self.assertGreater(len(files), 0)
            with open(os.path.join(tmp, files[0])) as f:
                content = f.read()
            self.assertIn("PARTIAL DATA", content)
            self.assertIn("1/2 workers responded", content)


if __name__ == "__main__":
    run_tests()
