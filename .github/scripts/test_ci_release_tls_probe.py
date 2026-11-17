# Owner(s): ["module: ci"]
import importlib.util
import os
import shutil
import ssl
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("ci_release_tls_probe.py")
SPEC = importlib.util.spec_from_file_location("probe", SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ReleaseTLSProbeTest(unittest.TestCase):
    def test_exact_route(self):
        good = {
            "GITHUB_REPOSITORY": probe.REPOSITORY,
            "GITHUB_REF": f"refs/pull/{probe.PR_NUMBER}/merge",
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_ACTOR": "d4l3k",
            "GITHUB_TRIGGERING_ACTOR": "d4l3k",
            "GITHUB_SHA": "a" * 40,
            "CANARY_PR_NUMBER": probe.PR_NUMBER,
            "CANARY_PR_AUTHOR": "d4l3k",
            "CANARY_HEAD_REPO": probe.REPOSITORY,
            "CANARY_HEAD_REF": probe.HEAD_REF,
            "CANARY_HEAD_SHA": "b" * 40,
        }
        probe.validate_route(good)
        for name in good:
            message = {
                "GITHUB_SHA": "unexpected workflow SHA",
                "CANARY_HEAD_SHA": "unexpected canary SHA",
            }.get(name, "unexpected workflow route")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                probe.validate_route({**good, name: "wrong"})
        for name in probe.FORBIDDEN_ENV:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "unexpected credential environment"),
            ):
                probe.validate_route({**good, name: "present"})

    def test_source_digest_required(self):
        for raw in (b"", b"curl --retry 3 -kL", b"modified source"):
            with (
                self.subTest(raw=raw),
                self.assertRaisesRegex(ValueError, "RC10 script digest mismatch"),
            ):
                probe.source_details(raw)

    def test_exact_http_target(self):
        header = b"CONNECT www.python.org:443 HTTP/1.1\r\nHost: ignored\r\n\r\n"
        self.assertTrue(probe.valid_request(header, "CONNECT", "www.python.org:443"))
        for target in (
            "example.com:443",
            "www.python.org:444",
            "www.python.org:443/path",
        ):
            self.assertFalse(probe.valid_request(header, "CONNECT", target))
        self.assertFalse(probe.valid_request(header, "GET", "www.python.org:443"))

    def test_child_environment(self):
        original = dict.fromkeys(probe.PROXY_ENV | probe.FORBIDDEN_ENV, "unwanted")
        original["PATH"] = "stock-tools"
        child = probe.child_environment(
            original, "http://127.0.0.1:1234", Path("marker")
        )
        self.assertEqual(child["PATH"], "stock-tools")
        self.assertEqual(child["HTTPS_PROXY"], "http://127.0.0.1:1234")
        self.assertEqual(child["https_proxy"], child["HTTPS_PROXY"])
        self.assertEqual(child["NO_PROXY"], "")
        self.assertEqual(child["DESIRED_PYTHON"], "3.13")
        for name in probe.FORBIDDEN_ENV:
            self.assertNotIn(name, child)
        for name in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "GIT_SSL_NO_VERIFY"):
            self.assertNotIn(name, child)

    def test_command_path(self):
        self.assertEqual(
            probe.cmd_path(r"C:\Program Files\tool.exe"),
            '"C:\\Program Files\\tool.exe"',
        )
        for char in '"%!&|<>^\r\n':
            with (
                self.subTest(char=char),
                self.assertRaisesRegex(ValueError, "unsupported command path"),
            ):
                probe.cmd_path("prefix" + char + "suffix")

    def test_completed_requests_wait(self):
        events = probe.RequestEvents()
        entered, released, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        results = []

        def waiter():
            entered.set()
            results.append(events.require(["payload-served"]))
            finished.set()

        def producer():
            released.wait()
            events.append("payload-served")

        waiting = threading.Thread(target=waiter)
        producing = threading.Thread(target=producer)
        waiting.start()
        producing.start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertFalse(finished.is_set())
        finally:
            released.set()
            waiting.join(20)
            producing.join(20)
        self.assertFalse(waiting.is_alive())
        self.assertFalse(producing.is_alive())
        self.assertEqual(results, [["payload-served"]])
        with self.assertRaisesRegex(ValueError, "unexpected completed request"):
            events.require(["wrong"])

    @unittest.skipIf(os.name == "nt", "the native Windows controller tests TLS itself")
    def test_real_tls_proxy(self):
        openssl, curl = shutil.which("openssl"), shutil.which("curl")
        self.assertIsNotNone(openssl)
        self.assertIsNotNone(curl)
        with tempfile.TemporaryDirectory(prefix="release-tls-test-") as temporary:
            root = Path(temporary)
            cert, key, output = root / "cert", root / "key", root / "output"
            subprocess.run(
                [
                    openssl,
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    str(key),
                    "-out",
                    str(cert),
                    "-days",
                    "1",
                    "-subj",
                    f"/CN={probe.HOST}",
                    "-addext",
                    f"subjectAltName=DNS:{probe.HOST}",
                ],
                check=True,
                capture_output=True,
            )
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
            with probe.local_proxy(context, "/exact-path", b"transport-control") as (
                proxy,
                events,
            ):
                env = probe.child_environment(
                    {"PATH": os.environ["PATH"]}, proxy, root / "marker"
                )
                argv = [
                    curl,
                    "--silent",
                    "--show-error",
                    f"https://{probe.HOST}/exact-path",
                    "--output",
                    str(output),
                ]
                negative = subprocess.run(
                    argv, env=env, capture_output=True, timeout=30
                )
                self.assertEqual(negative.returncode, 60)
                self.assertFalse(output.exists())
                positive = subprocess.run(
                    [*argv, "--insecure"], env=env, capture_output=True, timeout=30
                )
                self.assertEqual(positive.returncode, 0)
                self.assertEqual(output.read_bytes(), b"transport-control")
                self.assertEqual(
                    events.require(["tls-rejected", "payload-served"]),
                    ["tls-rejected", "payload-served"],
                )

    @unittest.skipIf(
        os.name == "nt", "the native Windows controller tests MSVC/PE itself"
    )
    def test_portable_marker(self):
        compiler = shutil.which("cc")
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory(prefix="release-marker-test-") as temporary:
            root = Path(temporary)
            exe, marker = root / "marker-program", root / "marker"
            subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    str(SOURCE.with_name("ci_release_tls_marker.c")),
                    "-o",
                    str(exe),
                ],
                check=True,
            )
            absent = subprocess.run([str(exe)], env={"PATH": os.environ["PATH"]})
            self.assertEqual(absent.returncode, 72)
            result = subprocess.run(
                [str(exe)],
                env={"PATH": os.environ["PATH"], "RLS05_MARKER": str(marker)},
            )
            self.assertEqual(result.returncode, 73)
            self.assertEqual(marker.read_bytes(), probe.MARKER)


if __name__ == "__main__":
    unittest.main()
