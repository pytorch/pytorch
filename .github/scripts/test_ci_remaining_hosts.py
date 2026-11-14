#!/usr/bin/env python3
# Owner(s): ["module: ci"]
"""Offline scope/transport/output controls; never contact a real Docker daemon."""

import contextlib
import io
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import ci_remaining_hosts as p


ENV = {
    "GITHUB_REPOSITORY": "pytorch/pytorch",
    "GITHUB_EVENT_NAME": "pull_request",
    "GITHUB_ACTOR": "ptdbot",
    "GITHUB_RUN_ID": "123",
    "GITHUB_RUN_ATTEMPT": "1",
    "AUDIT_HEAD_SHA": "a" * 40,
    "EXPECTED_LABEL": "linux.idc.xpu",
}
SECRET = "UNRELATED_DAEMON_FIELD_MUST_NOT_ESCAPE"
SOCK = SimpleNamespace(st_dev=1, st_ino=2, st_mode=stat.S_IFSOCK | 0o660, st_uid=0)


def wire(obj=None, *, status=200, extra=b"", body=None, chunked=False):
    if body is None:
        body = json.dumps(
            obj
            if obj is not None
            else {
                "SecurityOptions": ["name=seccomp,profile=builtin"],
                "Unrelated": SECRET,
            }
        ).encode()
    headers = (
        b"HTTP/1.1 "
        + str(status).encode()
        + b" status\r\nContent-Type: application/json\r\n"
        + extra
    )
    if chunked:
        return (
            headers
            + b"Transfer-Encoding: chunked\r\n\r\n"
            + format(len(body), "x").encode()
            + b"\r\n"
            + body
            + b"\r\n0\r\n\r\n"
        )
    return headers + b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body


class Socket:
    def __init__(self, data):
        self.data = data
        self.connected = []
        self.sent = []
        self.closed = False

    def settimeout(self, value):
        p.require(0 < value <= 3)

    def connect(self, path):
        self.connected.append(path)

    def sendall(self, data):
        self.sent.append(data)

    def makefile(self, mode):
        p.require(mode == "rb")
        return io.BytesIO(self.data)

    def close(self):
        self.closed = True


def request(data, before=SOCK, after=SOCK):
    sock = Socket(data)
    with (
        mock.patch.object(p.os, "lstat", side_effect=[before, after]),
        mock.patch.object(p.socket, "socket", return_value=sock) as ctor,
        mock.patch.dict(
            os.environ,
            {
                "DOCKER_HOST": "tcp://evil.example",
                "DOCKER_CONFIG": "/evil/plugins",
                "HTTP_PROXY": "http://evil.example",
            },
        ),
    ):
        result = p.docker_security(p.time.monotonic() + 60)
    unittest.TestCase().assertEqual(
        ctor.call_args.args, (p.socket.AF_UNIX, p.socket.SOCK_STREAM)
    )
    unittest.TestCase().assertEqual(sock.connected, [p.SOCKET])
    unittest.TestCase().assertEqual(sock.sent, [p.REQUEST])
    unittest.TestCase().assertTrue(sock.closed)
    return result


class WireTests(unittest.TestCase):
    def test_success_and_chunked(self):
        self.assertEqual(request(wire()), (False, False))
        self.assertEqual(
            request(
                wire(
                    {"SecurityOptions": ["name=rootless", "name=userns"]}, chunked=True
                )
            ),
            (True, True),
        )

    def test_final_symlink_blocks_connect(self):
        link = SimpleNamespace(
            st_dev=1, st_ino=2, st_mode=stat.S_IFLNK | 0o777, st_uid=0
        )
        with (
            mock.patch.object(p.os, "lstat", return_value=link),
            mock.patch.object(p.socket, "socket") as ctor,
            self.assertRaisesRegex(p.Boundary, "^refused$"),
        ):
            p.docker_security(p.time.monotonic() + 60)
        ctor.assert_not_called()

    def test_replacement_blocks_request(self):
        changed = SimpleNamespace(st_dev=1, st_ino=3, st_mode=SOCK.st_mode, st_uid=0)
        sock = Socket(wire())
        with (
            mock.patch.object(p.os, "lstat", side_effect=[SOCK, changed]),
            mock.patch.object(p.socket, "socket", return_value=sock),
            self.assertRaisesRegex(p.Boundary, "^refused$"),
        ):
            p.docker_security(p.time.monotonic() + 60)
        self.assertEqual(sock.sent, [])
        self.assertTrue(sock.closed)

    def test_bad_http(self):
        cases = [
            wire(status=302),
            wire(status=404),
            wire(extra=b"X-Large: " + b"x" * 16384 + b"\r\n"),
            wire(body=b"x" * (p.MAX_BODY + 1)),
            wire(body=b"{}", extra=b"Transfer-Encoding: gzip\r\n"),
            wire(body=b'{"SecurityOptions":[],"SecurityOptions":[]}'),
            wire({"SecurityOptions": "rootless"}),
            wire({"SecurityOptions": [3]}),
            wire({"SecurityOptions": ["x"] * 65}),
            wire({"SecurityOptions": ["x" * 257]}),
        ]
        for data in cases:
            with (
                self.subTest(size=len(data)),
                self.assertRaisesRegex(p.Boundary, "^refused$"),
            ):
                request(data)

    def test_missing_field_is_unknown(self):
        with self.assertRaisesRegex(p.Boundary, "^refused$"):
            request(wire({"Unrelated": SECRET}))

    def test_connect_failure_has_no_request(self):
        sock = Socket(wire())
        with (
            mock.patch.object(p.os, "lstat", return_value=SOCK),
            mock.patch.object(p.socket, "socket", return_value=sock),
            mock.patch.object(sock, "connect", side_effect=OSError(SECRET)),
            self.assertRaisesRegex(OSError, SECRET),
        ):
            p.docker_security(p.time.monotonic() + 60)
        self.assertEqual(sock.sent, [])
        self.assertTrue(sock.closed)


class ScopeTests(unittest.TestCase):
    def run_probe(self, env=ENV, machine="x86_64", docker=(False, False)):
        with (
            mock.patch.object(
                p.os,
                "uname",
                return_value=SimpleNamespace(sysname="Linux", machine=machine),
            ),
            mock.patch.object(p.os, "geteuid", return_value=1001),
            mock.patch.object(p.os.path, "exists", return_value=False),
            mock.patch.object(p, "docker_security", return_value=docker) as daemon,
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()) as error,
        ):
            result = p.probe(env)
        self.assertNotIn(
            SECRET, json.dumps(result) + output.getvalue() + error.getvalue()
        )
        self.assertTrue(all(type(value) is bool for value in result.values()))
        self.assertTrue(result["sudo_probe_not_attempted"])
        self.assertFalse(result["sudo_probe_known"])
        self.assertFalse(result["sudo_root_shell"])
        return result, daemon

    def test_four_labels(self):
        self.assertEqual(
            set(p.LABELS),
            {
                "linux.idc.xpu",
                "linux.rocm.gpu.mi300.1",
                "linux.dgx.b200.8",
                "linux.s390x",
            },
        )
        for label, architecture in p.LABELS.items():
            result, daemon = self.run_probe(
                {**ENV, "EXPECTED_LABEL": label}, architecture
            )
            self.assertTrue(result["completed"])
            self.assertFalse(result["probe_error"])
            self.assertTrue(result["docker_security_options_known"])
            daemon.assert_called_once()

    def test_bad_scope_no_commands(self):
        for key in ENV:
            result, daemon = self.run_probe({**ENV, key: "../bad"})
            self.assertTrue(result["probe_error"])
            daemon.assert_not_called()

    def test_architecture_stops_probes(self):
        result, daemon = self.run_probe(machine="aarch64")
        self.assertTrue(result["probe_error"])
        daemon.assert_not_called()

    def test_missing_socket_timeout_cancellation(self):
        for error in (
            FileNotFoundError(SECRET),
            TimeoutError(SECRET),
            p.Boundary(SECRET),
            KeyboardInterrupt(SECRET),
            p.StopProbe(SECRET),
        ):
            with (
                mock.patch.object(p, "docker_security", side_effect=error),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                result = p.probe(ENV)
            self.assertNotIn(SECRET, output.getvalue() + json.dumps(result))
            self.assertFalse(result["docker_security_options_known"])
            self.assertTrue(result["sudo_probe_not_attempted"])
            if isinstance(error, (KeyboardInterrupt, p.StopProbe)):
                self.assertTrue(result["probe_error"])
                self.assertFalse(result["completed"])

    def test_no_subprocess_or_sudo(self):
        source = Path(p.__file__).read_text()
        self.assertFalse(hasattr(p, "sudo_root"))
        for forbidden in (
            "import subprocess",
            "import selectors",
            "Popen(",
            "os.system(",
            "/usr/bin/sudo",
            "process.kill(",
        ):
            self.assertNotIn(forbidden, source)

    def test_deadline(self):
        with self.assertRaisesRegex(p.StopProbe, "^deadline$"):
            p.remaining(0, 3)

    def test_embedded_source_matches(self):
        here = Path(__file__).resolve()
        source = here.with_name("ci_remaining_hosts.py").read_text()
        workflow = here.parents[1] / "workflows/ci-remaining-hosts-canary.yml"
        text = workflow.read_text()
        marker = "          /usr/bin/python3 -I -B -S - <<'PY_REMAINING_HOSTS'\n"
        embedded = text.split(marker, 1)[1].split("          PY_REMAINING_HOSTS\n", 1)[
            0
        ]
        self.assertEqual(
            "\n".join(line[10:] if line else "" for line in embedded.splitlines())
            + "\n",
            source,
        )
        self.assertNotIn("uses:", text)
        self.assertIn('"collector_available":false', text)
        self.assertIn(
            '"sudo_probe_known":false,"sudo_probe_not_attempted":true,"sudo_root_shell":false',
            text,
        )
        self.assertIn(
            "shell: /bin/bash --noprofile --norc -p -e -o pipefail {0}\n", text
        )
        self.assertFalse(any(line and line.isspace() for line in text.splitlines()))
        for label in p.LABELS:
            self.assertEqual(text.count("- " + label + "\n"), 1)


class StartupTests(unittest.TestCase):
    def test_bash_env_and_imported_function(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "startup-marker"
            startup = root / "startup.sh"
            startup.write_text(
                "builtin printf '%s' env > " + shlex.quote(str(marker)) + "\n"
            )
            script = root / "script.sh"
            script.write_text("printf '%s\\n' fixed\n")
            env = {
                **os.environ,
                "BASH_ENV": str(startup),
                "BASH_FUNC_printf%%": "() { builtin printf '%s' function > "
                + shlex.quote(str(marker))
                + '; builtin printf "$@"; }',
            }
            plain = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", str(script)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            self.assertEqual(plain.returncode, 0)
            self.assertTrue(marker.exists())
            marker.unlink()
            function_env = dict(env)
            function_env.pop("BASH_ENV")
            imported = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", str(script)],
                env=function_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            self.assertEqual(imported.returncode, 0)
            self.assertEqual(imported.stdout, b"fixed\n")
            self.assertEqual(marker.read_text(), "function")
            marker.unlink()
            isolated = subprocess.run(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-p",
                    "-e",
                    "-o",
                    "pipefail",
                    str(script),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            self.assertEqual(isolated.returncode, 0)
            self.assertEqual(isolated.stdout, b"fixed\n")
            self.assertFalse(marker.exists())

    def test_python_site_and_path_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "startup-marker"
            (root / "sitecustomize.py").write_text(
                "from pathlib import Path\nPath("
                + repr(str(marker))
                + ").write_text('site')\n"
            )
            env = {
                **os.environ,
                "PYTHONPATH": str(root),
                "PYTHONSTARTUP": str(root / "sitecustomize.py"),
            }
            plain = subprocess.run(
                [sys.executable, "-B", "-"],
                input=b"print('fixed')\n",
                env=env,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            self.assertEqual(plain.returncode, 0)
            self.assertTrue(marker.exists())
            marker.unlink()
            isolated = subprocess.run(
                [sys.executable, "-I", "-B", "-S", "-"],
                input=b"print('fixed')\n",
                env=env,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            self.assertEqual(isolated.returncode, 0)
            self.assertEqual(isolated.stdout, b"fixed\n")
            self.assertFalse(marker.exists())
            self.assertFalse((root / "__pycache__").exists())


if __name__ == "__main__":
    unittest.main()
