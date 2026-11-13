"""Offline controls for the actual Python3.11 Windows identity probe."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "windows_probe", Path(__file__).with_name("ci_windows_identity.py")
)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)
p.load_stdlib()
ID = "i-0123456789abcdef0"
DOC = {
    "accountId": p.ACCOUNT,
    "region": p.REGION,
    "instanceId": ID,
    "availabilityZone": "us-east-1a",
    "imageId": "ami-0123456789abcdef0",
}
INFO = {
    "Code": "Success",
    "InstanceProfileArn": p.PROFILE,
    "InstanceProfileId": "AIPA0123456789ABCDEF0",
}
BASE = {
    "token": b"SYNTHETICIMDSTOKEN",
    "document": json.dumps(DOC).encode(),
    "instance": ID.encode(),
    "region": p.REGION.encode(),
    "profile": json.dumps(INFO).encode(),
    "role": p.ROLE.encode(),
}
PATHS = {p.request_spec(op, token="placeholder")["path"]: op for op in BASE}


def changed_json(name, **changes):
    original = {"document": DOC, "profile": INFO}[name]
    return json.dumps({**original, **changes}).encode()


class ProbeTests(unittest.TestCase):
    def fixture(self, changes=None, statuses=None, runner=ID, env=None):
        responses = {**BASE, **(changes or {})}
        calls, requests = [], []

        def transport(request):
            p.validate_request(request)
            op = (
                PATHS[request["path"]]
                if request["host"] == "169.254.169.254"
                else request["host"].split(".")[0]
            )
            calls.append(op)
            requests.append(copy.deepcopy(request))
            return (statuses or {}).get(op, 200), responses[op]

        result = p.probe(runner, True, (env or {}).get, transport)
        self.assertEqual(set(result), set(p.KEYS))
        self.assertEqual(len(result), 17)
        self.assertTrue(all(type(v) is bool for v in result.values()))
        output = json.dumps(result)
        for value in (ID, p.PROFILE, p.ACCOUNT, p.ROLE, "SYNTHETICIMDSTOKEN"):
            self.assertNotIn(value, output)
        return result, calls, requests

    def test_success_and_schema_parity(self):
        result, calls, requests = self.fixture()
        self.assertFalse(result["probe_error"])
        self.assertFalse(result["ambient_credential_inputs_present"])
        self.assertTrue(
            all(
                value
                for key, value in result.items()
                if key not in ("ambient_credential_inputs_present", "probe_error")
            )
        )
        self.assertEqual(calls, list(BASE))
        self.assertTrue(all(r["host"] == "169.254.169.254" for r in requests))
        self.assertEqual(p.ACCOUNT, "308535385114")
        self.assertEqual(p.ROLE, "gh-ci-github-action-runners-runner-role")
        self.assertEqual(
            p.PROFILE,
            "arn:aws:iam::308535385114:instance-profile/gh-ci/gh-ci-github-action-runners-profile",
        )
        directory = Path(__file__).parent
        source = (directory / "ci_windows_identity_launcher.cs").read_text()
        self.assertIn('("' + " ".join(p.KEYS) + '").Split', source)
        wrapper = (directory / "ci_windows_identity.ps1").read_text()
        fallback = json.loads(wrapper.split("$json = '", 1)[1].split("'\n", 1)[0])
        self.assertEqual(tuple(fallback), p.KEYS)
        self.assertEqual(fallback, dict.fromkeys(p.KEYS, False) | {"probe_error": True})

    def test_per_field_identity_diagnostics(self):
        cases = [
            (
                "document",
                "instanceId",
                "i-1123456789abcdef0",
                "document_instance_matches_runner",
            ),
            (
                "document",
                "accountId",
                "391835788720",
                "document_account_matches_expected",
            ),
            ("document", "region", "us-west-2", "document_region_matches_expected"),
            (
                "document",
                "availabilityZone",
                "us-west-2a",
                "availability_zone_matches_expected",
            ),
            ("document", "imageId", "not-ami", "image_id_format_valid"),
            ("profile", "Code", "Failure", "profile_metadata_success"),
            (
                "profile",
                "InstanceProfileArn",
                p.PROFILE.replace("instance-profile/gh-ci/", "instance-profile/"),
                "profile_arn_matches_expected",
            ),
            ("profile", "InstanceProfileId", "wrong", "profile_id_format_valid"),
        ]
        for operation, field, value, flag in cases:
            with self.subTest(flag=flag):
                result, calls, _ = self.fixture(
                    {operation: changed_json(operation, **{field: value})}
                )
                self.assertTrue(result["probe_error"])
                self.assertFalse(result[flag])
                self.assertEqual(
                    calls[-1], "region" if operation == "document" else "profile"
                )
                scope = p.KEYS[4:12] if operation == "document" else p.KEYS[12:15]
                self.assertTrue(all(result[k] for k in scope if k != flag))
        for operation, value, flag in (
            ("instance", b"i-1123456789abcdef0", "metadata_instance_matches_runner"),
            ("region", b"us-west-2", "metadata_region_matches_expected"),
            ("role", (p.ROLE + "\nother-role").encode(), "role_name_matches_expected"),
            (
                "role",
                b"ghci-lf-github-action-runners-runner-role",
                "role_name_matches_expected",
            ),
        ):
            with self.subTest(flag=flag):
                result, calls, _ = self.fixture({operation: value})
                self.assertTrue(result["probe_error"])
                self.assertFalse(result[flag])
                self.assertEqual(calls[-1], "role" if operation == "role" else "region")

    def test_json_and_status_fail_closed(self):
        with self.assertRaises(ValueError):
            p.json_object(b'{"deep":' + b"[" * 17 + b"0" + b"]" * 17 + b"}")
        for operation, body in (
            (
                "document",
                BASE["document"].replace(
                    b'"accountId":', b'"accountId":"duplicate","accountId":'
                ),
            ),
            ("document", b'{"x":NaN}'),
            ("profile", b"{}"),
            ("token", b"a" * 32769),
        ):
            result, calls, _ = self.fixture({operation: body})
            self.assertTrue(result["probe_error"])
            self.assertEqual(calls[-1], operation)
        for operation in BASE:
            for status in (301, 302, 403, 404, 500):
                with self.subTest(operation=operation, status=status):
                    result, calls, _ = self.fixture(statuses={operation: status})
                    self.assertTrue(result["probe_error"])
                    self.assertEqual(calls[-1], operation)

    def test_overrides_and_runner(self):
        for name in p.OVERRIDES:
            with self.subTest(name=name):
                result, calls, _ = self.fixture(env={name: "synthetic"})
                self.assertTrue(
                    result["probe_error"]
                    and result["ambient_credential_inputs_present"]
                )
                self.assertEqual(calls, [])
        result, calls, _ = self.fixture(runner="not-instance")
        self.assertTrue(result["probe_error"])
        self.assertFalse(result["runner_name_well_formed"])
        self.assertEqual(calls, [])

    def test_request_allowlist(self):
        _, _, requests = self.fixture()
        changes = [
            ("host", "sts.us-east-1.amazonaws.com"),
            ("host", "ec2.us-east-1.amazonaws.com"),
            ("host", "169.254.170.2"),
            ("method", "POST"),
            ("path", "/other"),
            ("path", "/latest/meta-data/iam/security-credentials/" + p.ROLE),
            ("body", b"Action=GetCallerIdentity&Version=2011-06-15"),
            ("limit", 999999),
            ("tls", True),
            ("headers", {"Authorization": "synthetic"}),
        ]
        for name, value in changes:
            with self.subTest(name=name, value=value):
                request = copy.deepcopy(requests[-1])
                request[name] = value
                with self.assertRaises(ValueError):
                    p.validate_request(request)
        for operation in ("credentials", "sts", "ec2", "iam", p.ROLE):
            with self.assertRaises(ValueError):
                p.request_spec(operation)
        for name in ("signed_headers", "credentials", "confirm_sts", "confirm_ec2"):
            self.assertFalse(hasattr(p, name))

    def test_runtime_and_import_failure(self):
        self.assertTrue(p.runtime_matches("nt", (3, 11, 9), p.PYTHON.lower()))
        for args in (
            ("posix", (3, 11), p.PYTHON),
            ("nt", (3, 12), p.PYTHON),
            ("nt", (3, 11), r"C:\other\python.exe"),
        ):
            self.assertFalse(p.runtime_matches(*args))

        with (
            mock.patch.object(p, "runtime_matches", return_value=False),
            mock.patch.object(
                p, "admin_token", side_effect=AssertionError("admin after mismatch")
            ),
            mock.patch.object(
                p, "http_request", side_effect=AssertionError("network after mismatch")
            ),
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(p.main(), 1)
            self.assertFalse(json.loads(output.getvalue())["runtime_matched"])
            self.assertFalse(json.loads(output.getvalue())["metadata_reachable"])

        def fail_import(name, **kwargs):
            if name == "http.client":
                raise ImportError("synthetic secret")
            return __import__(name, **kwargs)

        with self.assertRaises(ImportError):
            p.load_stdlib(fail_import)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = p.main(lambda: p.load_stdlib(fail_import))
        self.assertEqual(exit_code, 1)
        value = json.loads(output.getvalue())
        self.assertEqual(set(value), set(p.KEYS))
        self.assertTrue(value["probe_error"])
        self.assertNotIn("synthetic secret", output.getvalue())

    def test_isolated_entrypoint(self):
        # Invalid public runner binding prevents metadata even on the real AMI.
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-S",
                str(Path(__file__).with_name("ci_windows_identity.py")),
            ],
            env={**os.environ, "EXPECTED_RUNNER_NAME": "not-instance"},
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stderr, b"")
        value = json.loads(completed.stdout)
        self.assertEqual(set(value), set(p.KEYS))
        self.assertTrue(value["probe_error"])
        self.assertFalse(value["metadata_reachable"])

    def test_windows_membership_signatures(self):
        class Function:
            def __init__(self, implementation):
                self.implementation = implementation

            def __call__(self, *args):
                return self.implementation(*args)

        def create(kind, domain, sid, size):
            self.assertEqual(kind, 26)
            self.assertIsNone(domain)
            self.assertEqual(
                p.ctypes.cast(size, p.ctypes.POINTER(p.wintypes.DWORD))[0], 68
            )
            return 1

        def check(token, sid, member):
            self.assertIsNone(token)
            p.ctypes.cast(member, p.ctypes.POINTER(p.wintypes.BOOL))[0] = 1
            return 1

        api = type("Api", (), {})()
        api.CreateWellKnownSid, api.CheckTokenMembership = (
            Function(create),
            Function(check),
        )
        with mock.patch.object(
            p.ctypes, "WinDLL", return_value=api, create=True
        ) as loader:
            self.assertTrue(p.admin_token())
            loader.assert_called_once_with("advapi32.dll", use_last_error=True)
        self.assertEqual(len(api.CreateWellKnownSid.argtypes), 4)
        self.assertEqual(len(api.CheckTokenMembership.argtypes), 3)
        self.assertIs(api.CheckTokenMembership.restype, p.wintypes.BOOL)

    def test_http_bounded_stream_and_deadline(self):
        class Connection:
            def __init__(self, chunks, length=None, advance=None):
                self.chunks, self.length, self.advance = iter(chunks), length, advance
                self.sock, self.closed, self.timeouts, self.status = (
                    self,
                    False,
                    [],
                    200,
                )

            def set_debuglevel(self, level):
                self.debug = level

            def connect(self):
                pass

            def settimeout(self, timeout):
                self.timeouts.append(timeout)

            def request(self, *args, **kwargs):
                self.request_args = (args, kwargs)

            def getresponse(self):
                return self

            def getheader(self, name):
                self.header = name
                return self.length

            def isclosed(self):
                return self.closed

            def read1(self, size):
                if self.advance:
                    self.advance()
                data = next(self.chunks, b"")
                self.assert_size = size
                return data

            def close(self):
                self.closed = True

        request = p.request_spec("token")
        clock = [0]
        conn = Connection([b"one", b"two", b""])
        status, body = p.http_request(request, 120, lambda: clock[0], lambda r, t: conn)
        self.assertEqual((status, body), (200, b"onetwo"))
        self.assertTrue(conn.closed)
        self.assertEqual(conn.debug, 0)
        self.assertTrue(all(0 < value <= 10 for value in conn.timeouts))
        for connection in (
            Connection([], "32769"),
            Connection([b"a" * 32769]),
            Connection([], "1,2"),
        ):
            with self.subTest(length=connection.length), self.assertRaises(ValueError):
                p.http_request(request, 120, lambda: 0, lambda r, t: connection)
            self.assertTrue(connection.closed)
        conn = Connection(
            [b"a", b"b"], advance=lambda: clock.__setitem__(0, clock[0] + 70)
        )
        with self.assertRaises(ValueError):
            p.http_request(request, 120, lambda: clock[0], lambda r, t: conn)
        self.assertTrue(conn.closed)
        with self.assertRaises(ValueError):
            p.http_request(
                request,
                120,
                lambda: 121,
                lambda r, t: self.fail("connection after deadline"),
            )

    def real_response(self, wire, clock=None, response_class=None):
        client, server = socket.socketpair()
        server.settimeout(3)

        def respond():
            try:
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = server.recv(4096)
                    if not chunk:
                        return
                    request += chunk
                server.sendall(wire)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                server.close()

        class LocalConnection(p.http.HTTPConnection):
            def connect(self):
                self.sock = client

        if response_class is not None:
            LocalConnection.response_class = response_class
        peer = threading.Thread(target=respond, daemon=True)
        peer.start()
        try:
            now = clock or time.monotonic
            return p.http_request(
                p.request_spec("token"),
                now() + 120,
                now,
                lambda request, timeout: LocalConnection(
                    request["host"], timeout=timeout
                ),
            )
        finally:
            client.close()
            peer.join(4)
            self.assertFalse(peer.is_alive())

    def test_genuine_http_close_and_eof(self):
        cases = [
            (b"HTTP/1.0 200 OK\r\n\r\nx", b"x"),
            (b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 0\r\n\r\n", b""),
            (
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 1\r\n\r\nx",
                b"x",
            ),
            (b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nx", b"x"),
            (
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nx\r\n0\r\n\r\n",
                b"x",
            ),
        ]
        for wire, expected in cases:
            with self.subTest(wire=wire[:40]):
                self.assertEqual(self.real_response(wire), (200, expected))

    def test_genuine_http_bounds_and_deadline(self):
        oversize = (
            b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 32769\r\n\r\n"
        )
        with self.assertRaises(ValueError):
            self.real_response(oversize)
        short = b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 2\r\n\r\nx"
        with self.assertRaises(ValueError):
            self.real_response(short)
        chunks = (
            b"HTTP/1.1 200 OK\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n8001\r\n"
            + b"x" * 32769
            + b"\r\n0\r\n\r\n"
        )
        with self.assertRaises(ValueError):
            self.real_response(chunks)
        clock = [0]

        class DeadlineResponse(p.http.HTTPResponse):
            def read1(self, amount=-1):
                result = super().read1(amount)
                clock[0] += 130
                return result

        wire = b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 1\r\n\r\nx"
        with self.assertRaises(ValueError):
            self.real_response(wire, lambda: clock[0], DeadlineResponse)


if __name__ == "__main__":
    unittest.main()
