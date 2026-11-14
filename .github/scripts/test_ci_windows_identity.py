"""Offline controls for the actual Python3.11 Windows identity probe."""

from __future__ import annotations

import contextlib
import copy
import datetime
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
NOW = datetime.datetime(2026, 11, 13, 23, tzinfo=datetime.UTC)
SECRET = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
SESSION = "SYNTHETICSESSIONTOKEN"
ACCESS = "ASIA0123456789ABCDEF"
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
CREDS = {
    "Code": "Success",
    "Type": "AWS-HMAC",
    "AccessKeyId": ACCESS,
    "SecretAccessKey": SECRET,
    "Token": SESSION,
    "Expiration": "2026-11-14T00:00:00Z",
}
STS = f'<GetCallerIdentityResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><GetCallerIdentityResult><Account>{p.ACCOUNT}</Account><Arn>arn:aws:sts::{p.ACCOUNT}:assumed-role/{p.ROLE}/{ID}</Arn></GetCallerIdentityResult></GetCallerIdentityResponse>'
ITEM = f"<instanceId>{ID}</instanceId><placement><availabilityZone>us-east-1a</availabilityZone></placement><iamInstanceProfile><arn>{p.PROFILE}</arn></iamInstanceProfile><rootDeviceType>ebs</rootDeviceType><rootDeviceName>/dev/sda1</rootDeviceName><blockDeviceMapping><item><deviceName>/dev/sda1</deviceName><ebs><deleteOnTermination>true</deleteOnTermination></ebs></item></blockDeviceMapping><tagSet><item><key>Environment</key><value>gh-ci</value></item><item><key>InstanceManagement</key><value>dynamic</value></item></tagSet>"
EC2 = f'<DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/"><reservationSet><item><ownerId>{p.ACCOUNT}</ownerId><instancesSet><item>{ITEM}</item></instancesSet></item></reservationSet></DescribeInstancesResponse>'
BASE = {
    "token": b"SYNTHETICIMDSTOKEN",
    "document": json.dumps(DOC).encode(),
    "instance": ID.encode(),
    "region": p.REGION.encode(),
    "profile": json.dumps(INFO).encode(),
    "role": p.ROLE.encode(),
    "credentials": json.dumps(CREDS).encode(),
    "sts": STS.encode(),
    "ec2": EC2.encode(),
}
PATHS = {
    p.request_spec(op, token="placeholder")["path"]: op
    for op in (
        "token",
        "document",
        "instance",
        "region",
        "profile",
        "role",
        "credentials",
    )
}


def changed_json(name, **changes):
    original = {"document": DOC, "profile": INFO, "credentials": CREDS}[name]
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

        result = p.probe(runner, True, (env or {}).get, transport, NOW)
        self.assertEqual(set(result), set(p.KEYS))
        self.assertEqual(len(result), 20)
        self.assertTrue(all(type(v) is bool for v in result.values()))
        output = json.dumps(result)
        for value in (ID, p.PROFILE, SECRET, ACCESS, SESSION, "SYNTHETICIMDSTOKEN"):
            self.assertNotIn(value, output)
        return result, calls, requests

    def test_official_signing_vector(self):
        # botocore1.35.99 tests/unit/auth/aws4_testsuite/get-vanilla.
        headers = p.signed_headers(
            "GET",
            "example.amazonaws.com",
            "service",
            "us-east-1",
            b"",
            {},
            {"AccessKeyId": "AKIDEXAMPLE", "SecretAccessKey": SECRET},
            "20150830T123600Z",
        )
        self.assertEqual(
            headers["Authorization"],
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date, Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31",
        )

    def test_success(self):
        result, calls, requests = self.fixture()
        self.assertFalse(result["probe_error"])
        self.assertTrue(
            result["sts_authenticated"]
            and result["own_instance_identity_matched"]
            and result["own_root_delete_on_termination"]
        )
        self.assertEqual(
            calls,
            [
                "token",
                "document",
                "instance",
                "region",
                "profile",
                "role",
                "credentials",
                "sts",
                "ec2",
            ],
        )
        self.assertEqual(
            requests[-1]["body"],
            b"Action=DescribeInstances&Version=2016-11-15&InstanceId.1=" + ID.encode(),
        )

    def test_meta_only_order_and_negative_gates(self):
        self.assertEqual(p.ACCOUNT, "308535385114")
        self.assertEqual(p.REGION, "us-east-1")
        self.assertEqual(p.ROLE, "gh-ci-github-action-runners-runner-role")
        self.assertEqual(
            p.PROFILE,
            "arn:aws:iam::308535385114:instance-profile/gh-ci/gh-ci-github-action-runners-profile",
        )
        order = [
            "token",
            "document",
            "instance",
            "region",
            "profile",
            "role",
            "credentials",
            "sts",
            "ec2",
        ]
        cases = [
            (
                {"document": changed_json("document", accountId="391835788720")},
                "region",
            ),
            ({"instance": b"i-1123456789abcdef0"}, "region"),
            (
                {
                    "profile": changed_json(
                        "profile",
                        InstanceProfileArn="arn:aws:iam::391835788720:instance-profile/ghci-lf/ghci-lf-github-action-runners-profile",
                    )
                },
                "profile",
            ),
            (
                {
                    "profile": changed_json(
                        "profile",
                        InstanceProfileArn=p.PROFILE.replace(
                            "instance-profile/gh-ci/", "instance-profile/"
                        ),
                    )
                },
                "profile",
            ),
            ({"role": b"ghci-lf-github-action-runners-runner-role"}, "role"),
            ({"role": (p.ROLE + "\nother-role").encode()}, "role"),
            ({"sts": STS.replace(p.ACCOUNT, "391835788720").encode()}, "sts"),
            (
                {
                    "sts": STS.replace(
                        p.ROLE, "ghci-lf-github-action-runners-runner-role"
                    ).encode()
                },
                "sts",
            ),
            ({"sts": STS.replace(ID, "i-1123456789abcdef0").encode()}, "sts"),
        ]
        for changes, last in cases:
            with self.subTest(changes=changes, last=last):
                actual_call = p.call
                operations = {op: mock.Mock() for op in order}

                def observed_call(transport, operation, *args):
                    operations[operation]()
                    return actual_call(transport, operation, *args)

                with mock.patch.object(p, "call", side_effect=observed_call):
                    result, calls, _ = self.fixture(changes)
                self.assertTrue(result["probe_error"])
                stop = order.index(last) + 1
                self.assertEqual(calls, order[:stop])
                for op in order[:stop]:
                    operations[op].assert_called_once_with()
                for op in order[stop:]:
                    operations[op].assert_not_called()
        _, _, requests = self.fixture()
        self.assertEqual(
            requests[6]["path"],
            "/latest/meta-data/iam/security-credentials/gh-ci-github-action-runners-runner-role",
        )
        for request in requests:
            self.assertNotIn("ghci-lf", repr(request))
        result, calls, _ = self.fixture(
            {
                "ec2": EC2.replace(
                    "<value>gh-ci</value>", "<value>ghci-lf</value>"
                ).encode()
            }
        )
        self.assertFalse(result["probe_error"])
        self.assertFalse(result["own_instance_environment_matched"])
        self.assertTrue(result["own_instance_identity_matched"])
        self.assertEqual(calls, order)

    def test_post_signatures(self):
        # Independently generated by pinned botocore1.42.49 SigV4Auth.
        expected = {
            "sts": "1980b44dcd51380d3915d7e69da61bacb5afba8ac4f5d987ed74adeb8722fe92",
            "ec2": "34b3b6b63eeea4ed8cb5f6061775425ca844835cae192a08dde8f90d6afe760e",
        }
        for service, signature in expected.items():
            request = p.request_spec(service, ID, "", CREDS, "20261113T230000Z")
            self.assertTrue(request["headers"]["Authorization"].endswith(signature))

    def test_json_depth(self):
        with self.assertRaises(ValueError):
            p.json_object(b'{"deep":' + b"[" * 17 + b"0" + b"]" * 17 + b"}")

    def test_fail_closed(self):
        cases = [
            ("document", changed_json("document", accountId="391835788720"), "region"),
            ("document", changed_json("document", region="us-west-2"), "region"),
            (
                "document",
                BASE["document"].replace(
                    b'"accountId":', b'"accountId":"duplicate","accountId":'
                ),
                "document",
            ),
            ("document", changed_json("document", imageId="not-ami"), "region"),
            ("document", b'{"x":NaN}', "document"),
            ("instance", b"i-1123456789abcdef0", "region"),
            ("region", b"us-west-2", "region"),
            (
                "profile",
                changed_json(
                    "profile",
                    InstanceProfileArn=p.PROFILE.replace(
                        "instance-profile/gh-ci/", "instance-profile/"
                    ),
                ),
                "profile",
            ),
            ("role", (p.ROLE + "\nother-role").encode(), "role"),
            (
                "credentials",
                changed_json("credentials", Expiration="2026-11-12T00:00:00Z"),
                "credentials",
            ),
            (
                "credentials",
                changed_json("credentials", Expiration="2026-11-15T00:00:00Z"),
                "credentials",
            ),
            (
                "credentials",
                changed_json("credentials", AccessKeyId="AKIA0123456789ABCDEF"),
                "credentials",
            ),
            ("sts", STS.replace(ID, "i-1123456789abcdef0").encode(), "sts"),
            ("sts", ("<!DOCTYPE x [<!ENTITY p 'secret'>]>" + STS).encode(), "sts"),
            (
                "sts",
                STS.replace(
                    "</Account>", f"</Account><Account>{p.ACCOUNT}</Account>"
                ).encode(),
                "sts",
            ),
            (
                "ec2",
                EC2.replace(f"<ownerId>{p.ACCOUNT}", "<ownerId>391835788720").encode(),
                "ec2",
            ),
            ("ec2", EC2.replace(ID, "i-1123456789abcdef0").encode(), "ec2"),
            (
                "ec2",
                EC2.replace(
                    "</instancesSet>", f"<item>{ITEM}</item></instancesSet>"
                ).encode(),
                "ec2",
            ),
            (
                "ec2",
                EC2.replace(
                    "</DescribeInstancesResponse>",
                    "<nextToken>more</nextToken></DescribeInstancesResponse>",
                ).encode(),
                "ec2",
            ),
            (
                "ec2",
                EC2.replace(
                    "<deleteOnTermination>true", "<deleteOnTermination>unknown"
                ).encode(),
                "ec2",
            ),
            ("token", b"a" * 32769, "token"),
        ]
        for operation, value, last in cases:
            with self.subTest(operation=operation, last=last):
                result, calls, _ = self.fixture({operation: value})
                self.assertTrue(result["probe_error"])
                self.assertEqual(calls[-1], last)

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
        self.assertEqual(calls, [])

    def test_denials_redirect_and_missing_optional(self):
        result, calls, _ = self.fixture(statuses={"token": 302})
        self.assertTrue(result["probe_error"])
        self.assertEqual(calls, ["token"])
        for code in ("UnauthorizedOperation", "AccessDenied", "InvalidClientTokenId"):
            error = f"<Response><Errors><Error><Code>{code}</Code></Error></Errors></Response>".encode()
            result, _, _ = self.fixture({"ec2": error}, {"ec2": 403})
            self.assertEqual(result["probe_error"], code == "InvalidClientTokenId")
            self.assertEqual(
                result["own_instance_describe_denied"], code != "InvalidClientTokenId"
            )
        bare = EC2.replace(ITEM, ITEM.split("<rootDeviceType>")[0])
        result, _, _ = self.fixture({"ec2": bare.encode()})
        self.assertFalse(
            result["probe_error"]
            or result["own_root_device_ebs"]
            or result["own_instance_dynamic"]
        )

    def test_request_allowlist(self):
        _, _, requests = self.fixture()
        changes = [
            ("host", "ec2.us-west-2.amazonaws.com"),
            ("method", "GET"),
            ("path", "/other"),
            ("body", b"Action=DescribeInstances&Version=2016-11-15"),
            ("limit", 999999),
            ("tls", False),
        ]
        for name, value in changes:
            with self.subTest(name=name):
                request = copy.deepcopy(requests[-1])
                request[name] = value
                with self.assertRaises(ValueError):
                    p.validate_request(request)
        with self.assertRaises(ValueError):
            p.request_spec("iam")

    def test_runtime_and_import_failure(self):
        self.assertTrue(p.runtime_matches("nt", (3, 11, 9), p.PYTHON.lower()))
        for args in (
            ("posix", (3, 11), p.PYTHON),
            ("nt", (3, 12), p.PYTHON),
            ("nt", (3, 11), r"C:\other\python.exe"),
        ):
            self.assertFalse(p.runtime_matches(*args))

        def fail_import(name, **kwargs):
            if name == "ssl":
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
