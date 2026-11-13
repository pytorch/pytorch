#!/usr/bin/env python3
"""Offline fixed-scope controls for ci_tpu_identity; never contact metadata/cloud."""

import contextlib
import io
import json
import unittest
from unittest import mock

import ci_tpu_identity as p


VALUES = (
    "audit-project",
    "123456789",
    "987654321",
    "audit-vm",
    "projects/123456789/zones/us-central1-a",
    "audit@audit-project.iam.gserviceaccount.com",
    "https://www.googleapis.com/auth/cloud-platform",
)
ENV = {
    "GITHUB_REPOSITORY": "pytorch/pytorch",
    "GITHUB_EVENT_NAME": "pull_request",
    "GITHUB_ACTOR": "ptdbot",
    "GITHUB_RUN_ID": "123",
    "GITHUB_RUN_ATTEMPT": "2",
    "RUNNER_NAME": "audit-runner",
    "AUDIT_HEAD_SHA": "a" * 40,
}
TOKEN = "DO_NOT_LEAK_TEST_TOKEN_7c19"
IDENTITY = p.Identity.parse(VALUES)


def reply(code=200, obj=None, flavor="Google", content="application/json"):
    data = json.dumps(obj).encode() if obj is not None else b""
    return p.Reply(code, flavor, content, data)


def instance():
    prefix = f"https://www.googleapis.com/compute/v1/projects/{IDENTITY.project}/zones/{IDENTITY.zone}"
    return {
        "id": IDENTITY.instance_id,
        "name": IDENTITY.name,
        "zone": prefix,
        "selfLink": prefix + "/instances/" + IDENTITY.name,
        "serviceAccounts": [{"email": IDENTITY.email, "scopes": list(IDENTITY.scopes)}],
    }


class Fixture(p.Transport):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.values = list(VALUES)
        self.second_values = None
        self.metadata_count = 0
        self.control_status = 403
        self.unsigned_status = 401
        self.token_reply = reply(
            obj={"access_token": TOKEN, "token_type": "Bearer", "expires_in": 3599}
        )
        self.compute_reply = reply(obj=instance())
        self.permission_reply = {}
        self.failure = None

    def _perform(self, host, method, path, body, token, flavor, limit):
        self.calls.append((host, method, path, body, token, flavor, limit))
        if self.failure:
            raise self.failure
        if host == p.META:
            key = path.removeprefix(p.PREFIX)
            if not flavor:
                return reply(self.control_status)
            if key == p.TOKEN_KEY:
                return self.token_reply
            values = (
                self.second_values
                if self.second_values is not None
                and self.metadata_count >= len(p.META_KEYS)
                else self.values
            )
            self.metadata_count += 1
            return p.Reply(
                200, "Google", "text/plain", values[p.META_KEYS.index(key)].encode()
            )
        specs = self.identity.specs()
        route = next(k for k, spec in specs.items() if (host, method, path) == spec[:3])
        if route == "compute":
            return self.compute_reply if token else reply(self.unsigned_status)
        p.require(token == TOKEN)
        p.require(json.loads(body) == {"permissions": list(p.PERMISSIONS[route])})
        return self.permission_reply.get(
            route, reply(obj={"permissions": list(p.PERMISSIONS[route])})
        )


def run(client, env=None):
    with (
        mock.patch.object(p.resource, "setrlimit") as limit,
        contextlib.redirect_stdout(io.StringIO()) as out,
        contextlib.redirect_stderr(io.StringIO()) as err,
    ):
        result = p.probe(client, ENV if env is None else env)
    wire = json.dumps(result)
    p.require(TOKEN not in wire + out.getvalue() + err.getvalue())
    p.require(VALUES[0] not in wire)
    p.require(result["cleanup_confirmed"])
    return result, limit.call_count


def tokens(client):
    return [x for x in client.calls if x[2] == p.PREFIX + p.TOKEN_KEY]


def posts(client):
    return [x for x in client.calls if x[1] == "POST"]


class ProbeTests(unittest.TestCase):
    def test_full_path(self):
        f = Fixture()
        result, limits = run(f)
        self.assertEqual(limits, 1)
        self.assertFalse(result["probe_error"])
        self.assertTrue(result["instance_identity_matches"])
        self.assertEqual(len(tokens(f)), 1)
        self.assertEqual(len(posts(f)), 3)
        self.assertTrue(
            all(x["decision_received"] for x in result["permissions"].values())
        )
        self.assertFalse(result["metadata_server_cryptographically_authenticated"])

    def test_context(self):
        for key in ENV:
            f = Fixture()
            env = {**ENV, key: "../invalid"}
            result, _ = run(f, env)
            self.assertTrue(result["probe_error"])
            self.assertEqual(f.calls, [])

    def test_metadata_control(self):
        for code in (200, 404, 500):
            f = Fixture()
            f.control_status = code
            result, _ = run(f)
            self.assertTrue(result["probe_error"])
            self.assertEqual(tokens(f), [])

    def test_invalid_identity(self):
        invalid = (
            (0, "../project"),
            (1, "0"),
            (2, str(2**64)),
            (3, "bad/name"),
            (4, "projects/999/zones/us-central1-a"),
            (5, "x@example.com/evil"),
            (6, "https://evil.example/scope"),
        )
        for index, value in invalid:
            f = Fixture()
            f.values[index] = value
            result, _ = run(f)
            self.assertTrue(result["probe_error"])
            self.assertEqual(tokens(f), [])

    def test_identity_changes(self):
        f = Fixture()
        f.second_values = list(VALUES)
        f.second_values[2] = "1234"
        result, _ = run(f)
        self.assertTrue(result["probe_error"])
        self.assertEqual(tokens(f), [])

    def test_unsigned_control(self):
        for code in (200, 404, 500):
            f = Fixture()
            f.unsigned_status = code
            result, _ = run(f)
            self.assertTrue(result["probe_error"])
            self.assertEqual(tokens(f), [])

    def test_token_schema(self):
        good = {"access_token": TOKEN, "token_type": "Bearer", "expires_in": 3599}
        changes = (
            {"token_type": "Basic"},
            {"expires_in": True},
            {"expires_in": 0},
            {"expires_in": 86401},
            {"refresh_token": TOKEN},
            {"access_token": "bad\nvalue"},
            {"access_token": "x" * 8193},
        )
        for change in changes:
            f = Fixture()
            f.token_reply = reply(obj={**good, **change})
            result, _ = run(f)
            self.assertTrue(result["probe_error"])
            self.assertEqual(posts(f), [])

    def test_compute_denied(self):
        for code, expected in (
            (403, "access_denied"),
            (404, "not_found"),
            (429, "rate_limited"),
        ):
            f = Fixture()
            f.compute_reply = reply(code)
            result, _ = run(f)
            self.assertEqual(result["compute_status"], expected)
            self.assertFalse(result["instance_identity_matches"])
            self.assertEqual(posts(f), [])

    def test_compute_identity(self):
        changes = (
            {"id": "3"},
            {"name": "other"},
            {"zone": "https://evil.example/"},
            {
                "selfLink": "https://www.googleapis.com/compute/v1/projects/other/zones/us-central1-a/instances/audit-vm"
            },
            {"serviceAccounts": []},
            {
                "serviceAccounts": [
                    {
                        "email": "other@audit-project.iam.gserviceaccount.com",
                        "scopes": list(IDENTITY.scopes),
                    }
                ]
            },
            {"serviceAccounts": [{"email": IDENTITY.email, "scopes": []}]},
        )
        for change in changes:
            f = Fixture()
            f.compute_reply = reply(obj={**instance(), **change})
            result, _ = run(f)
            self.assertTrue(result["probe_error"])
            self.assertEqual(posts(f), [])

    def test_permission_denials(self):
        f = Fixture()
        f.permission_reply = {
            "project": reply(
                403, {"error": {"details": [{"reason": "SERVICE_DISABLED"}]}}
            ),
            "instance": reply(429),
            "service_account": reply(404),
        }
        result, _ = run(f)
        for route, expected in (
            ("project", "service_disabled"),
            ("instance", "rate_limited"),
            ("service_account", "not_found"),
        ):
            self.assertEqual(result["permissions"][route]["status"], expected)
            self.assertFalse(result["permissions"][route]["decision_received"])
        self.assertFalse(result["probe_error"])
        self.assertEqual(len(posts(f)), 3)

    def test_permission_schema(self):
        permission = p.PERMISSIONS["project"][0]
        for obj in (
            {"permissions": [permission, permission]},
            {"permissions": ["evil.operation"]},
            {"permissions": [3]},
            {"permissions": "bad"},
            {"unexpected": []},
        ):
            f = Fixture()
            f.permission_reply["project"] = reply(obj=obj)
            result, _ = run(f)
            self.assertTrue(result["probe_error"])
            self.assertEqual(len(posts(f)), 1)
            self.assertFalse(result["permissions"]["project"]["decision_received"])

    def test_empty_permission_subset(self):
        f = Fixture()
        f.permission_reply["project"] = reply(obj={})
        result, _ = run(f)
        self.assertTrue(result["permissions"]["project"]["decision_received"])
        self.assertFalse(any(result["permissions"]["project"]["allowed"].values()))

    def test_duplicate_json(self):
        with self.assertRaises(p.Boundary):
            p.parse_json(b'{"permissions":[],"permissions":[]}')

    def test_cancellation_and_error_leakage(self):
        for error in (p.Boundary(TOKEN), OSError(TOKEN), KeyboardInterrupt(TOKEN)):
            f = Fixture()
            f.failure = error
            result, _ = run(f)
            self.assertTrue(result["probe_error"])

    def test_exact_routes(self):
        f = Fixture()
        for key in (
            "instance/attributes",
            "instance/service-accounts/",
            "instance/service-accounts/default/identity?audience=x",
            "../token",
        ):
            with self.assertRaises(p.Boundary):
                f.metadata(key)
        with self.assertRaises(p.Boundary):
            f.api("compute", TOKEN)
        with self.assertRaises(p.Boundary):
            f.bind(p.dataclasses.replace(IDENTITY, project="../other"))
        f.bind(IDENTITY)
        with self.assertRaises(p.Boundary):
            f.bind(IDENTITY)
        with self.assertRaises(p.Boundary):
            f.api("list", TOKEN)
        with self.assertRaises(p.Boundary):
            f.api("project")
        self.assertEqual(f.calls, [])


class WireTests(unittest.TestCase):
    def test_peer_redirect_size_and_proxy(self):
        for peer, code, data, ok in (
            (p.META, 200, b"7", True),
            ("127.0.0.1", 200, b"7", False),
            (p.META, 302, b"", False),
            (p.META, 200, b"x" * 2049, False),
        ):
            conn = mock.MagicMock()
            conn.sock.getpeername.return_value = (peer, 80)
            response = conn.getresponse.return_value
            response.status = code
            response.read.return_value = data
            response.getheader.side_effect = lambda key, default="": {
                "Metadata-Flavor": "Google",
                "Content-Type": "text/plain",
            }.get(key, default)
            with (
                mock.patch.dict(
                    p.os.environ,
                    {
                        "HTTP_PROXY": "http://evil.example",
                        "HTTPS_PROXY": "http://evil.example",
                    },
                ),
                mock.patch.object(
                    p.http.client, "HTTPConnection", return_value=conn
                ) as constructor,
            ):
                transport = p.Transport()
                if ok:
                    self.assertEqual(transport.metadata("instance/id").data, b"7")
                    self.assertEqual(constructor.call_args.args, (p.META,))
                    self.assertEqual(
                        conn.request.call_args.args[:2],
                        ("GET", p.PREFIX + "instance/id"),
                    )
                    self.assertEqual(
                        conn.request.call_args.kwargs["headers"],
                        {
                            "Accept": "text/plain, application/json",
                            "Metadata-Flavor": "Google",
                        },
                    )
                else:
                    with self.assertRaises(p.Boundary):
                        transport.metadata("instance/id")
                conn.close.assert_called_once()

    def test_https_fixed_trust_and_headers(self):
        conn = mock.MagicMock()
        response = conn.getresponse.return_value
        response.status = 401
        response.read.return_value = b"{}"
        response.getheader.side_effect = lambda key, default="": {
            "Content-Type": "application/json"
        }.get(key, default)
        context = mock.MagicMock()
        paths = mock.Mock(
            openssl_cafile="/system/ca.pem", openssl_capath="/system/certs"
        )
        with (
            mock.patch.dict(
                p.os.environ,
                {"HTTPS_PROXY": "http://evil.example", "SSL_CERT_FILE": "/evil/ca.pem"},
            ),
            mock.patch.object(p.ssl, "get_default_verify_paths", return_value=paths),
            mock.patch.object(p.ssl, "SSLContext", return_value=context),
            mock.patch.object(
                p.http.client, "HTTPSConnection", return_value=conn
            ) as constructor,
        ):
            transport = p.Transport()
            transport.bind(IDENTITY)
            transport.api("compute")
            context.load_verify_locations.assert_called_once_with(
                cafile="/system/ca.pem", capath="/system/certs"
            )
            self.assertEqual(constructor.call_args.args, ("compute.googleapis.com",))
            self.assertNotIn("Authorization", conn.request.call_args.kwargs["headers"])
            self.assertNotIn("Cookie", conn.request.call_args.kwargs["headers"])
            self.assertEqual(
                conn.request.call_args.args[:2], IDENTITY.specs()["compute"][1:3]
            )
            conn.close.assert_called_once()

    def test_wrong_headers_and_limits(self):
        for response in (
            p.Reply(200, "wrong", "text/plain", b"123"),
            p.Reply(200, "Google", "text/html", b"123"),
            p.Reply(302, "Google", "text/plain", b"123"),
        ):
            with self.assertRaises(p.Boundary):
                p.text(response)
        transport = p.Transport()
        transport.deadline = 0
        with self.assertRaises(p.Boundary):
            transport.remaining(3)


if __name__ == "__main__":
    unittest.main()
