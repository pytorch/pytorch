# Owner(s): ["module: ci"]
"""Offline controls. Synthetic responses are not service/deployment evidence."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import socket
import ssl
import unittest
from pathlib import Path
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "probe", Path(__file__).with_name("ci_hf_cache_metadata_tail_probe.py")
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
CREDS = ("ASIA" + "A" * 16, "secret" * 5, "token" * 8)


def s3(name, body=""):
    return f'<{name} xmlns="{probe.S3_NS[1:-1]}">{body}</{name}>'.encode()


def identity(account=probe.ACCOUNT, role=probe.ROLE):
    return f'<GetCallerIdentityResponse xmlns="{probe.STS_NS[1:-1]}"><GetCallerIdentityResult><Account>{account}</Account><Arn>arn:aws:sts::{account}:assumed-role/{role}/ci-test</Arn></GetCallerIdentityResult></GetCallerIdentityResponse>'.encode()


def error(code="AccessDenied", status=403):
    return (
        status,
        {},
        f"<Error><Code>{code}</Code><Message>NEVER_PRINT</Message></Error>".encode(),
    )


def list_response(versions=False, rows="", count=0):
    body = f"<Name>{probe.BUCKET}</Name><Prefix>{probe.OBJECT_NAME}</Prefix><MaxKeys>8</MaxKeys><IsTruncated>false</IsTruncated>"
    if not versions:
        body += f"<KeyCount>{count}</KeyCount>"
    return s3("ListVersionsResult" if versions else "ListBucketResult", body + rows)


def responses():
    return {
        "identity": (200, {}, identity()),
        "HeadBucket": (200, {"x-amz-bucket-region": probe.REGION}, b""),
        "GetBucketLocation": (200, {}, s3("LocationConstraint")),
        "GetBucketVersioning": (200, {}, s3("VersioningConfiguration")),
        "GetObjectLockConfiguration": error(
            "ObjectLockConfigurationNotFoundError", 404
        ),
        "GetBucketReplication": error("ReplicationConfigurationNotFoundError", 404),
        "GetBucketNotificationConfiguration": (
            200,
            {},
            s3("NotificationConfiguration"),
        ),
        "GetBucketLifecycleConfiguration": error("NoSuchLifecycleConfiguration", 404),
        "GetBucketLogging": (200, {}, s3("BucketLoggingStatus")),
        "GetBucketPolicy": (
            200,
            {},
            b'{"Version":"2012-10-17","Statement":[{"Sid":"NEVER_PRINT","Effect":"Deny"}]}',
        ),
        "GetBucketPolicyStatus": (
            200,
            {},
            s3("PolicyStatus", "<IsPublic>false</IsPublic>"),
        ),
        "GetPublicAccessBlock": (
            200,
            {},
            s3(
                "PublicAccessBlockConfiguration",
                "".join(
                    f"<{x}>true</{x}>"
                    for x in (
                        "BlockPublicAcls",
                        "IgnorePublicAcls",
                        "BlockPublicPolicy",
                        "RestrictPublicBuckets",
                    )
                ),
            ),
        ),
        "GetBucketOwnershipControls": (
            200,
            {},
            s3(
                "OwnershipControls",
                "<Rule><ObjectOwnership>BucketOwnerEnforced</ObjectOwnership></Rule>",
            ),
        ),
        "GetBucketEncryption": (
            200,
            {},
            s3(
                "ServerSideEncryptionConfiguration",
                "<Rule><ApplyServerSideEncryptionByDefault><SSEAlgorithm>AES256</SSEAlgorithm></ApplyServerSideEncryptionByDefault></Rule>",
            ),
        ),
        "HeadObject": (404, {}, b""),
        "ListObjectsV2": (200, {}, list_response()),
        "ListObjectVersions": (200, {}, list_response(True)),
    }


def environment():
    return dict(
        GITHUB_REPOSITORY="pytorch/pytorch",
        GITHUB_REPOSITORY_ID="65600975",
        GITHUB_EVENT_NAME="push",
        GITHUB_REF=probe.TAG,
        GITHUB_ACTOR=probe.BOT,
        GITHUB_ACTOR_ID="54816060",
        GITHUB_TRIGGERING_ACTOR=probe.BOT,
        GITHUB_RUN_ATTEMPT="1",
        GITHUB_SHA="a" * 40,
        GITHUB_WORKFLOW_SHA="a" * 40,
        GITHUB_WORKFLOW_REF=f"pytorch/pytorch/{probe.WORKFLOW}@{probe.TAG}",
        RUNNER_ENVIRONMENT="github-hosted",
        RUNNER_OS="Linux",
        CANARY_SENDER_ID="54816060",
        CANARY_GITHUB_TOKEN="NONSECRET-FIXTURE",
        CANARY_ROLE_NAME=probe.ROLE,
        ACTIONS_ID_TOKEN_REQUEST_URL="https://fixture.invalid/never-called",
        ACTIONS_ID_TOKEN_REQUEST_TOKEN="NONSECRET-FIXTURE",
        AWS_EC2_METADATA_DISABLED="true",
        AWS_MAX_ATTEMPTS="1",
        AWS_CONFIG_FILE="/dev/null",
        AWS_SHARED_CREDENTIALS_FILE="/dev/null",
    )


@contextlib.contextmanager
def activated_scope():
    """Offline-only numbered source fixture; never edits the workflow."""
    with mock.patch.multiple(
        probe,
        PR_NUMBER="123",
        HEAD_REF="gh/d4l3k/456/head",
        BASE_REF="gh/d4l3k/456/base",
        TAG="refs/tags/ciflow/inductor-cu126/123",
    ):
        yield


def source():
    repo = {"full_name": "pytorch/pytorch", "id": 65600975}
    return {
        "number": int(probe.PR_NUMBER),
        "state": "open",
        "draft": True,
        "user": {"login": "d4l3k", "id": 909104},
        "head": {"repo": repo.copy(), "ref": probe.HEAD_REF, "sha": "a" * 40},
        "base": {"repo": repo.copy(), "ref": probe.BASE_REF},
    }


def wire(status=200, headers=(), body=b"", version=b"HTTP/1.1"):
    return (
        version
        + b" "
        + str(status).encode()
        + b" Test\r\n"
        + b"".join(
            name.encode() + b": " + value.encode() + b"\r\n" for name, value in headers
        )
        + b"\r\n"
        + body
    )


def framed(operation, row):
    status, headers, body = row
    fields = list(headers.items())
    if "content-length" not in headers:
        fields.append(("Content-Length", str(len(body))))
    method = probe.request_spec(operation)[2]
    return probe.read_wire(io.BytesIO(wire(status, fields, body)), method)


class FakeTransport:
    def __init__(self, overrides=None, presend=None):
        self.values = responses() | (overrides or {})
        self.counts = dict.fromkeys(("identity", *probe.OPERATIONS), 0)
        self.calls = []
        self.presend = presend
        self.cleanup_state = {"failed": False}

    def request(self, operation):
        if operation == self.presend:
            raise probe.GlobalFatalFailure("request-boundary", "request")
        if self.counts[operation]:
            raise AssertionError("repeated operation")
        self.calls.append(operation)
        self.counts[operation] += 1
        value = self.values[operation]
        if isinstance(value, BaseException):
            raise value
        return framed(operation, value)


def run(overrides=None, presend=None, cleanup=True):
    transport = FakeTransport(overrides, presend)
    out = probe.collect(transport, probe.result_skeleton())
    return probe.complete_result(out, cleanup), transport


def run_wire(overrides=None):
    """Keep the real dispatcher, signer, counters, parser and collector."""
    values = {}
    request_ids = {}
    for operation in ("identity", *probe.OPERATIONS):
        status, headers, body = responses()[operation]
        fields = list(headers.items())
        if "content-length" not in headers:
            fields.append(("Content-Length", str(len(body))))
        values[operation] = wire(status, fields, body)
        _, host, method, path, query, _, _ = probe.request_spec(operation)
        request_ids[(host, method, path + ("?" + query if query else ""))] = operation
    values.update(overrides or {})
    calls = []

    def offline_https(host, method, target, body, headers, cleanup_state=None):
        operation = request_ids[(host, method, target)]
        calls.append(operation)
        if body != probe.request_spec(operation)[-1] or "authorization" not in headers:
            raise AssertionError("request boundary")
        if (
            operation != "identity"
            and headers.get("x-amz-expected-bucket-owner") != probe.ACCOUNT
        ):
            raise AssertionError("owner boundary")
        return probe.read_wire(io.BytesIO(values[operation]), method)

    with mock.patch.object(probe, "https_request", offline_https):
        transport = probe.Transport(CREDS)
        result = probe.collect(transport, probe.result_skeleton())
        return probe.complete_result(result, True), calls


class ShortStream(io.BytesIO):
    def read(self, count=-1):
        return super().read(min(count, 2) if count >= 0 else 2)


class FailingBody(io.BytesIO):
    def __init__(self, data, error):
        super().__init__(data)
        self.error = error

    def read(self, count=-1):
        raise self.error


def independent_witnesses():
    EMPTY = probe.EMPTY_HASH
    NAMES = list(probe.OPERATIONS)

    def envelope(status=200, head=False):
        return {
            "http_status": status,
            "framing": "head-no-body" if head else "content-length",
            "response_complete": True,
            "response_bytes": 0,
            "response_sha256": EMPTY,
            "request_id": None,
            "body_limit_exceeded": False,
        }

    def idle():
        return {
            "disposition": "not-attempted",
            "response": None,
            "projection": None,
            "failure": None,
        }

    def record(name, p, status=200):
        return {
            "disposition": "projected",
            "response": envelope(status, name in {"HeadBucket", "HeadObject"}),
            "projection": p,
            "failure": None,
        }

    P = {
        "HeadBucket": {"state": "owner-region-matched"},
        "GetBucketLocation": {"state": "region-matched"},
        "GetBucketPolicy": {"state": "not-configured"},
        "GetBucketPolicyStatus": {"state": "present", "public": False},
        "GetPublicAccessBlock": {
            "state": "present",
            "block_public_acls": True,
            "ignore_public_acls": True,
            "block_public_policy": True,
            "restrict_public_buckets": True,
            "all_enabled": True,
        },
        "GetBucketOwnershipControls": {"state": "BucketOwnerEnforced"},
        "GetBucketEncryption": {"state": "configured", "algorithms": ["AES256"]},
        "HeadObject": {"state": "current-key-not-found"},
        "ListObjectsV2": {
            "state": "complete",
            "exact_entries": 0,
            "other_prefix_entries": 0,
        },
        "ListObjectVersions": {
            "state": "complete",
            "exact_entries": 0,
            "exact_current_versions": 0,
            "exact_noncurrent_versions": 0,
            "exact_delete_markers": 0,
            "exact_current_delete_markers": 0,
            "other_prefix_entries": 0,
        },
    }
    BASE = {
        "schema": 1,
        "identity_matches": True,
        "owner_region_matches": True,
        "operations": {
            n: record(
                n,
                copy.deepcopy(P[n]),
                404 if n in {"GetBucketPolicy", "HeadObject"} else 200,
            )
            for n in NAMES
        },
        "request_counts": dict.fromkeys(["identity", *NAMES], 1),
        "all_operations_attempted": True,
        "all_responses_projected": True,
        "probe_error": False,
        "fatal_stage": None,
        "fatal_code": None,
        "write_authorized": False,
        "collector_cleanup_confirmed": True,
    }

    def edit(fn):
        d = copy.deepcopy(BASE)
        fn(d)
        return d

    def truncate_at(d, name, count=1, code="network", inner="transport"):
        d.update(
            probe_error=True,
            fatal_stage=name,
            fatal_code=code,
            all_operations_attempted=False,
            all_responses_projected=False,
        )
        i = NAMES.index(name)
        d["operations"][name] = {
            "disposition": "fatal",
            "response": None,
            "projection": None,
            "failure": {"class": "global-fatal", "stage": inner, "code": code},
        }
        d["request_counts"][name] = count
        for n in NAMES[i + 1 :]:
            d["request_counts"][n] = 0
            d["operations"][n] = idle()
        if i < 2:
            d["owner_region_matches"] = False
        return d

    witnesses = {}
    for n in (
        "HeadBucket",
        "GetBucketLocation",
        "GetBucketPolicyStatus",
        "GetPublicAccessBlock",
        "GetBucketOwnershipControls",
        "GetBucketEncryption",
        "ListObjectsV2",
        "ListObjectVersions",
    ):
        witnesses["success-on-500-" + n] = edit(
            lambda d, n=n: d["operations"][n]["response"].update(http_status=500)
        )
    witnesses["status-no-body-on-200"] = edit(
        lambda d: d["operations"]["GetBucketEncryption"]["response"].update(
            framing="status-no-body"
        )
    )
    witnesses["access-denied-on-500"] = edit(
        lambda d: d["operations"].__setitem__(
            "GetBucketPolicy",
            record(
                "GetBucketPolicy",
                {"state": "unavailable", "reason": "access-denied"},
                500,
            ),
        )
    )
    d = truncate_at(copy.deepcopy(BASE), "GetBucketEncryption")
    d["fatal_code"] = "authentication"
    witnesses["root-operation-fatal-code-disagreement"] = d
    d = copy.deepcopy(BASE)
    d.update(
        identity_matches=False,
        owner_region_matches=False,
        all_operations_attempted=False,
        all_responses_projected=False,
    )
    d["operations"] = {n: idle() for n in NAMES}
    d["request_counts"] = dict.fromkeys(["identity", *NAMES], 0)
    witnesses["unfinished-nonfatal-zero-skeleton"] = d
    witnesses["identity-fatal-after-all-S3"] = edit(
        lambda d: d.update(
            probe_error=True, fatal_stage="identity", fatal_code="identity"
        )
    )
    witnesses["finalization-failed-but-cleanup-confirmed"] = edit(
        lambda d: d.update(
            probe_error=True, fatal_stage="finalization", fatal_code="finalization"
        )
    )
    d = truncate_at(
        copy.deepcopy(BASE), "GetBucketEncryption", 0, "request-boundary", "request"
    )
    for n in (
        "GetBucketPolicy",
        "GetBucketPolicyStatus",
        "GetPublicAccessBlock",
        "GetBucketOwnershipControls",
    ):
        d["request_counts"][n] = 0
        d["operations"][n] = idle()
    witnesses["presend-fatal-skips-prior-operations"] = d
    return witnesses


class ProbeTests(unittest.TestCase):
    def assert_fatal(self, payload, code="framing", method="GET", stream=None):
        with self.assertRaises(probe.GlobalFatalFailure) as caught:
            probe.read_wire(stream or io.BytesIO(payload), method)
        self.assertEqual(caught.exception.code, code)
        response = caught.exception.response
        if response is not None:
            self.assertFalse(response["response_complete"])
            self.assertIsNone(response["response_sha256"])
            self.assertIsNone(response["response_bytes"])
        return caught.exception

    def test_fixed_requests_and_policy(self):
        expected = [
            ("HeadBucket", "HEAD", "/", ""),
            ("GetBucketLocation", "GET", "/", "location="),
            ("GetBucketPolicy", "GET", "/", "policy="),
            ("GetBucketPolicyStatus", "GET", "/", "policyStatus="),
            ("GetPublicAccessBlock", "GET", "/", "publicAccessBlock="),
            ("GetBucketOwnershipControls", "GET", "/", "ownershipControls="),
            ("GetBucketEncryption", "GET", "/", "encryption="),
            ("HeadObject", "HEAD", "/" + probe.OBJECT_NAME, ""),
            (
                "ListObjectsV2",
                "GET",
                "/",
                "list-type=2&max-keys=8&prefix=" + probe.OBJECT_NAME,
            ),
            (
                "ListObjectVersions",
                "GET",
                "/",
                "max-keys=8&prefix=" + probe.OBJECT_NAME + "&versions=",
            ),
        ]
        self.assertEqual(tuple(row[0] for row in expected), probe.OPERATIONS)
        for name, method, path, query in expected:
            service, host, m, p, q, headers, body = probe.request_spec(name)
            self.assertEqual(
                (service, host, m, p, q, body),
                ("s3", probe.S3_HOST, method, path, query, b""),
            )
            self.assertEqual(
                headers,
                {
                    "x-amz-expected-bucket-owner": probe.ACCOUNT,
                    "x-amz-content-sha256": probe.EMPTY_HASH,
                },
            )
        for name in (
            "GetObject",
            "DeleteObject",
            "GetBucketLogging",
            "GetBucketReplication",
            "AssumeRole",
            "identity2",
            "",
        ):
            with self.assertRaises(probe.GlobalFatalFailure):
                probe.request_spec(name)
        workflow = (
            Path(__file__).parents[1]
            / "workflows/ci-security-hf-cache-metadata-tail-read-canary.yml"
        )
        text = workflow.read_text()
        policy = text.split("inline-session-policy: >-\n            ", 1)[
            1
        ].splitlines()[0]
        self.assertEqual(len(policy), 1056)
        self.assertEqual(
            hashlib.sha256(policy.encode()).hexdigest(), probe.POLICY_SHA256
        )
        parsed = json.loads(policy)
        bucket = "arn:aws:s3:::" + probe.BUCKET
        self.assertEqual(
            parsed["Statement"][1],
            {
                "Effect": "Deny",
                "Action": "s3:*",
                "NotResource": [bucket, bucket + "/" + probe.OBJECT_NAME],
            },
        )
        self.assertEqual(probe.PR_NUMBER, "205614")
        self.assertEqual(probe.HEAD_REF, "gh/d4l3k/352/head")
        self.assertEqual(probe.BASE_REF, "gh/d4l3k/352/base")
        self.assertEqual(probe.TAG, "refs/tags/ciflow/inductor-cu126/205614")
        self.assertIn("tags: [ciflow/inductor-cu126/205614]", text)
        self.assertEqual(text.count("&& 205614 != 0"), 2)
        self.assertIn("runs-on: ubuntu-24.04", text)
        self.assertIn("disable-retry: true", text)
        self.assertNotIn("workflow_dispatch:", text)
        self.assertNotIn("role-chaining:", text)

    def test_complete_response_framing(self):
        body = b"abcdef"
        for version in (b"HTTP/1.0", b"HTTP/1.1"):
            for headers, raw, framing in (
                ([("Content-Length", "6")], body + b"unused", "content-length"),
                (
                    [("Transfer-Encoding", "chunked")],
                    b"2\r\nab\r\n4\r\ncdef\r\n0\r\nx-trace: ignored\r\n\r\n",
                    "chunked",
                ),
                ([], body, "close-delimited"),
            ):
                with self.subTest(version=version, framing=framing):
                    metadata, _, got = probe.read_wire(
                        ShortStream(wire(200, headers, raw, version)), "GET"
                    )
                    self.assertEqual(got, body)
                    self.assertEqual(
                        (
                            metadata["framing"],
                            metadata["response_bytes"],
                            metadata["response_sha256"],
                        ),
                        (framing, 6, hashlib.sha256(body).hexdigest()),
                    )
                    self.assertTrue(metadata["response_complete"])
        metadata, _, data = probe.read_wire(
            io.BytesIO(
                wire(
                    200,
                    [("Content-Length", "99999999999999999999")],
                    b"object-must-not-be-read",
                )
            ),
            "HEAD",
        )
        self.assertEqual(data, b"")
        self.assertEqual(
            (
                metadata["framing"],
                metadata["response_bytes"],
                metadata["response_sha256"],
            ),
            ("head-no-body", 0, probe.EMPTY_HASH),
        )
        metadata, _, data = probe.read_wire(io.BytesIO(wire(204)), "GET")
        self.assertEqual((metadata["framing"], data), ("status-no-body", b""))

    def test_status_and_header_negative_framing(self):
        cases = {
            "initial-100": b"HTTP/1.1 100 Continue\r\n\r\n" + wire(),
            "initial-101": wire(101),
            "http2": wire(version=b"HTTP/2.0"),
            "status-600": wire(600),
            "bad-status": b"HTTP/1.1 xyz Test\r\n\r\n",
            "missing-crlf": b"HTTP/1.1 200 OK\n\n",
            "obs-fold": b"HTTP/1.1 200 OK\r\n x-test: a\r\n\r\n",
            "space-name": b"HTTP/1.1 200 OK\r\nx test: a\r\n\r\n",
            "control-value": b"HTTP/1.1 200 OK\r\nx-test: a\x00\r\n\r\n",
            "duplicate-cl": wire(
                headers=[("Content-Length", "0"), ("content-length", "0")]
            ),
            "conflicting-cl": wire(
                headers=[("Content-Length", "0"), ("content-length", "1")]
            ),
            "duplicate-te": wire(
                headers=[
                    ("Transfer-Encoding", "chunked"),
                    ("transfer-encoding", "chunked"),
                ]
            ),
            "cl-te": wire(
                headers=[("Content-Length", "0"), ("Transfer-Encoding", "chunked")]
            ),
            "leading-zero-cl": wire(headers=[("Content-Length", "00")]),
            "negative-cl": wire(headers=[("Content-Length", "-1")]),
            "too-large-cl-number": wire(
                headers=[("Content-Length", "100000000000000000000")]
            ),
            "stacked-te": wire(headers=[("Transfer-Encoding", "gzip, chunked")]),
            "unsupported-content-encoding": wire(
                headers=[("Content-Encoding", "gzip")]
            ),
            "duplicate-id": wire(
                headers=[("x-amz-request-id", "ok"), ("X-Amz-Request-Id", "ok")]
            ),
            "invalid-id": wire(headers=[("x-amz-request-id", "bad value")]),
            "invalid-region": wire(headers=[("x-amz-bucket-region", "../bad")]),
            "invalid-marker": wire(headers=[("x-amz-delete-marker", "yes")]),
            "invalid-version": wire(headers=[("x-amz-version-id", "bad value")]),
            "204-cl": wire(204, [("Content-Length", "0")]),
            "header-count": wire(headers=[("x-test", "a")] * 101),
            "header-line-limit": wire(headers=[("x-test", "a" * 8192)]),
            "header-total-limit": wire(headers=[("x-test", "a" * 8000)] * 9),
        }
        for name, data in cases.items():
            with self.subTest(case=name):
                self.assert_fatal(data)
        for status in range(300, 400):
            self.assert_fatal(wire(status), "redirect")
        # Header ambiguity is forbidden even for a HEAD response.
        self.assert_fatal(cases["cl-te"], method="HEAD")

    def test_body_framing_negative(self):
        self.assert_fatal(wire(headers=[("Content-Length", "6")], body=b"abc"))
        for raw in (
            b"",
            b"1\r\naXX0\r\n\r\n",
            b"2\r\na",
            b"x\r\n",
            b"1;ext=x\r\na\r\n0\r\n\r\n",
            b"0\r\n",
            b"0\r\nx: y\r\n",
            b"0\r\nContent-Length: 0\r\n\r\n",
            b"0\r\nHost: elsewhere\r\n\r\n",
            b"0\r\nConnection: keep-alive\r\n\r\n",
            b"0\r\n" + b"x: y\r\n" * 101 + b"\r\n",
        ):
            with self.subTest(raw=raw[:40]):
                self.assert_fatal(
                    wire(headers=[("Transfer-Encoding", "chunked")], body=raw)
                )
        for headers, body in (
            ([("Content-Length", str(probe.LIMIT + 1))], b""),
            (
                [("Transfer-Encoding", "chunked")],
                f"{probe.LIMIT + 1:x}".encode() + b"\r\n",
            ),
            ([], b"a" * (probe.LIMIT + 1)),
        ):
            with self.assertRaises(probe.OperationLocalFailure) as caught:
                probe.read_wire(io.BytesIO(wire(headers=headers, body=body)), "GET")
            metadata = caught.exception.response
            self.assertEqual(caught.exception.code, "response-limit")
            self.assertTrue(metadata["body_limit_exceeded"])
            self.assertFalse(metadata["response_complete"])
            self.assertIsNone(metadata["response_sha256"])
        self.assert_fatal(
            b"", "tls", stream=FailingBody(wire(), ssl.SSLEOFError("inert"))
        )
        self.assert_fatal(b"", "network", stream=FailingBody(wire(), OSError("inert")))
        self.assert_fatal(
            b"",
            "deadline",
            stream=FailingBody(wire(), probe.GlobalFatalFailure("deadline")),
        )

    def test_exact_limit_and_head_no_read(self):
        body = b"x" * probe.LIMIT
        cases = [
            ([("Content-Length", str(probe.LIMIT))], body),
            (
                [("Transfer-Encoding", "chunked")],
                f"{probe.LIMIT:x}".encode() + b"\r\n" + body + b"\r\n0\r\n\r\n",
            ),
            ([], body),
        ]
        for headers, encoded in cases:
            metadata, _, result = probe.read_wire(
                io.BytesIO(wire(headers=headers, body=encoded)), "GET"
            )
            self.assertEqual(result, body)
            self.assertEqual(metadata["response_bytes"], probe.LIMIT)
        for headers in (
            [("Content-Length", "0")],
            [("Content-Length", "99999999999999999999")],
            [("Transfer-Encoding", "chunked")],
        ):
            metadata, _, body = probe.read_wire(
                FailingBody(
                    wire(headers=headers), AssertionError("body read forbidden")
                ),
                "HEAD",
            )
            self.assertEqual((metadata["response_bytes"], body), (0, b""))

    def test_failure_classes_all_tail_positions(self):
        for operation in probe.OPERATIONS[2:]:
            for code in sorted(probe.LOCAL_CODES):
                metadata = probe.finish_response(
                    probe.envelope(
                        200,
                        "head-no-body"
                        if operation == "HeadObject"
                        else "content-length",
                    ),
                    b"",
                )
                if code == "response-limit":
                    # HEAD has no entity, so it cannot encounter a body-size failure.
                    if operation == "HeadObject":
                        continue
                    metadata.update(
                        response_complete=False,
                        response_bytes=None,
                        response_sha256=None,
                        body_limit_exceeded=True,
                    )
                out, transport = run(
                    {
                        operation: probe.OperationLocalFailure(
                            code,
                            "transport" if code == "response-limit" else "projection",
                            metadata,
                        )
                    }
                )
                self.assertEqual(out["operations"][operation]["failure"]["code"], code)
                self.assertEqual(transport.calls, ["identity", *probe.OPERATIONS])
                self.assertFalse(out["probe_error"])
            for code in sorted(probe.FATAL_CODES):
                out, transport = run({operation: probe.GlobalFatalFailure(code)})
                self.assertEqual(out["fatal_code"], code)
                self.assertEqual(transport.calls[-1], operation)
            for exception in (
                ValueError,
                AssertionError,
                KeyError,
                TypeError,
                RecursionError,
                RuntimeError,
            ):
                out, transport = run(
                    {operation: exception("SECRET_EXCEPTION_SENTINEL")}
                )
                self.assertEqual(out["fatal_code"], "internal")
                self.assertEqual(transport.calls[-1], operation)
                self.assertNotIn("SECRET_EXCEPTION_SENTINEL", json.dumps(out))

    def test_actual_ssl_ragged_eof_control(self):
        # Exercise the selected stdlib method without opening a socket.
        sock = ssl.SSLSocket.__new__(ssl.SSLSocket)
        sock._closed = False
        sock._io_refs = 0
        sock._sslobj = mock.Mock()
        sock._sslobj.read.side_effect = ssl.SSLEOFError(ssl.SSL_ERROR_EOF, "inert")
        sock.suppress_ragged_eofs = True
        self.assertEqual(sock.read(1), b"")
        sock.suppress_ragged_eofs = False
        with self.assertRaises(ssl.SSLEOFError):
            sock.read(1)
        sock._sslobj = None
        sock.close()

    def test_exact_transport_budget_and_no_retry(self):
        transport = probe.Transport(CREDS)
        with mock.patch.object(
            probe, "https_request", side_effect=probe.GlobalFatalFailure("network")
        ) as invoke:
            with self.assertRaises(probe.GlobalFatalFailure):
                transport.request("HeadBucket")
            self.assertEqual(transport.counts["HeadBucket"], 1)
            with self.assertRaises(probe.GlobalFatalFailure) as second:
                transport.request("HeadBucket")
            self.assertEqual(second.exception.code, "request-budget")
            self.assertEqual(invoke.call_count, 1)
        with mock.patch.object(probe, "https_request") as invoke:
            with self.assertRaises(probe.GlobalFatalFailure):
                transport.request("GetObject")
            invoke.assert_not_called()

    def test_raw_tls_cleanup_and_first_failure(self):
        for response_error, close_error in (
            (None, False),
            (None, True),
            (probe.GlobalFatalFailure("network"), True),
            (probe.OperationLocalFailure("response-limit"), True),
        ):
            state = {"failed": False}
            sock = mock.Mock(spec=ssl.SSLSocket)
            stream = mock.Mock()
            sock.makefile.return_value = stream
            connection = mock.Mock()
            connection.sock = sock
            if close_error:
                stream.close.side_effect = OSError("inert")
            with (
                mock.patch.object(
                    probe.http.client, "HTTPSConnection", return_value=connection
                ),
                mock.patch.object(
                    probe,
                    "read_wire",
                    side_effect=response_error,
                    return_value=("fixed", {}, b""),
                ),
            ):
                if response_error or close_error:
                    with self.assertRaises(probe.GlobalFatalFailure) as caught:
                        probe.https_request(probe.S3_HOST, "HEAD", "/", b"", {}, state)
                    self.assertEqual(
                        caught.exception.code,
                        "network"
                        if isinstance(response_error, probe.GlobalFatalFailure)
                        else "finalization",
                    )
                else:
                    self.assertEqual(
                        probe.https_request(probe.S3_HOST, "HEAD", "/", b"", {}, state)[
                            0
                        ],
                        "fixed",
                    )
            self.assertEqual(sock.suppress_ragged_eofs, False)
            self.assertEqual(state["failed"], close_error)
            stream.close.assert_called_once()
            connection.close.assert_called_once()

    def test_complete_fixed_sequence(self):
        out, transport = run()
        self.assertEqual(transport.calls, ["identity", *probe.OPERATIONS])
        self.assertTrue(out["all_operations_attempted"])
        self.assertTrue(out["all_responses_projected"])
        self.assertFalse(out["probe_error"])
        self.assertFalse(out["write_authorized"])
        self.assertNotIn("NEVER_PRINT", json.dumps(out))
        self.assertEqual(
            out["operations"]["GetPublicAccessBlock"]["projection"],
            {
                "state": "present",
                "block_public_acls": True,
                "ignore_public_acls": True,
                "block_public_policy": True,
                "restrict_public_buckets": True,
                "all_enabled": True,
            },
        )
        self.assertEqual(
            out["operations"]["ListObjectVersions"]["projection"],
            {
                "state": "complete",
                "exact_entries": 0,
                "exact_current_versions": 0,
                "exact_noncurrent_versions": 0,
                "exact_delete_markers": 0,
                "exact_current_delete_markers": 0,
                "other_prefix_entries": 0,
            },
        )

    def test_operation_errors_continue_but_boundaries_stop(self):
        for name in probe.OPERATIONS[2:]:
            with self.subTest(operation=name):
                bad = (
                    (200, {"content-length": "0"}, b"")
                    if name == "HeadObject"
                    else (200, {}, b"<bad>")
                )
                if name == "HeadObject":
                    bad = (200, {"x-amz-delete-marker": "true"}, b"")
                out, transport = run({name: bad})
                self.assertEqual(
                    out["operations"][name]["disposition"], "operation-error"
                )
                self.assertFalse(out["probe_error"])
                self.assertEqual(len(transport.calls), 11)
                self.assertFalse(out["all_responses_projected"])
        for name, row, fatal in (
            ("identity", (200, {}, identity(role="wrong")), "identity"),
            ("HeadBucket", (403, {}, b""), "owner-region"),
            (
                "GetBucketLocation",
                (200, {}, s3("LocationConstraint", "eu-west-1")),
                "owner-region",
            ),
        ):
            out, transport = run({name: row})
            self.assertEqual((out["fatal_stage"], out["fatal_code"]), (name, fatal))
            self.assertEqual(transport.calls[-1], name)
        out, transport = run({"GetBucketPolicy": error("InvalidToken", 400)})
        self.assertEqual(out["fatal_code"], "authentication")
        self.assertEqual(transport.calls[-1], "GetBucketPolicy")
        out, _ = run({"GetBucketPolicy": error("InvalidToken", 500)})
        self.assertEqual(out["fatal_code"], "authentication")
        out, _ = run({"GetBucketPolicy": error("AccessDenied", 500)})
        self.assertEqual(
            out["operations"]["GetBucketPolicy"]["projection"]["reason"],
            "service-status",
        )

    def test_all_fatal_positions_and_finalization(self):
        for operation in probe.OPERATIONS:
            out, transport = run({operation: probe.GlobalFatalFailure("network")})
            self.assertEqual(
                (out["fatal_stage"], out["fatal_code"]), (operation, "network")
            )
            self.assertEqual(transport.calls[-1], operation)
            out, transport = run(presend=operation)
            self.assertEqual(out["request_counts"][operation], 0)
            self.assertEqual(
                out["operations"][operation]["failure"]["code"], "request-boundary"
            )
        out, _ = run(cleanup=False)
        self.assertEqual(
            (out["fatal_stage"], out["fatal_code"]), ("finalization", "finalization")
        )
        self.assertFalse(out["collector_cleanup_confirmed"])
        out, _ = run(
            {"GetBucketPolicy": probe.GlobalFatalFailure("network")}, cleanup=False
        )
        self.assertEqual(
            (out["fatal_stage"], out["fatal_code"]), ("GetBucketPolicy", "network")
        )

    def test_absence_status_and_headers(self):
        for operation, code in probe.ABSENT_CONFIG.items():
            out, _ = run({operation: error(code, 404)})
            self.assertEqual(
                out["operations"][operation]["projection"], {"state": "not-configured"}
            )
            out, _ = run({operation: error(code, 403)})
            self.assertEqual(
                out["operations"][operation]["projection"]["state"], "unavailable"
            )
        for headers, state in (
            ({}, "current-key-not-found"),
            (
                {"x-amz-delete-marker": "true", "x-amz-version-id": "fixed-version"},
                "delete-marker-present",
            ),
            ({"x-amz-version-id": "fixed-version"}, "unavailable"),
        ):
            out, _ = run({"HeadObject": (404, headers, b"")})
            self.assertEqual(
                out["operations"]["HeadObject"]["projection"]["state"], state
            )
        out, _ = run(
            {
                "HeadObject": (
                    200,
                    {"content-length": "123", "x-amz-version-id": "fixed-version"},
                    b"",
                )
            }
        )
        self.assertEqual(
            out["operations"]["HeadObject"]["projection"],
            {
                "state": "current-key-present",
                "bytes": 123,
                "version_id_present": True,
                "delete_marker": False,
            },
        )

    def test_version_and_list_shape(self):
        def entry(kind, key, version, latest):
            return f"<{kind}><Key>{key}</Key><VersionId>{version}</VersionId><IsLatest>{str(latest).lower()}</IsLatest></{kind}>"

        key = probe.OBJECT_NAME
        rows = (
            entry("Version", key, "v1", False)
            + entry("Version", key, "v2", False)
            + entry("DeleteMarker", key, "v3", True)
            + entry("Version", key + "-other", "v4", True)
        )
        out, _ = run({"ListObjectVersions": (200, {}, list_response(True, rows))})
        projection = out["operations"]["ListObjectVersions"]["projection"]
        self.assertEqual(
            projection,
            {
                "state": "complete",
                "exact_entries": 3,
                "exact_current_versions": 0,
                "exact_noncurrent_versions": 2,
                "exact_delete_markers": 1,
                "exact_current_delete_markers": 1,
                "other_prefix_entries": 1,
            },
        )
        self.assertNotIn("v1", json.dumps(projection))
        invalid = [
            entry("Version", key, "v1", True) * 2,
            entry("Version", key, "v1", True) + entry("DeleteMarker", key, "v2", True),
            entry("Version", "outside", "v1", False),
            entry("Version", key, "", False),
            entry("Version", key, "v1", False) * 9,
        ]
        for rows in invalid:
            out, _ = run({"ListObjectVersions": (200, {}, list_response(True, rows))})
            self.assertEqual(
                out["operations"]["ListObjectVersions"]["disposition"],
                "operation-error",
            )
        for data in (
            list_response().replace(b"<IsTruncated>false", b"<IsTruncated>true"),
            list_response(rows=f"<Contents><Key>{key}</Key></Contents>" * 2, count=2),
            list_response(count=1),
            list_response().replace(key.encode(), b"other", 1),
        ):
            out, _ = run({"ListObjectsV2": (200, {}, data)})
            self.assertEqual(
                out["operations"]["ListObjectsV2"]["disposition"], "operation-error"
            )

    def test_xml_json_controls(self):
        for data in (
            b'<!DOCTYPE x [<!ENTITY e "bad">]><x/>',
            b"<x>\0</x>",
            b"\xff",
            b"<bad>",
            b"<x>" + b"<y/>" * 4097 + b"</x>",
        ):
            with self.assertRaises(probe.OperationLocalFailure):
                probe.xml(data)
        for data in (
            b'{"Statement":[],"Statement":[]}',
            b'{"Statement":[]} trailing',
            b'{"Statement":"bad"}',
            b'{"Statement":[],"Bad":1}',
            b"\xff",
        ):
            out, _ = run({"GetBucketPolicy": (200, {}, data)})
            self.assertEqual(
                out["operations"]["GetBucketPolicy"]["failure"]["code"], "json-shape"
            )

    def test_independent_d1_d2_witnesses(self):
        witnesses = independent_witnesses()
        self.assertEqual(len(witnesses), 15)
        for name, value in witnesses.items():
            with (
                self.subTest(witness=name),
                self.assertRaisesRegex(
                    probe.GlobalFatalFailure, "^internal$"
                ) as caught,
            ):
                probe.validate_output(value)
            self.assertEqual(
                (caught.exception.code, caught.exception.stage),
                ("internal", "validation"),
            )

    def test_http401_stops_actual_collector(self):
        baseline, all_calls = run_wire()
        self.assertFalse(baseline["probe_error"])
        self.assertEqual(all_calls, ["identity", *probe.OPERATIONS])
        payloads = (
            wire(401),
            wire(401, [("Content-Length", str(probe.LIMIT + 1))]),
            wire(
                401,
                [("Transfer-Encoding", "chunked")],
                f"{probe.LIMIT + 1:x}\r\n".encode(),
            ),
            wire(401, body=b"x" * (probe.LIMIT + 1)),
        )
        for operation in ("identity", *probe.OPERATIONS):
            for payload in payloads:
                with self.subTest(operation=operation, payload_size=len(payload)):
                    out, calls = run_wire({operation: payload})
                    self.assertEqual(calls, all_calls[: all_calls.index(operation) + 1])
                    self.assertEqual(
                        (out["fatal_stage"], out["fatal_code"]),
                        (operation, "authentication"),
                    )
                    if operation != "identity":
                        record = out["operations"][operation]
                        self.assertIsNone(record["projection"])
                        self.assertEqual(record["response"], probe.envelope(401))
        out, calls = run_wire({"HeadObject": wire(403)})
        self.assertEqual(calls, all_calls)
        self.assertFalse(out["probe_error"])
        self.assertEqual(
            out["operations"]["HeadObject"]["projection"],
            {"state": "unavailable", "reason": "head-status"},
        )
        for operation in ("HeadBucket", "HeadObject", "GetBucketPolicy"):
            with self.assertRaisesRegex(probe.GlobalFatalFailure, "^authentication$"):
                probe.status_projection(operation, 401, {}, b"")

    def test_http401_and_head_output_invariants(self):
        baseline, _ = run_wire()
        for name in ("HeadBucket", "HeadObject"):
            for framing in (
                "content-length",
                "chunked",
                "close-delimited",
                "head-no-body",
                "unaccepted",
            ):
                invalid = copy.deepcopy(baseline)
                response = probe.envelope(200, framing)
                response["body_limit_exceeded"] = True
                invalid["operations"][name] = {
                    "disposition": "operation-error",
                    "response": response,
                    "projection": None,
                    "failure": {
                        "class": "operation-local",
                        "stage": "transport",
                        "code": "response-limit",
                    },
                }
                invalid["all_responses_projected"] = False
                with (
                    self.subTest(name=name, framing=framing),
                    self.assertRaisesRegex(probe.GlobalFatalFailure, "^internal$"),
                ):
                    probe.validate_output(invalid)
            for framing in ("unaccepted", "head-no-body"):
                out, _ = run(
                    {
                        name: probe.GlobalFatalFailure(
                            "framing", response=probe.envelope(200, framing)
                        )
                    }
                )
                self.assertEqual(
                    (out["fatal_stage"], out["fatal_code"]), (name, "framing")
                )
        invalid = copy.deepcopy(baseline)
        record = invalid["operations"]["HeadObject"]
        record["response"]["http_status"] = 401
        record["projection"] = {"state": "unavailable", "reason": "head-status"}
        with self.assertRaisesRegex(probe.GlobalFatalFailure, "^internal$"):
            probe.validate_output(invalid)
        invalid = copy.deepcopy(baseline)
        response = probe.envelope(401, "content-length")
        response["body_limit_exceeded"] = True
        invalid["operations"]["GetBucketPolicy"] = {
            "disposition": "operation-error",
            "response": response,
            "projection": None,
            "failure": {
                "class": "operation-local",
                "stage": "transport",
                "code": "response-limit",
            },
        }
        invalid["all_responses_projected"] = False
        with self.assertRaisesRegex(probe.GlobalFatalFailure, "^internal$"):
            probe.validate_output(invalid)

    def test_output_schema_negative_relations(self):
        base, _ = run()
        mutations = [
            lambda d: d.update(extra="NEVER_PRINT"),
            lambda d: d.update(write_authorized=True),
            lambda d: d["request_counts"].update(HeadObject=True),
            lambda d: d["operations"]["GetBucketPolicy"]["response"].update(
                response_complete=False
            ),
            lambda d: d["operations"]["GetPublicAccessBlock"]["projection"].update(
                all_enabled=False
            ),
            lambda d: d["operations"]["GetBucketEncryption"]["projection"].update(
                algorithms=["AES256", "AES256"]
            ),
            lambda d: d["operations"]["ListObjectVersions"]["projection"].update(
                exact_entries=1
            ),
            lambda d: d["operations"]["ListObjectsV2"]["projection"].update(
                key="NEVER_PRINT"
            ),
            lambda d: d["operations"]["GetBucketPolicy"]["projection"].update(
                response_sha256="0" * 64
            ),
        ]
        for mutation in mutations:
            value = copy.deepcopy(base)
            mutation(value)
            with self.assertRaises(probe.GlobalFatalFailure):
                probe.validate_output(value)

    def test_source_environment_and_no_credentials(self):
        self.assertTrue(probe.scope_valid(environment()))
        self.assertTrue(probe.source_matches(source(), "a" * 40))
        self.assertFalse(
            probe.scope_valid(
                environment() | {"GITHUB_REF": "refs/tags/ciflow/inductor-cu126/205615"}
            )
        )
        for part in ("head", "base"):
            invalid = source()
            invalid[part]["ref"] = f"gh/d4l3k/351/{part}"
            self.assertFalse(probe.source_matches(invalid, "a" * 40))
        with mock.patch.object(probe, "PR_NUMBER", "0"):
            self.assertFalse(probe.scope_valid(environment()))
        with activated_scope():
            env = environment()
            self.assertTrue(probe.scope_valid(env))
            self.assertTrue(probe.preflight(env))
            self.assertTrue(probe.source_matches(source(), "a" * 40))
            for key, value in (
                ("GITHUB_ACTOR_ID", "1"),
                ("GITHUB_RUN_ATTEMPT", "2"),
                ("GITHUB_SHA", "b" * 40),
                ("RUNNER_ENVIRONMENT", "self-hosted"),
                ("AWS_PROFILE", "other"),
                ("AWS_ENDPOINT_URL", "https://other"),
                ("AWS_MAX_ATTEMPTS", "2"),
                ("SSL_CERT_FILE", "/tmp/other"),
                ("INPUT_ROLE_TO_ASSUME", "other"),
            ):
                with self.subTest(key=key), self.assertRaises(probe.GlobalFatalFailure):
                    probe.preflight(env | {key: value})
            with mock.patch.object(
                probe,
                "https_request",
                return_value=(
                    probe.finish_response(probe.envelope(200, "content-length"), b""),
                    {},
                    json.dumps(source()).encode(),
                ),
            ) as request:
                probe.validate_source(env)
                self.assertEqual(request.call_count, 1)
                self.assertEqual(
                    request.call_args.args[:3],
                    ("api.github.com", "GET", "/repos/pytorch/pytorch/pulls/123"),
                )
            with self.assertRaises(probe.GlobalFatalFailure):
                probe.runtime_inputs(env)
            self.assertEqual(
                probe.runtime_inputs(env | dict(zip(probe.CREDENTIAL_KEYS, CREDS))),
                CREDS,
            )

    def test_main_zero_skeleton_and_redaction(self):
        with (
            mock.patch.dict(probe.os.environ, {}, clear=True),
            mock.patch.object(probe.signal, "signal"),
            mock.patch.object(probe.signal, "alarm"),
            mock.patch.object(probe.resource, "setrlimit"),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(probe.main(), 1)
        parsed = json.loads(output.getvalue())
        self.assertEqual(parsed["fatal_stage"], "environment")
        self.assertEqual(sum(parsed["request_counts"].values()), 0)
        self.assertTrue(parsed["collector_cleanup_confirmed"])
        probe.validate_output(parsed)


if __name__ == "__main__":

    def deny_network(*_args, **_kwargs):
        raise AssertionError("offline-network-denied")

    with (
        mock.patch.object(socket.socket, "connect", deny_network),
        mock.patch.object(socket.socket, "connect_ex", deny_network),
        mock.patch.object(socket, "create_connection", deny_network),
        mock.patch.object(socket, "getaddrinfo", deny_network),
    ):
        unittest.main()
