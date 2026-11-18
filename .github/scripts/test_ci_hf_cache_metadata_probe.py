# Owner(s): ["module: ci"]
"""Offline controls. Synthetic responses are not service/deployment evidence."""

import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import unittest
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "probe", Path(__file__).with_name("ci_hf_cache_metadata_probe.py")
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


def responder(overrides=None):
    values = responses() | (overrides or {})
    calls = []

    def invoke(operation):
        calls.append(operation)
        return values[operation]

    return invoke, calls


def environment():
    return dict(
        GITHUB_REPOSITORY="pytorch/pytorch",
        GITHUB_REPOSITORY_ID="65600975",
        GITHUB_EVENT_NAME="pull_request",
        GITHUB_REF=f"refs/pull/{probe.PR_NUMBER}/merge",
        CANARY_PR_NUMBER=probe.PR_NUMBER,
        CANARY_PR_AUTHOR="ptdbot",
        CANARY_PR_AUTHOR_ID="283013143",
        CANARY_HEAD_REPO="ptdbot/pytorch",
        CANARY_HEAD_REPO_ID="1372177501",
        CANARY_HEAD_REF=probe.HEAD_REF,
        CANARY_ROLE_NAME=probe.ROLE,
        ACTIONS_ID_TOKEN_REQUEST_URL="https://fixture.invalid/never-called",
        ACTIONS_ID_TOKEN_REQUEST_TOKEN="NONSECRET-FIXTURE",
        AWS_EC2_METADATA_DISABLED="true",
        AWS_MAX_ATTEMPTS="1",
        AWS_CONFIG_FILE="/dev/null",
        AWS_SHARED_CREDENTIALS_FILE="/dev/null",
    )


class TestHfCacheMetadataProbe(unittest.TestCase):
    def test_exact_request_allowlist(self):
        self.assertEqual(len(probe.OPERATIONS), 16)
        self.assertEqual(len(set(probe.OPERATIONS)), 16)
        for operation in probe.OPERATIONS:
            service, host, method, path, query, headers, body = probe.request_spec(
                operation
            )
            self.assertEqual((service, host, body), ("s3", probe.S3_HOST, b""))
            self.assertEqual(
                method, "HEAD" if operation in ("HeadBucket", "HeadObject") else "GET"
            )
            self.assertEqual(
                path, "/" + probe.OBJECT_NAME if operation == "HeadObject" else "/"
            )
            self.assertEqual(
                headers,
                {
                    "x-amz-expected-bucket-owner": probe.ACCOUNT,
                    "x-amz-content-sha256": hashlib.sha256(b"").hexdigest(),
                },
            )
            parsed = urllib.parse.parse_qsl(
                query, keep_blank_values=True, strict_parsing=True
            )
            self.assertEqual(parsed, sorted(parsed))
            if operation in ("ListObjectsV2", "ListObjectVersions"):
                self.assertEqual(
                    dict(parsed),
                    {
                        ("list-type" if operation == "ListObjectsV2" else "versions"): (
                            "2" if operation == "ListObjectsV2" else ""
                        ),
                        "max-keys": "8",
                        "prefix": probe.OBJECT_NAME,
                    },
                )
            elif operation in probe.CONFIG_QUERIES:
                self.assertEqual(parsed, [(probe.CONFIG_QUERIES[operation], "")])
            else:
                self.assertEqual(query, "")
        self.assertEqual(
            probe.request_spec("identity")[:5],
            ("sts", "sts.us-east-1.amazonaws.com", "POST", "/", ""),
        )
        for operation in (
            "GetObject",
            "PutObject",
            "DeleteObject",
            "DeleteObjectVersion",
            "CopyObject",
            "AssumeRole",
            "ListBuckets",
            "",
            None,
        ):
            with self.assertRaisesRegex(ValueError, "^operation$"):
                probe.request_spec(operation)

    def test_complete_read_sequence_and_redaction(self):
        invoke, calls = responder()
        out = probe.run_probe(invoke)
        self.assertFalse(out["probe_error"] or out["write_authorized"])
        self.assertTrue(
            out["identity_matches"] and out["all_metadata_requests_attempted"]
        )
        self.assertEqual(calls, ["identity", *probe.OPERATIONS])
        self.assertEqual(
            out["operations"]["GetBucketVersioning"]["state"], "never-enabled"
        )
        self.assertEqual(
            out["operations"]["GetBucketPolicy"]["state"], "present-unreviewed"
        )
        self.assertNotIn("NEVER_PRINT", json.dumps(out))
        self.assertNotIn("Statement", json.dumps(out))
        self.assertEqual(out["operations"]["ListObjectVersions"]["exact_entries"], 0)

    def test_identity_and_bucket_gates(self):
        for key, value, expected in (
            ("identity", (200, {}, identity(account="000000000000")), ["identity"]),
            ("identity", (200, {}, identity(role="other-role")), ["identity"]),
            ("HeadBucket", (403, {}, b""), ["identity", "HeadBucket"]),
            (
                "HeadBucket",
                (200, {"x-amz-bucket-region": "us-west-2"}, b""),
                ["identity", "HeadBucket"],
            ),
            (
                "GetBucketLocation",
                error(),
                ["identity", "HeadBucket", "GetBucketLocation"],
            ),
            (
                "GetBucketLocation",
                (200, {}, s3("LocationConstraint", "eu-west-1")),
                ["identity", "HeadBucket", "GetBucketLocation"],
            ),
        ):
            invoke, calls = responder({key: value})
            out = probe.run_probe(invoke)
            self.assertEqual(calls, expected)
            self.assertFalse(out.get("all_metadata_requests_attempted", False))
            self.assertFalse(out["write_authorized"])

    def test_config_denial_is_not_absence(self):
        for operation in probe.CONFIG_QUERIES:
            denied = probe.project(operation, error())
            self.assertEqual(denied["state"], "unavailable")
            self.assertEqual(denied["reason"], "access-denied")
            absent = probe.project(operation, error("NoSuchBucket", 404))
            self.assertEqual(absent["state"], "unavailable")
        for operation, code in probe.ABSENT_CONFIG.items():
            self.assertEqual(
                probe.project(operation, error(code, 404))["state"], "not-configured"
            )
            self.assertEqual(
                probe.project(operation, error(code, 403))["state"], "unavailable"
            )
        invoke, calls = responder(
            {
                "GetBucketReplication": error(),
                "GetBucketLifecycleConfiguration": error(),
            }
        )
        out = probe.run_probe(invoke)
        self.assertFalse(out["probe_error"])
        self.assertEqual(calls, ["identity", *probe.OPERATIONS])
        self.assertFalse(out["write_authorized"])

    def test_version_lock_notification_boundaries(self):
        for value in ("Enabled", "Suspended"):
            row = probe.project(
                "GetBucketVersioning",
                (
                    200,
                    {},
                    s3(
                        "VersioningConfiguration",
                        f"<Status>{value}</Status><MfaDelete>Enabled</MfaDelete>",
                    ),
                ),
            )
            self.assertEqual((row["state"], row["mfa_delete"]), (value, "Enabled"))
        cases = (
            (
                "GetBucketVersioning",
                s3("VersioningConfiguration", "<Status>Disabled</Status>"),
                "S3 value",
            ),
            (
                "GetBucketVersioning",
                s3("VersioningConfiguration", "<Unknown/>"),
                "S3 XML shape",
            ),
            (
                "GetBucketVersioning",
                s3(
                    "VersioningConfiguration",
                    "<Status>Enabled</Status><Status>Enabled</Status>",
                ),
                "S3 value",
            ),
            (
                "GetObjectLockConfiguration",
                s3(
                    "ObjectLockConfiguration",
                    "<ObjectLockEnabled>Disabled</ObjectLockEnabled>",
                ),
                "S3 value",
            ),
            (
                "GetBucketNotificationConfiguration",
                s3("NotificationConfiguration", "<Unexpected/>"),
                "S3 XML shape",
            ),
            (
                "GetBucketPolicyStatus",
                s3("PolicyStatus", "<IsPublic>unknown</IsPublic>"),
                "S3 value",
            ),
            (
                "GetBucketOwnershipControls",
                s3(
                    "OwnershipControls",
                    "<Rule><ObjectOwnership>unknown</ObjectOwnership></Rule>",
                ),
                "S3 value",
            ),
        )
        for operation, body, expected in cases:
            with self.subTest(operation=operation, body=body):
                with self.assertRaisesRegex(ValueError, f"^{expected}$"):
                    probe.project(operation, (200, {}, body))
        lock = probe.project(
            "GetObjectLockConfiguration",
            (
                200,
                {},
                s3(
                    "ObjectLockConfiguration",
                    "<ObjectLockEnabled>Enabled</ObjectLockEnabled><Rule/>",
                ),
            ),
        )
        self.assertEqual(
            (lock["state"], lock["default_rule_present"]), ("enabled", True)
        )
        notify = probe.project(
            "GetBucketNotificationConfiguration",
            (200, {}, s3("NotificationConfiguration", "<EventBridgeConfiguration/>")),
        )
        self.assertEqual((notify["state"], notify["rules"]), ("configured", 1))

    def test_exact_key_and_version_inventory(self):
        ordinary = f"<Contents><Key>{probe.OBJECT_NAME}</Key><ETag>NEVER_PRINT</ETag></Contents>"
        other = f"<Contents><Key>{probe.OBJECT_NAME}-other</Key></Contents>"
        row = probe.project(
            "ListObjectsV2", (200, {}, list_response(rows=ordinary + other, count=2))
        )
        self.assertEqual((row["exact_entries"], row["other_prefix_entries"]), (1, 1))
        self.assertNotIn("NEVER_PRINT", json.dumps(row))
        version = f"<Version><Key>{probe.OBJECT_NAME}</Key><VersionId>opaque</VersionId><IsLatest>false</IsLatest></Version>"
        marker = f"<DeleteMarker><Key>{probe.OBJECT_NAME}</Key><VersionId>null</VersionId><IsLatest>true</IsLatest></DeleteMarker>"
        row = probe.project(
            "ListObjectVersions", (200, {}, list_response(True, version + marker))
        )
        self.assertEqual((row["exact_entries"], row["exact_delete_markers"]), (2, 1))
        absent = probe.project("HeadObject", (404, {}, b""))
        marker = probe.project(
            "HeadObject", (404, {"x-amz-delete-marker": "true"}, b"")
        )
        denied = probe.project("HeadObject", (403, {}, b""))
        ambiguous = probe.project(
            "HeadObject", (404, {"x-amz-version-id": "opaque"}, b"")
        )
        self.assertEqual(ambiguous["state"], "unavailable")
        with self.assertRaisesRegex(ValueError, "delete marker header"):
            probe.project("HeadObject", (404, {"x-amz-delete-marker": "unknown"}, b""))
        self.assertEqual(
            [x["state"] for x in (absent, marker, denied)],
            ["current-key-not-found", "delete-marker-present", "unavailable"],
        )
        present = probe.project(
            "HeadObject",
            (200, {"content-length": "17", "x-amz-version-id": "NEVER_PRINT"}, b""),
        )
        self.assertEqual(present["bytes"], 17)
        self.assertTrue(present["version_id_present"])
        self.assertNotIn("NEVER_PRINT", json.dumps(present))

    def test_listing_errors_fail_closed(self):
        normal = list_response()
        changes = (
            (normal.replace(b"<IsTruncated>false", b"<IsTruncated>true"), "S3 value"),
            (normal.replace(b"<MaxKeys>8", b"<MaxKeys>9"), "list limit"),
            (normal.replace(b"<KeyCount>0", b"<KeyCount>1"), "key count"),
            (normal.replace(probe.BUCKET.encode(), b"other-bucket"), "list scope"),
            (normal.replace(probe.OBJECT_NAME.encode(), b"other-prefix"), "list scope"),
            (
                normal.replace(
                    b"</ListBucketResult>",
                    b"<NextContinuationToken>opaque</NextContinuationToken></ListBucketResult>",
                ),
                "S3 XML shape",
            ),
            (
                list_response(rows="<Contents><Key>other</Key></Contents>", count=1),
                "listed key",
            ),
            (
                list_response(
                    rows=(f"<Contents><Key>{probe.OBJECT_NAME}</Key></Contents>" * 9),
                    count=9,
                ),
                "list count",
            ),
        )
        for body, expected in changes:
            with self.subTest(body=body):
                with self.assertRaisesRegex(ValueError, f"^{expected}$"):
                    probe.project("ListObjectsV2", (200, {}, body))
        body = list_response(True).replace(
            b"</ListVersionsResult>",
            b"<NextKeyMarker>opaque</NextKeyMarker></ListVersionsResult>",
        )
        with self.assertRaisesRegex(ValueError, "^S3 XML shape$"):
            probe.project("ListObjectVersions", (200, {}, body))

    def test_policy_xml_and_header_redaction(self):
        for body, expected in (
            (b'{"Statement":[],"Statement":[]}', "duplicate key"),
            (b"[]", "policy shape"),
            (b'{"Statement":["bad"]}', "policy statements"),
            (b'{"Statement":[],"unexpected":true}', "policy shape"),
        ):
            with self.subTest(body=body):
                with self.assertRaisesRegex(ValueError, f"^{expected}$"):
                    probe.project("GetBucketPolicy", (200, {}, body))
        with self.assertRaisesRegex(ValueError, "XML boundary"):
            probe.xml(b"<!ENTITY example>")
        with self.assertRaisesRegex(ValueError, "XML boundary"):
            probe.xml("<!DOCTYPE root><root/>".encode("utf-16"))
        with self.assertRaisesRegex(ValueError, "versioning shape"):
            probe.project(
                "GetBucketVersioning",
                (
                    200,
                    {},
                    s3("VersioningConfiguration", "<MfaDelete>Enabled</MfaDelete>"),
                ),
            )
        with self.assertRaisesRegex(ET.ParseError, "^syntax error"):
            probe.xml(b"not xml")
        with self.assertRaisesRegex(ValueError, "projection boundary"):
            probe.project("GetBucketPolicy", (200, {}, b"x" * (probe.LIMIT + 1)))
        result = probe.project(
            "HeadObject",
            (
                404,
                {"x-amz-request-id": "valid_ID-1", "x-amz-version-id": "NEVER_PRINT"},
                b"",
            ),
        )
        self.assertEqual(result["request_id"], "valid_ID-1")
        result = probe.project(
            "HeadObject", (404, {"x-amz-request-id": "unsafe\nNEVER_PRINT"}, b"")
        )
        self.assertNotIn("request_id", result)
        self.assertNotIn("NEVER_PRINT", json.dumps(result))

    def test_transport_scope_budget_and_boundaries(self):
        response = mock.Mock(status=200)
        response.read.return_value = b""
        response.getheaders.return_value = [("x-amz-bucket-region", probe.REGION)]
        connection = mock.Mock()
        connection.getresponse.return_value = response
        with mock.patch.object(
            probe.http.client, "HTTPSConnection", return_value=connection
        ) as factory:
            transport = probe.Transport(CREDS)
            transport.request("HeadObject")
            self.assertEqual(factory.call_args.args, (probe.S3_HOST,))
            args, kwargs = connection.request.call_args
            self.assertEqual(args, ("HEAD", "/" + probe.OBJECT_NAME))
            self.assertEqual(kwargs["body"], b"")
            self.assertEqual(
                kwargs["headers"]["x-amz-expected-bucket-owner"], probe.ACCOUNT
            )
            with self.assertRaisesRegex(ValueError, "request budget"):
                transport.request("HeadObject")
            self.assertEqual(factory.call_count, 1)
            transport.request("identity")
            with self.assertRaisesRegex(ValueError, "request budget"):
                transport.request("identity")
            self.assertEqual(transport.counts["identity"], 1)
            self.assertEqual(factory.call_count, 2)
            with self.assertRaisesRegex(ValueError, "operation"):
                transport.request("GetObject")
        self.assertTrue(connection.close.called)
        for status, data, response_headers, expected in (
            (302, b"", [], "redirect"),
            (200, b"x" * (probe.LIMIT + 1), [], "response size"),
            (
                200,
                b"",
                [("x-amz-bucket-region", "a"), ("X-Amz-Bucket-Region", "b")],
                "duplicate response header",
            ),
        ):
            (
                response.status,
                response.read.return_value,
                response.getheaders.return_value,
            ) = status, data, response_headers
            with mock.patch.object(
                probe.http.client, "HTTPSConnection", return_value=connection
            ):
                with self.assertRaisesRegex(ValueError, expected):
                    probe.Transport(CREDS).request("HeadBucket")
        self.assertEqual(response.read.call_args.args, (probe.LIMIT + 1,))

    def test_exact_owned_routes(self):
        self.assertEqual(probe.PR_NUMBER, "205566")
        self.assertEqual(probe.TAG, "refs/tags/ciflow/inductor-cu126/205566")
        with mock.patch.object(probe, "PR_NUMBER", "0"):
            self.assertFalse(probe.scope_valid(environment()))
        with self.subTest(pr_number=probe.PR_NUMBER):
            env = environment()
            self.assertTrue(probe.scope_valid(env))
            self.assertTrue(probe.preflight(env))
            for key in (
                "GITHUB_REPOSITORY",
                "GITHUB_REPOSITORY_ID",
                "GITHUB_REF",
                "CANARY_PR_NUMBER",
                "CANARY_PR_AUTHOR",
                "CANARY_PR_AUTHOR_ID",
                "CANARY_HEAD_REPO",
                "CANARY_HEAD_REPO_ID",
                "CANARY_HEAD_REF",
            ):
                self.assertFalse(probe.scope_valid(env | {key: "wrong"}), key)
            push = env | {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": probe.TAG}
            self.assertTrue(probe.scope_valid(push))
            self.assertFalse(probe.scope_valid(push | {"GITHUB_REF": probe.TAG + "0"}))
            for key in (
                "AWS_PROFILE",
                "AWS_ROLE_ARN",
                "AWS_WEB_IDENTITY_TOKEN_FILE",
                "AWS_ENDPOINT_URL",
                "AWS_ENDPOINT_URL_S3",
                "AWS_CA_BUNDLE",
                "SSL_CERT_FILE",
                "NODE_OPTIONS",
                "INPUT_AWS_ACCESS_KEY_ID",
                *probe.CREDENTIAL_KEYS,
            ):
                with self.assertRaisesRegex(ValueError, "configuration"):
                    probe.preflight(env | {key: "forbidden"})
            creds_env = env | dict(zip(probe.CREDENTIAL_KEYS, CREDS))
            self.assertEqual(probe.runtime_inputs(creds_env), CREDS)
            for invalid in (None, "", "0", "2", "01", "1junk", "1.0", " 1", "1 "):
                with self.subTest(max_attempts=invalid):
                    for inputs, validate in (
                        (env, probe.preflight),
                        (creds_env, probe.runtime_inputs),
                    ):
                        bad = inputs | {"AWS_MAX_ATTEMPTS": invalid}
                        if invalid is None:
                            del bad["AWS_MAX_ATTEMPTS"]
                        with self.assertRaisesRegex(ValueError, "^configuration$"):
                            validate(bad)
            with self.assertRaisesRegex(ValueError, "credentials"):
                probe.runtime_inputs(
                    creds_env | {"AWS_ACCESS_KEY_ID": "AKIA" + "A" * 16}
                )
            with self.assertRaisesRegex(ValueError, "configuration"):
                probe.runtime_inputs(creds_env | {"AWS_REGION": "us-west-2"})

    def test_workflow_and_session_policy(self):
        path = (
            Path(__file__).parents[1]
            / "workflows/ci-security-hf-cache-metadata-read-canary.yml"
        )
        text = path.read_text()
        self.assertIn("tags: [ciflow/inductor-cu126/205566]", text)
        self.assertEqual(text.count("github.event.pull_request.number == 205566"), 2)
        for clause in (
            "github.repository_id == '65600975'",
            "github.event.pull_request.user.id == 283013143",
            "github.event.pull_request.head.repo.id == 1372177501",
            f"github.event.pull_request.head.ref == '{probe.HEAD_REF}'",
        ):
            self.assertEqual(text.count(clause), 2)
        self.assertIn("uses: ./.github/workflows/_check-ci.yml", text)
        self.assertIn("needs: check-ci", text)
        self.assertIn(
            "ref: ${{ github.event.pull_request.head.sha || github.sha }}", text
        )
        self.assertIn("persist-credentials: false", text)
        self.assertIn("@ececac1a45f3b08a01d2dd070d28d111c5fe6722", text)
        self.assertEqual(text.count("AWS_MAX_ATTEMPTS:"), 1)
        self.assertIn("      AWS_MAX_ATTEMPTS: '1'\n", text)
        self.assertLess(text.index("AWS_MAX_ATTEMPTS:"), text.index("    steps:"))
        self.assertIn("disable-retry: true", text)
        policy_text = text.split(
            "          inline-session-policy: >-\n            ", 1
        )[1].splitlines()[0]
        self.assertEqual(
            hashlib.sha256((policy_text + "\n").encode()).hexdigest(),
            probe.POLICY_SHA256,
        )
        policy = json.loads(policy_text)
        deny, sts, bucket, obj = policy["Statement"]
        self.assertEqual((deny["Effect"], deny["Resource"]), ("Deny", "*"))
        self.assertEqual(
            sts, {"Effect": "Allow", "Action": "sts:GetCallerIdentity", "Resource": "*"}
        )
        self.assertEqual(bucket["Resource"], "arn:aws:s3:::" + probe.BUCKET)
        self.assertEqual(
            obj,
            {
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::" + probe.BUCKET + "/" + probe.OBJECT_NAME,
            },
        )
        allowed = set(bucket["Action"]) | {"sts:GetCallerIdentity", "s3:GetObject"}
        self.assertEqual(set(deny["NotAction"]), allowed)
        self.assertTrue(
            all(x.startswith(("s3:Get", "s3:List", "sts:Get")) for x in allowed)
        )
        for forbidden in (
            "role-chaining:",
            "web-identity-token-file:",
            "aws-access-key-id:",
            "managed-session-policies:",
            "environment:",
            "workflow_dispatch:",
        ):
            self.assertNotIn(forbidden, text)

    def test_signer_matches_independent_botocore(self):
        expected = SIGNATURES
        self.assertEqual(set(expected), {"identity", *probe.OPERATIONS})
        now = datetime.datetime(2026, 11, 17, 18, 0, tzinfo=datetime.UTC)
        for operation, signature in expected.items():
            signed = probe.sign(probe.request_spec(operation), CREDS, now)
            self.assertEqual(signed["authorization"].split("Signature=")[1], signature)
            self.assertEqual(signed["x-amz-security-token"], CREDS[2])

    def test_no_exception_or_response_leak(self):
        def fail(_operation):
            raise RuntimeError("NEVER_PRINT")

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            out = probe.run_probe(fail)
        self.assertEqual(stream.getvalue(), "")
        self.assertTrue(out["probe_error"])
        self.assertNotIn("NEVER_PRINT", json.dumps(out))

    def test_cli_result_contract(self):
        for name, overrides, status, complete, count in (
            ("complete", {}, 0, True, 16),
            ("owner-unavailable", {"HeadBucket": (403, {}, b"")}, 0, False, 1),
            ("config-unavailable", {"GetBucketReplication": error()}, 0, True, 16),
            (
                "identity-rejected",
                {"identity": (200, {}, identity(role="other-role"))},
                1,
                None,
                0,
            ),
        ):
            with self.subTest(name=name):
                respond, calls = responder(overrides)
                counts = dict.fromkeys(("identity", *probe.OPERATIONS), 0)

                def invoke(operation):
                    counts[operation] += 1
                    return respond(operation)

                stream = io.StringIO()
                with (
                    mock.patch.object(probe, "runtime_inputs", return_value=CREDS),
                    mock.patch.object(
                        probe,
                        "Transport",
                        return_value=mock.Mock(request=invoke, counts=counts),
                    ),
                    mock.patch.object(probe.resource, "setrlimit"),
                    mock.patch.object(probe.signal, "signal"),
                    mock.patch.object(probe.signal, "alarm"),
                    contextlib.redirect_stdout(stream),
                ):
                    self.assertEqual(probe.main(), status)
                out = json.loads(stream.getvalue())
                self.assertEqual(out["probe_error"], bool(status))
                self.assertEqual(out["identity_matches"], not status)
                self.assertEqual(out.get("all_metadata_requests_attempted"), complete)
                self.assertEqual(out["request_counts"]["identity"], 1)
                self.assertEqual(len(out["operations"]), count)
                self.assertEqual(len(calls), count + 1)
                self.assertFalse(out["write_authorized"])
                self.assertTrue(out["collector_cleanup_confirmed"])
                if name == "config-unavailable":
                    self.assertEqual(
                        out["operations"]["GetBucketReplication"]["state"],
                        "unavailable",
                    )

        stream = io.StringIO()
        with (
            mock.patch.object(
                probe, "runtime_inputs", side_effect=ValueError("NEVER_PRINT")
            ),
            mock.patch.object(probe, "Transport") as transport,
            mock.patch.object(probe.resource, "setrlimit"),
            mock.patch.object(probe.signal, "signal"),
            mock.patch.object(probe.signal, "alarm"),
            contextlib.redirect_stdout(stream),
        ):
            self.assertEqual(probe.main(), 1)
        transport.assert_not_called()
        self.assertEqual(
            json.loads(stream.getvalue()),
            {
                "probe_error": True,
                "write_authorized": False,
                "collector_cleanup_confirmed": True,
            },
        )


SIGNATURES = {
    "GetBucketEncryption": "3b8f74f916a9298b9f37ccb29f0f59a11336b424240a27f1a28fc496c8e44ebd",
    "GetBucketLifecycleConfiguration": "a5a9bdab78be83bb6264bf5b7b24784bba55f925627fec11a59b774fb5a65cfe",
    "GetBucketLocation": "ef3776af60b988e575bb9951b414683abc69ae43f0b875fcb1de4d51c51d786c",
    "GetBucketLogging": "d0f2e37e56b41553f2a5e78771730877eef84bb67918789dce0e1140e01b8bf2",
    "GetBucketNotificationConfiguration": "24c2645a7254a533756735ab2ee2bcd97166b2b07eff7e598e1b29b2cba51a8e",
    "GetBucketOwnershipControls": "3cef318a6d5c588b6bd4c5ad109fd78055267541109b0fc1c6e8a6db0ac7c048",
    "GetBucketPolicy": "acd866c7e4d8ff89372b0ea24461278f7a18a9d4e10135c44a5064f2cb874f7d",
    "GetBucketPolicyStatus": "8516337eada63180d86776443064153b7ce1609bf5eedd3aeb1053defd657bea",
    "GetBucketReplication": "bbed049a5da0117763bea6b9a9d0f563fa18ba8bc23eb9387e5dd45876dca742",
    "GetBucketVersioning": "2139b9b02fe99ce44567b0fc65bc28245b5e28505d62e8bf94177ef80da12dfe",
    "GetObjectLockConfiguration": "e42ddf6995a0456e6a0d7fb363e3129e0fe40aae95d90ef1276e9ed611fc7ab3",
    "GetPublicAccessBlock": "08437e88bd7739ce7d6edad17348fea08db190bde5a79bac7e47779ea39053b6",
    "HeadBucket": "fcc62f15e764151861d40577189dfa8581925b78b0ec59d4c528c6abf938449a",
    "HeadObject": "3520a56d2fcb3d4f29136bba3f2275a34e82958851940fd0d12990d01983ae57",
    "ListObjectVersions": "06c048d6f0cb2357b665104d42f8ce5509011b7cdff07a0c1a59cf5ea934804e",
    "ListObjectsV2": "7b748243da551808f93eb0384d04d98cfce4aa39530e9a843df48252790929df",
    "identity": "89db0060d8785b3945410ea72fb4297d3e3c74edfe0ecac59a9782ce25c23d77",
}

if __name__ == "__main__":
    unittest.main()
