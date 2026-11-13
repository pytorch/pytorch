# Owner(s): ["module: ci"]
"""Offline scope/parser tests; fixtures are not deployment evidence."""

import contextlib
import datetime
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "probe", Path(__file__).with_name("ci_arc_ecr_probe.py")
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
CREDS = ("ASIA" + "A" * 16, "secret" * 5, "token" * 8)
IDENTITY = b'<GetCallerIdentityResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><GetCallerIdentityResult><Account>308535385114</Account><Arn>arn:aws:sts::308535385114:assumed-role/arc/ci-test</Arn></GetCallerIdentityResult></GetCallerIdentityResponse>'
REPO = {
    "registryId": probe.ACCOUNT,
    "repositoryName": probe.REPOSITORY,
    "repositoryArn": probe.ARN,
    "imageTagMutability": "IMMUTABLE",
    "encryptionConfiguration": {"encryptionType": "AES256"},
}


def environment():
    return dict(
        GITHUB_REPOSITORY="pytorch/pytorch", GITHUB_EVENT_NAME="pull_request",
        GITHUB_REF="refs/pull/204870/merge", CANARY_PR_NUMBER="204870",
        CANARY_PR_AUTHOR="ptdbot", CANARY_HEAD_REPO="ptdbot/pytorch",
        CANARY_HEAD_REF="ci-ecr-read-20261113",
        ACTIONS_ID_TOKEN_REQUEST_URL="https://fixture.invalid/never-called",
        ACTIONS_ID_TOKEN_REQUEST_TOKEN="NONSECRET-FIXTURE",
        AWS_EC2_METADATA_DISABLED="true", AWS_CONFIG_FILE="/dev/null",
        AWS_SHARED_CREDENTIALS_FILE="/dev/null",
    )


def encoded(value):
    return json.dumps(value).encode()


def denied(code="AccessDeniedException"):
    return 400, encoded({"__type": code, "message": "NEVER_PRINT"})


def responder(describe=None, policy=None, identity=IDENTITY, unsigned=None):
    calls = []

    def invoke(operation, credentials, unsigned=False):
        calls.append((operation, unsigned))
        if credentials != CREDS:
            raise ValueError("unexpected credentials")
        if operation == "identity":
            return 200, identity
        if unsigned:
            return (
                denied("MissingAuthenticationTokenException")
                if unsigned_response is None
                else unsigned_response
            )
        if operation == "describe":
            return (
                (200, encoded({"repositories": [REPO]}))
                if describe is None
                else describe
            )
        if operation == "policy":
            return (
                denied("RepositoryPolicyNotFoundException")
                if policy is None
                else policy
            )
        raise ValueError("scope")

    unsigned_response = unsigned
    return invoke, calls


class TestProbe(unittest.TestCase):
    def test_exact_requests(self):
        self.assertEqual(
            probe.request_spec("identity")[1:],
            (
                "sts.us-east-1.amazonaws.com",
                {"content-type": "application/x-www-form-urlencoded; charset=utf-8"},
                b"Action=GetCallerIdentity&Version=2011-06-15",
            ),
        )
        for operation, name, field in (
            ("describe", "DescribeRepositories", "repositoryNames"),
            ("policy", "GetRepositoryPolicy", "repositoryName"),
        ):
            service, host, headers, body = probe.request_spec(operation)
            self.assertEqual(
                (service, host), ("ecr", "api.ecr.us-east-1.amazonaws.com")
            )
            self.assertEqual(
                headers,
                {
                    "content-type": "application/x-amz-json-1.1",
                    "x-amz-target": probe.PREFIX + name,
                },
            )
            self.assertEqual(
                json.loads(body),
                {
                    "registryId": probe.ACCOUNT,
                    field: [probe.REPOSITORY]
                    if operation == "describe"
                    else probe.REPOSITORY,
                },
            )
        for operation in ("GetAuthorizationToken", "list", "DescribeImages", ""):
            with self.assertRaisesRegex(ValueError, "^operation$"):
                probe.request_spec(operation)

    def test_success_and_denials(self):
        invoke, calls = responder()
        out = probe.run_probe(CREDS, invoke)
        self.assertFalse(out["probe_error"])
        self.assertTrue(
            out["describe_allowed"]
            and out["policy_absent"]
            and out["negative_control_passed"]
        )
        self.assertEqual(
            calls,
            [
                ("identity", False),
                ("describe", True),
                ("describe", False),
                ("policy", False),
            ],
        )
        invoke, calls = responder(describe=denied())
        out = probe.run_probe(CREDS, invoke)
        self.assertTrue(out["describe_access_denied"])
        self.assertFalse(out["probe_error"])
        self.assertNotIn(("policy", False), calls)
        out = probe.run_probe(CREDS, responder(policy=denied())[0])
        self.assertTrue(out["policy_access_denied"])

    def test_identity_scope(self):
        for data in (
            IDENTITY.replace(b"308535385114", b"391835788720"),
            IDENTITY.replace(b"/arc/", b"/other/"),
            IDENTITY + b"<extra/>",
            b"<!DOCTYPE test>",
            b"<arbitrary/>",
        ):
            invoke, calls = responder(identity=data)
            self.assertTrue(probe.run_probe(CREDS, invoke)["probe_error"])
            self.assertEqual(calls, [("identity", False)])

    def test_repository_schema(self):
        variants = [
            {"repositories": []},
            {"repositories": [REPO, REPO]},
            {"repositories": [REPO], "nextToken": "opaque"},
        ]
        for key, value in (
            ("registryId", "391835788720"),
            ("repositoryName", "other"),
            ("repositoryArn", "wrong"),
            ("imageTagMutability", "new"),
            ("encryptionConfiguration", {"encryptionType": "new"}),
            ("encryptionConfiguration", None),
        ):
            variants.append({"repositories": [dict(REPO, **{key: value})]})
        for payload in variants:
            invoke, calls = responder(describe=(200, encoded(payload)))
            out = probe.run_probe(CREDS, invoke)
            self.assertTrue(out["probe_error"])
            self.assertFalse(out["describe_allowed"])
            self.assertNotIn(("policy", False), calls)

    def test_policy_schema(self):
        correct = {
            "registryId": probe.ACCOUNT,
            "repositoryName": probe.REPOSITORY,
            "policyText": '{"Statement":[]}',
        }
        self.assertTrue(
            probe.run_probe(CREDS, responder(policy=(200, encoded(correct)))[0])[
                "policy_read_allowed"
            ]
        )
        for row in (
            dict(correct, repositoryName="other"),
            dict(correct, policyText="[]"),
            dict(correct, policyText='{"a":1,"a":2}'),
        ):
            self.assertTrue(
                probe.run_probe(CREDS, responder(policy=(200, encoded(row)))[0])[
                    "probe_error"
                ]
            )

    def test_unknown_error_and_unsigned_success(self):
        for value in (
            (302, b"NEVER_PRINT"),
            denied("ThrottlingException"),
            (200, b'{"repositories":[],"repositories":[]}'),
            (200, b"X" * (probe.LIMIT + 1)),
        ):
            self.assertTrue(
                probe.run_probe(CREDS, responder(describe=value)[0])["probe_error"]
            )
        invoke, calls = responder(unsigned=(200, encoded({"repositories": [REPO]})))
        self.assertTrue(probe.run_probe(CREDS, invoke)["probe_error"])
        self.assertNotIn(("describe", False), calls)

    def test_configuration_scope(self):
        env = environment()
        self.assertTrue(probe.preflight(env))
        credentials = dict(zip(probe.CREDENTIAL_KEYS, CREDS))
        self.assertEqual(probe.runtime_inputs(dict(env, **credentials)), CREDS)
        for key, value in (
            ("GITHUB_EVENT_NAME", "push"), ("GITHUB_REPOSITORY", "ptdbot/pytorch"),
            ("CANARY_PR_NUMBER", "204850"), ("CANARY_HEAD_REF", "other"),
            ("CANARY_HEAD_REPO", "other/pytorch"), ("CANARY_PR_AUTHOR", "d4l3k"),
            ("GITHUB_REF", "refs/pull/204850/merge"),
        ):
            with self.assertRaisesRegex(ValueError, "^workflow scope$"):
                probe.preflight(dict(env, **{key: value}))
        tag = dict(env, GITHUB_EVENT_NAME="push", GITHUB_REF=probe.TAG)
        self.assertTrue(probe.preflight(tag))
        for ref in (probe.TAG + "0", "refs/heads/main", "refs/tags/ciflow/trunk/204870"):
            with self.assertRaisesRegex(ValueError, "^workflow scope$"):
                probe.preflight(dict(tag, GITHUB_REF=ref))
        with self.assertRaisesRegex(ValueError, "^credentials$"):
            probe.runtime_inputs(dict(env, **dict(credentials, AWS_SESSION_TOKEN="x\ny")))

    def test_pre_action_provider_gate(self):
        env = environment()
        for missing in ("ACTIONS_ID_TOKEN_REQUEST_URL", "ACTIONS_ID_TOKEN_REQUEST_TOKEN"):
            absent = dict(env)
            del absent[missing]
            self.assertFalse(probe.preflight(absent))
            with self.assertRaisesRegex(ValueError, "^OIDC unavailable$"):
                probe.runtime_inputs(dict(absent, **dict(zip(probe.CREDENTIAL_KEYS, CREDS))))
        overrides = (
            *probe.CREDENTIAL_KEYS, "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN",
            "AWS_ROLE_SESSION_NAME", "AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_SDK_LOAD_CONFIG",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_AUTHORIZATION_TOKEN", "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
            "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_STS", "AWS_CA_BUNDLE", "AWS_LOGIN_CACHE_DIRECTORY",
            "AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE", "AWS_EC2_METADATA_DISABLED",
            "AWS_FUTURE_PROVIDER", "INPUT_WEB_IDENTITY_TOKEN_FILE", "INPUT_ROLE_CHAINING",
            "INPUT_AWS_ACCESS_KEY_ID", "INPUT_FUTURE_PROVIDER", *probe.TLS_OVERRIDES,
        )
        for key in overrides:
            with self.assertRaisesRegex(ValueError, "^configuration$", msg=key):
                probe.preflight(dict(env, **{key: "NEVER_PRINT"}))
        for key in ("AWS_REGION", "AWS_DEFAULT_REGION"):
            self.assertTrue(probe.preflight(dict(env, **{key: probe.REGION})))
            with self.assertRaisesRegex(ValueError, "^configuration$"):
                probe.preflight(dict(env, **{key: "us-west-2"}))

    def test_workflow_provider_and_exact_tag_guard(self):
        text = Path(__file__).parents[1].joinpath("workflows/ci-security-ecr-read-canary.yml").read_text()
        self.assertIn("tags: [ciflow/inductor-cu126/204870]", text)
        self.assertNotIn("ciflow/inductor-cu126/*", text)
        for clause in ("github.event.pull_request.number == 204870", "github.event.pull_request.user.login == 'ptdbot'", "github.event.pull_request.head.repo.full_name == 'ptdbot/pytorch'", "github.event.pull_request.head.ref == 'ci-ecr-read-20261113'", "github.ref == 'refs/tags/ciflow/inductor-cu126/204870'"):
            self.assertIn(clause, text)
        self.assertIn("ref: ${{ github.event.pull_request.head.sha || github.sha }}", text)
        self.assertIn("python3 -I .github/scripts/ci_arc_ecr_probe.py --preflight", text)
        action = text.split("      - name: Assume exact ordinary CI role\n", 1)[1].split("      - name:", 1)[0]
        self.assertIn("if: steps.provider.outputs.oidc_requestable == 'true'", action)
        self.assertEqual(text.count("uses: aws-actions/configure-aws-credentials@"), 1)
        for item in ("@ececac1a45f3b08a01d2dd070d28d111c5fe6722", "role-to-assume: arn:aws:iam::308535385114:role/arc", "audience: sts.amazonaws.com", "unset-current-credentials: true", "disable-retry: true"):
            self.assertIn(item, action)
        for forbidden in ("role-chaining:", "web-identity-token-file:", "aws-access-key-id:", "inline-session-policy:", "managed-session-policies:"):
            self.assertNotIn(forbidden, action)

    def test_signer(self):
        service, host, headers, body = probe.request_spec("describe")
        now = datetime.datetime(2026, 11, 13, 20, 0, tzinfo=datetime.UTC)
        signed = probe.sign(service, host, headers, body, CREDS, now)
        self.assertEqual(signed["x-amz-date"], "20261113T200000Z")
        self.assertEqual(signed["x-amz-security-token"], CREDS[2])
        self.assertIn("/20261113/us-east-1/ecr/aws4_request", signed["authorization"])
        self.assertEqual(
            signed["authorization"].split("Signature=")[1],
            "adaac3cffc0fcd81a0e665e35aa94d266c456a50ea549e5c5d23e11ebd40c1a3",
        )

    def test_no_response_or_exception_leak(self):
        def fail(*args, **kwargs):
            raise RuntimeError("NEVER_PRINT")

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            output = probe.run_probe(CREDS, fail)
        self.assertEqual(stream.getvalue(), "")
        self.assertNotIn("NEVER_PRINT", json.dumps(output))
        self.assertTrue(output["probe_error"])

    def test_https_scope_and_unsigned_headers(self):
        class Response:
            status = 200

            def read(self, size):
                self.size = size
                return b"{}"

        connection = mock.Mock()
        connection.getresponse.return_value = Response()
        with mock.patch.object(
            probe.http.client, "HTTPSConnection", return_value=connection
        ) as factory:
            probe.call("describe", CREDS, unsigned=True)
        self.assertEqual(factory.call_args.args, ("api.ecr.us-east-1.amazonaws.com",))
        args, kwargs = connection.request.call_args
        self.assertEqual(args, ("POST", "/"))
        self.assertEqual(set(kwargs["headers"]), {"content-type", "x-amz-target"})
        self.assertTrue(connection.close.called)
        with self.assertRaisesRegex(ValueError, "^unsigned operation$"):
            probe.call("policy", CREDS, unsigned=True)


if __name__ == "__main__":
    unittest.main()
