# Owner(s): ["module: ci"]
"""Offline canary controls; synthetic responses are not deployment evidence."""

import contextlib
import datetime
import importlib.util
import io
import json
import unittest
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "probe", Path(__file__).with_name("ci_release_role_probe.py")
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
ROLE = probe.ROLES[0]
CREDS = ("ASIA" + "A" * 16, "secret" * 5, "token" * 8)


def identity(role=ROLE, account=probe.ACCOUNT):
    return f'<GetCallerIdentityResponse xmlns="{probe.STS_NS[1:-1]}"><GetCallerIdentityResult><Account>{account}</Account><Arn>arn:aws:sts::{account}:assumed-role/{role}/ci-test</Arn></GetCallerIdentityResult></GetCallerIdentityResponse>'.encode()


def error(code="AccessDenied"):
    return (
        403,
        f'<ErrorResponse xmlns="{probe.IAM_NS[1:-1]}"><Error><Code>{code}</Code><Message>NEVER_PRINT</Message></Error></ErrorResponse>'.encode(),
    )


def role_response(role=ROLE, policy='{"Version":"2012-10-17","Statement":[]}'):
    return f'<GetRoleResponse xmlns="{probe.IAM_NS[1:-1]}"><GetRoleResult><Role><RoleName>{role}</RoleName><Arn>{probe.role_arn(role)}</Arn><AssumeRolePolicyDocument>{urllib.parse.quote(policy)}</AssumeRolePolicyDocument></Role></GetRoleResult></GetRoleResponse>'.encode()


def simulation(operation, decision="implicitDeny"):
    _, _, target, action = probe.SIMULATIONS[operation]
    return f'<SimulatePrincipalPolicyResponse xmlns="{probe.IAM_NS[1:-1]}"><SimulatePrincipalPolicyResult><EvaluationResults><member><EvalActionName>{action}</EvalActionName><EvalResourceName>{target}</EvalResourceName><EvalDecision>{decision}</EvalDecision></member></EvaluationResults><IsTruncated>false</IsTruncated></SimulatePrincipalPolicyResult></SimulatePrincipalPolicyResponse>'.encode()


def environment():
    return dict(
        GITHUB_REPOSITORY="pytorch/pytorch",
        GITHUB_EVENT_NAME="pull_request",
        GITHUB_REF=f"refs/pull/{probe.PR_NUMBER}/merge",
        CANARY_PR_NUMBER=probe.PR_NUMBER,
        CANARY_PR_AUTHOR="ptdbot",
        CANARY_HEAD_REPO="ptdbot/pytorch",
        CANARY_HEAD_REF=probe.HEAD_REF,
        CANARY_ROLE_NAME=ROLE,
        ACTIONS_ID_TOKEN_REQUEST_URL="https://fixture.invalid/never-called",
        ACTIONS_ID_TOKEN_REQUEST_TOKEN="NONSECRET-FIXTURE",
        AWS_EC2_METADATA_DISABLED="true",
        AWS_CONFIG_FILE="/dev/null",
        AWS_SHARED_CREDENTIALS_FILE="/dev/null",
    )


def responder(overrides=None):
    calls = []
    overrides = overrides or {}

    def invoke(operation, role, credentials, unsigned=False):
        if role != ROLE or credentials != CREDS:
            raise ValueError("fixture scope")
        calls.append((operation, unsigned))
        if (operation, unsigned) in overrides:
            return overrides[(operation, unsigned)]
        if operation == "identity":
            return 200, identity()
        if unsigned:
            return error("MissingAuthenticationToken")
        if operation == "get-role":
            return 200, role_response()
        return 200, simulation(operation)

    return invoke, calls


class TestReleaseRoleProbe(unittest.TestCase):
    def test_exact_request_allowlist(self):
        allowed_actions = {"GetCallerIdentity", "GetRole", "SimulatePrincipalPolicy"}
        self.assertEqual(len(probe.SIMULATIONS), 9)
        for role in probe.ROLES:
            for operation in ("identity", "get-role", *probe.SIMULATIONS):
                service, host, headers, body = probe.request_spec(operation, role)
                fields = dict(
                    urllib.parse.parse_qsl(body.decode(), strict_parsing=True)
                )
                self.assertIn(fields["Action"], allowed_actions)
                self.assertEqual(
                    headers,
                    {
                        "content-type": "application/x-www-form-urlencoded; charset=utf-8"
                    },
                )
                if operation == "identity":
                    self.assertEqual(
                        (service, host, fields),
                        (
                            "sts",
                            "sts.us-east-1.amazonaws.com",
                            {"Action": "GetCallerIdentity", "Version": "2011-06-15"},
                        ),
                    )
                    continue
                self.assertEqual((service, host), ("iam", "iam.amazonaws.com"))
                if operation == "get-role":
                    self.assertEqual(
                        fields,
                        {
                            "Action": "GetRole",
                            "RoleName": role,
                            "Version": "2010-05-08",
                        },
                    )
                    continue
                _, _, target, action = probe.SIMULATIONS[operation]
                self.assertEqual(
                    fields,
                    {
                        "Action": "SimulatePrincipalPolicy",
                        "PolicySourceArn": probe.role_arn(role),
                        "ActionNames.member.1": action,
                        "ResourceArns.member.1": target,
                        "MaxItems": "1",
                        "Version": "2010-05-08",
                    },
                )
                self.assertIn(target, probe.OBJECTS.values())
                self.assertIn(action, probe.ACTIONS.values())
        for role in ("arc", "", None, ROLE + "/other"):
            with self.assertRaisesRegex(ValueError, "^role scope$"):
                probe.request_spec("identity", role)
        for operation in (
            "PutObject",
            "DeleteObject",
            "ListRoles",
            "AssumeRole",
            "simulate-other-put",
            "",
        ):
            with self.assertRaisesRegex(ValueError, "^operation$"):
                probe.request_spec(operation, ROLE)

    def test_success_and_denial_boundaries(self):
        invoke, calls = responder()
        out = probe.run_probe(ROLE, CREDS, invoke)
        self.assertFalse(out["probe_error"])
        self.assertTrue(
            out["identity_matches"]
            and out["negative_control_passed"]
            and out["role_read_allowed"]
        )
        self.assertRegex(out["trust_policy_sha256"], "^[0-9a-f]{64}$")
        self.assertEqual(
            out["simulations"], dict.fromkeys(probe.SIMULATIONS, "implicitDeny")
        )
        self.assertEqual(
            calls,
            [
                ("identity", False),
                ("get-role", True),
                ("get-role", False),
                *((op, False) for op in probe.SIMULATIONS),
            ],
        )
        first = next(iter(probe.SIMULATIONS))
        invoke, calls = responder(
            {("get-role", False): error(), (first, False): error()}
        )
        out = probe.run_probe(ROLE, CREDS, invoke)
        self.assertFalse(out["probe_error"])
        self.assertTrue(
            out["role_read_access_denied"] and out["simulation_access_denied"]
        )
        self.assertEqual(out["simulations"], {})
        self.assertEqual(calls[-1], (first, False))

    def test_identity_and_negative_control(self):
        for data in (
            identity(account="308535385114"),
            identity(role=probe.ROLES[1]),
            identity() + b"<extra/>",
            b"<!DOCTYPE test>",
            b"<wrong/>",
        ):
            invoke, calls = responder({("identity", False): (200, data)})
            self.assertTrue(probe.run_probe(ROLE, CREDS, invoke)["probe_error"])
            self.assertEqual(calls, [("identity", False)])
        invoke, calls = responder({("get-role", True): (200, role_response())})
        self.assertTrue(probe.run_probe(ROLE, CREDS, invoke)["probe_error"])
        self.assertNotIn(("get-role", False), calls)

    def test_role_policy_schema(self):
        for data in (
            role_response(probe.ROLES[1]),
            role_response(policy="[]"),
            role_response(policy='{"Statement":[],"Statement":[]}'),
            role_response().replace(b"</Role>", b"<RoleName>other</RoleName></Role>"),
        ):
            out = probe.run_probe(
                ROLE, CREDS, responder({("get-role", False): (200, data)})[0]
            )
            self.assertTrue(out["probe_error"])
            self.assertFalse(out["role_read_allowed"])
            self.assertEqual(out["simulations"], {})

    def test_simulation_schema(self):
        operation = next(iter(probe.SIMULATIONS))
        good = simulation(operation)
        bad = (
            good.replace(b"s3:GetObject", b"s3:ListBucket"),
            good.replace(probe.OBJECT_NAME.encode(), b"other.whl"),
            good.replace(b"implicitDeny", b"unknown"),
            good.replace(b"<IsTruncated>false", b"<IsTruncated>true"),
            good.replace(b"</EvaluationResults>", b"<member/></EvaluationResults>"),
            good.replace(
                b"</SimulatePrincipalPolicyResult>",
                b"<Marker>opaque</Marker></SimulatePrincipalPolicyResult>",
            ),
            b"<!DOCTYPE test>",
            b"X" * (probe.LIMIT + 1),
        )
        for data in bad:
            out = probe.run_probe(
                ROLE, CREDS, responder({(operation, False): (200, data)})[0]
            )
            self.assertTrue(out["probe_error"])
            self.assertEqual(out["simulations"], {})
        for decision in probe.DECISIONS:
            self.assertEqual(
                probe.simulation_decision(simulation(operation, decision), operation),
                decision,
            )

    def test_exact_context_and_credentials(self):
        env = environment()
        self.assertTrue(probe.preflight(env))
        credentials = dict(zip(probe.CREDENTIAL_KEYS, CREDS))
        self.assertEqual(probe.runtime_inputs(dict(env, **credentials)), (ROLE, CREDS))
        for key, value in (
            ("GITHUB_EVENT_NAME", "push"),
            ("GITHUB_REPOSITORY", "ptdbot/pytorch"),
            ("CANARY_PR_NUMBER", "123"),
            ("CANARY_HEAD_REF", "other"),
            ("CANARY_HEAD_REPO", "other/pytorch"),
            ("CANARY_PR_AUTHOR", "d4l3k"),
            ("GITHUB_REF", "refs/heads/main"),
        ):
            with self.assertRaisesRegex(ValueError, "^workflow scope$"):
                probe.preflight(dict(env, **{key: value}))
        tag = dict(env, GITHUB_EVENT_NAME="push", GITHUB_REF=probe.TAG)
        self.assertTrue(probe.preflight(tag))
        for ref in (
            probe.TAG + "0",
            "refs/heads/main",
            "refs/tags/ciflow/trunk/" + probe.PR_NUMBER,
        ):
            with self.assertRaisesRegex(ValueError, "^workflow scope$"):
                probe.preflight(dict(tag, GITHUB_REF=ref))
        with self.assertRaisesRegex(ValueError, "^credentials$"):
            probe.runtime_inputs(
                dict(env, **dict(credentials, AWS_SESSION_TOKEN="x\ny"))
            )
        with self.assertRaisesRegex(ValueError, "^role scope$"):
            probe.preflight(dict(env, CANARY_ROLE_NAME="arc"))

    def test_provider_gate(self):
        env = environment()
        for missing in (
            "ACTIONS_ID_TOKEN_REQUEST_URL",
            "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        ):
            absent = dict(env)
            del absent[missing]
            self.assertFalse(probe.preflight(absent))
            with self.assertRaisesRegex(ValueError, "^OIDC unavailable$"):
                probe.runtime_inputs(
                    dict(absent, **dict(zip(probe.CREDENTIAL_KEYS, CREDS)))
                )
        overrides = (
            *probe.CREDENTIAL_KEYS,
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_ROLE_ARN",
            "AWS_ROLE_SESSION_NAME",
            "AWS_PROFILE",
            "AWS_DEFAULT_PROFILE",
            "AWS_SDK_LOAD_CONFIG",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_AUTHORIZATION_TOKEN",
            "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
            "AWS_ENDPOINT_URL",
            "AWS_ENDPOINT_URL_STS",
            "AWS_CA_BUNDLE",
            "AWS_LOGIN_CACHE_DIRECTORY",
            "AWS_CONFIG_FILE",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_EC2_METADATA_DISABLED",
            "AWS_FUTURE_PROVIDER",
            "INPUT_WEB_IDENTITY_TOKEN_FILE",
            "INPUT_ROLE_CHAINING",
            "INPUT_AWS_ACCESS_KEY_ID",
            "INPUT_FUTURE_PROVIDER",
            *probe.TLS_OVERRIDES,
        )
        for key in overrides:
            with self.assertRaisesRegex(ValueError, "^configuration$", msg=key):
                probe.preflight(dict(env, **{key: "NEVER_PRINT"}))
        for key in ("AWS_REGION", "AWS_DEFAULT_REGION"):
            self.assertTrue(probe.preflight(dict(env, **{key: probe.REGION})))
            with self.assertRaisesRegex(ValueError, "^configuration$"):
                probe.preflight(dict(env, **{key: "us-west-2"}))

    def test_workflow_guards(self):
        text = (
            Path(__file__)
            .parents[1]
            .joinpath("workflows/ci-security-release-role-read-canary.yml")
            .read_text()
        )
        self.assertIn(f"tags: [ciflow/inductor-cu126/{probe.PR_NUMBER}]", text)
        self.assertNotIn("ciflow/inductor-cu126/*", text)
        for clause in (
            f"github.event.pull_request.number == {probe.PR_NUMBER}",
            "github.event.pull_request.user.login == 'ptdbot'",
            "github.event.pull_request.head.repo.full_name == 'ptdbot/pytorch'",
            f"github.event.pull_request.head.ref == '{probe.HEAD_REF}'",
            f"github.ref == '{probe.TAG}'",
        ):
            self.assertEqual(text.count(clause), 2)
        self.assertIn("uses: ./.github/workflows/_check-ci.yml", text)
        self.assertIn("needs: check-ci", text)
        self.assertIn(
            "ref: ${{ github.event.pull_request.head.sha || github.sha }}", text
        )
        for role in probe.ROLES:
            self.assertEqual(text.count("          - " + role + "\n"), 1)
        action = text.split("      - name: Assume exact release role\n", 1)[1].split(
            "      - name:", 1
        )[0]
        self.assertIn("if: steps.provider.outputs.oidc_requestable == 'true'", action)
        self.assertEqual(text.count("uses: aws-actions/configure-aws-credentials@"), 1)
        for item in (
            "@ececac1a45f3b08a01d2dd070d28d111c5fe6722",
            "role-to-assume: arn:aws:iam::749337293305:role/${{ matrix.role }}",
            "audience: sts.amazonaws.com",
            "unset-current-credentials: true",
            "disable-retry: true",
        ):
            self.assertIn(item, action)
        for forbidden in (
            "role-chaining:",
            "web-identity-token-file:",
            "aws-access-key-id:",
            "inline-session-policy:",
            "managed-session-policies:",
            "environment:",
        ):
            self.assertNotIn(forbidden, text)

    def test_signer_matches_botocore_fixture(self):
        expected = {
            "identity": "89db0060d8785b3945410ea72fb4297d3e3c74edfe0ecac59a9782ce25c23d77",
            "get-role": "217081ccbdc19bffe716d78f8045c870c7f3943db8341ab53b899b84f683ab0a",
            "simulate-test-put": "5c80b0c9e90152e51e56dfc563a3537d47b624ef5642197ab44c04573607207c",
        }
        now = datetime.datetime(2026, 11, 17, 18, 0, tzinfo=datetime.UTC)
        for operation, signature in expected.items():
            service, host, headers, body = probe.request_spec(operation, ROLE)
            signed = probe.sign(service, host, headers, body, CREDS, now)
            self.assertEqual(signed["authorization"].split("Signature=")[1], signature)
            self.assertEqual(signed["x-amz-security-token"], CREDS[2])

    def test_no_response_or_exception_leak(self):
        def fail(*args, **kwargs):
            raise RuntimeError("NEVER_PRINT")

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            out = probe.run_probe(ROLE, CREDS, fail)
        self.assertEqual(stream.getvalue(), "")
        self.assertNotIn("NEVER_PRINT", json.dumps(out))
        self.assertTrue(out["probe_error"])

    def test_transport_scope(self):
        class Response:
            status = 200

            def read(self, size):
                return b"{}"

        connection = mock.Mock()
        connection.getresponse.return_value = Response()
        with mock.patch.object(
            probe.http.client, "HTTPSConnection", return_value=connection
        ) as factory:
            probe.call("get-role", ROLE, CREDS, unsigned=True)
        self.assertEqual(factory.call_args.args, ("iam.amazonaws.com",))
        args, kwargs = connection.request.call_args
        self.assertEqual(args, ("POST", "/"))
        self.assertEqual(set(kwargs["headers"]), {"content-type"})
        self.assertTrue(connection.close.called)
        with self.assertRaisesRegex(ValueError, "^unsigned operation$"):
            probe.call("simulate-test-put", ROLE, CREDS, unsigned=True)
        with self.assertRaisesRegex(ValueError, "^XML boundary$"):
            probe.xml(b"<!ENTITY example>")
        with self.assertRaisesRegex(ET.ParseError, "syntax error"):
            probe.xml(b"not xml")


if __name__ == "__main__":
    unittest.main()
