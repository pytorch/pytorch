"""Temporary HF-cache role read canary. Never prints AWS response bodies."""

import datetime
import hashlib
import hmac
import http.client
import json
import os
import re
import resource
import signal
import ssl
import sys
import urllib.parse
import xml.etree.ElementTree as ET


ACCOUNT = "308535385114"
REGION = "us-east-1"
ROLES = ("gha_workflow_hf-cache-write",)
PR_NUMBER = "205434"
HEAD_REF = "d4l3k/hf-cache-role-audit"
TAG = f"refs/tags/ciflow/inductor-cu126/{PR_NUMBER}"
OBJECT_NAME = "release-audit-hf-cache-20261117-c1e8e9b4"
OBJECTS = {
    "staging": f"arn:aws:s3:::pytorch-hf-model-cache-meta-staging-aws-ue1/{OBJECT_NAME}",
    "prod": f"arn:aws:s3:::pytorch-hf-model-cache-meta-prod-aws-ue1/{OBJECT_NAME}",
}
ACTIONS = {"get": "s3:GetObject", "put": "s3:PutObject", "delete": "s3:DeleteObject"}
SIMULATIONS = {
    f"simulate-{channel}-{name}": (channel, name, arn, action)
    for channel, arn in OBJECTS.items()
    for name, action in ACTIONS.items()
}
LIMIT = 262144
IAM_NS = "{https://iam.amazonaws.com/doc/2010-05-08/}"
STS_NS = "{https://sts.amazonaws.com/doc/2011-06-15/}"
DENIED = {"AccessDenied", "AccessDeniedException", "UnauthorizedOperation"}
AUTH_ERRORS = DENIED | {
    "MissingAuthenticationToken",
    "InvalidClientTokenId",
    "SignatureDoesNotMatch",
    "IncompleteSignature",
    "InvalidSignatureException",
}
DECISIONS = {"allowed", "explicitDeny", "implicitDeny"}
CREDENTIAL_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
FIXED_AWS_ENV = {
    "AWS_REGION": REGION,
    "AWS_DEFAULT_REGION": REGION,
    "AWS_EC2_METADATA_DISABLED": "true",
    "AWS_CONFIG_FILE": "/dev/null",
    "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
}
TLS_OVERRIDES = {
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "NODE_TLS_REJECT_UNAUTHORIZED",
    "NODE_OPTIONS",
    "OPENSSL_CONF",
    "OPENSSL_MODULES",
}


def role_arn(role):
    if role not in ROLES:
        raise ValueError("role scope")
    return f"arn:aws:iam::{ACCOUNT}:role/{role}"


def request_spec(operation, role):
    arn = role_arn(role)
    headers = {"content-type": "application/x-www-form-urlencoded; charset=utf-8"}
    if operation == "identity":
        return (
            "sts",
            "sts.us-east-1.amazonaws.com",
            headers,
            b"Action=GetCallerIdentity&Version=2011-06-15",
        )
    if operation == "get-role":
        fields = [("Action", "GetRole"), ("RoleName", role)]
    elif operation in SIMULATIONS:
        _, _, target, action = SIMULATIONS[operation]
        fields = [
            ("Action", "SimulatePrincipalPolicy"),
            ("PolicySourceArn", arn),
            ("ActionNames.member.1", action),
            ("ResourceArns.member.1", target),
            ("MaxItems", "1"),
        ]
    else:
        raise ValueError("operation")
    fields.append(("Version", "2010-05-08"))
    return "iam", "iam.amazonaws.com", headers, urllib.parse.urlencode(fields).encode()


def sign(service, host, headers, body, credentials, now):
    access, secret, token = credentials
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    day = stamp[:8]
    headers = dict(
        headers, host=host, **{"x-amz-date": stamp, "x-amz-security-token": token}
    )
    names = ";".join(sorted(headers))
    canonical = "".join(k + ":" + headers[k].strip() + "\n" for k in sorted(headers))
    request = (
        "POST\n/\n\n"
        + canonical
        + "\n"
        + names
        + "\n"
        + hashlib.sha256(body).hexdigest()
    )
    scope = day + "/" + REGION + "/" + service + "/aws4_request"
    text = (
        "AWS4-HMAC-SHA256\n"
        + stamp
        + "\n"
        + scope
        + "\n"
        + hashlib.sha256(request.encode()).hexdigest()
    )
    key = ("AWS4" + secret).encode()
    for part in (day, REGION, service, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, text.encode(), hashlib.sha256).hexdigest()
    headers["authorization"] = (
        "AWS4-HMAC-SHA256 Credential="
        + access
        + "/"
        + scope
        + ", SignedHeaders="
        + names
        + ", Signature="
        + signature
    )
    return headers


def call(operation, role, credentials, unsigned=False):
    if unsigned and operation != "get-role":
        raise ValueError("unsigned operation")
    service, host, headers, body = request_spec(operation, role)
    if not unsigned:
        headers = sign(
            service,
            host,
            headers,
            body,
            credentials,
            datetime.datetime.now(datetime.UTC),
        )
    connection = http.client.HTTPSConnection(
        host, timeout=5, context=ssl.create_default_context()
    )
    try:
        connection.request("POST", "/", body=body, headers=headers)
        response = connection.getresponse()
        data = response.read(LIMIT + 1)
        if len(data) > LIMIT or response.status in range(300, 400):
            raise ValueError("response boundary")
        return response.status, data
    finally:
        connection.close()


def xml(data):
    if len(data) > LIMIT or b"<!" in data:
        raise ValueError("XML boundary")
    return ET.fromstring(data)


def one(parent, name):
    values = parent.findall(name)
    if len(values) != 1:
        raise ValueError("XML shape")
    return values[0]


def identity_matches(data, role):
    root = xml(data)
    if root.tag != STS_NS + "GetCallerIdentityResponse":
        return False
    row = one(root, STS_NS + "GetCallerIdentityResult")
    return (
        one(row, STS_NS + "Account").text == ACCOUNT
        and re.fullmatch(
            rf"arn:aws:sts::{ACCOUNT}:assumed-role/{re.escape(role)}/[A-Za-z0-9+=,.@_-]{{2,64}}",
            one(row, STS_NS + "Arn").text or "",
        )
        is not None
    )


def iam_result(data, operation):
    root = xml(data)
    if root.tag != IAM_NS + operation + "Response":
        raise ValueError("IAM response")
    return one(root, IAM_NS + operation + "Result")


def iam_error(data):
    root = xml(data)
    if root.tag != IAM_NS + "ErrorResponse":
        raise ValueError("IAM error")
    return one(one(root, IAM_NS + "Error"), IAM_NS + "Code").text


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def trust_policy_hash(data, role):
    row = one(iam_result(data, "GetRole"), IAM_NS + "Role")
    if (
        one(row, IAM_NS + "Arn").text != role_arn(role)
        or one(row, IAM_NS + "RoleName").text != role
    ):
        raise ValueError("role identity")
    policy = json.loads(
        urllib.parse.unquote(one(row, IAM_NS + "AssumeRolePolicyDocument").text or ""),
        object_pairs_hook=object_pairs,
    )
    if not isinstance(policy, dict) or not isinstance(
        policy.get("Statement"), (dict, list)
    ):
        raise ValueError("policy shape")
    return hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def simulation_decision(data, operation):
    result = iam_result(data, "SimulatePrincipalPolicy")
    truncated = result.findall(IAM_NS + "IsTruncated")
    if (
        len(truncated) > 1
        or (truncated and truncated[0].text != "false")
        or result.findall(IAM_NS + "Marker")
    ):
        raise ValueError("simulation pagination")
    row = one(one(result, IAM_NS + "EvaluationResults"), IAM_NS + "member")
    _, _, target, action = SIMULATIONS[operation]
    if (
        one(row, IAM_NS + "EvalActionName").text != action
        or one(row, IAM_NS + "EvalResourceName").text != target
    ):
        raise ValueError("simulation scope")
    decision = one(row, IAM_NS + "EvalDecision").text
    if decision not in DECISIONS:
        raise ValueError("simulation decision")
    return decision


def initial_output(role):
    role_arn(role)
    out = dict.fromkeys(
        (
            "oidc_requestable",
            "role_assumed",
            "identity_matches",
            "negative_control_passed",
            "role_read_allowed",
            "role_read_access_denied",
            "simulation_access_denied",
            "collector_cleanup_confirmed",
            "probe_error",
        ),
        False,
    )
    out.update(role=role, trust_policy_sha256=None, simulations={})
    return out


def run_probe(role, credentials, invoke=call):
    out = initial_output(role)
    try:
        status, data = invoke("identity", role, credentials)
        if status != 200 or not identity_matches(data, role):
            raise ValueError("identity")
        out.update(role_assumed=True, identity_matches=True)
        status, data = invoke("get-role", role, credentials, unsigned=True)
        if status not in (400, 401, 403) or iam_error(data) not in AUTH_ERRORS:
            raise ValueError("negative control")
        out["negative_control_passed"] = True
        status, data = invoke("get-role", role, credentials)
        if status == 200:
            out["trust_policy_sha256"] = trust_policy_hash(data, role)
            out["role_read_allowed"] = True
        elif status in (400, 401, 403) and iam_error(data) in DENIED:
            out["role_read_access_denied"] = True
        else:
            raise ValueError("get role")
        for operation in SIMULATIONS:
            status, data = invoke(operation, role, credentials)
            if status in (400, 401, 403) and iam_error(data) in DENIED:
                out["simulation_access_denied"] = True
                break
            if status != 200:
                raise ValueError("simulate")
            out["simulations"][operation] = simulation_decision(data, operation)
    except Exception:
        out["probe_error"] = True
    return out


def scope_valid(env):
    if (
        env.get("GITHUB_REPOSITORY") != "pytorch/pytorch"
        or env.get("GITHUB_REPOSITORY_ID") != "65600975"
    ):
        return False
    if env.get("GITHUB_EVENT_NAME") == "push":
        return env.get("GITHUB_REF") == TAG
    return (
        env.get("GITHUB_EVENT_NAME") == "pull_request"
        and env.get("GITHUB_REF") == f"refs/pull/{PR_NUMBER}/merge"
        and env.get("CANARY_PR_NUMBER") == PR_NUMBER
        and env.get("CANARY_PR_AUTHOR") == "ptdbot"
        and env.get("CANARY_PR_AUTHOR_ID") == "283013143"
        and env.get("CANARY_HEAD_REPO") == "ptdbot/pytorch"
        and env.get("CANARY_HEAD_REPO_ID") == "1372177501"
        and env.get("CANARY_HEAD_REF") == HEAD_REF
    )


def validate_environment(env, after_action=False):
    if not scope_valid(env):
        raise ValueError("workflow scope")
    role_arn(env.get("CANARY_ROLE_NAME"))
    for key in (
        "AWS_EC2_METADATA_DISABLED",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
    ):
        if env.get(key) != FIXED_AWS_ENV[key]:
            raise ValueError("configuration")
    for key, value in env.items():
        if not value:
            continue
        if key in TLS_OVERRIDES or key.startswith("INPUT_"):
            raise ValueError("configuration")
        if not key.startswith("AWS_"):
            continue
        if key in FIXED_AWS_ENV and value == FIXED_AWS_ENV[key]:
            continue
        if after_action and key in CREDENTIAL_KEYS:
            continue
        raise ValueError("configuration")


def preflight(env):
    validate_environment(env)
    return bool(
        env.get("ACTIONS_ID_TOKEN_REQUEST_URL")
        and env.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    )


def preflight_main():
    try:
        available = preflight(os.environ)
    except Exception:
        print("provider_gate_passed=false")
        return 1
    line = "oidc_requestable=" + str(available).lower()
    print(line)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(line + "\n")
    return 0


def runtime_inputs(env):
    validate_environment(env, after_action=True)
    if not (
        env.get("ACTIONS_ID_TOKEN_REQUEST_URL")
        and env.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    ):
        raise ValueError("OIDC unavailable")
    credentials = tuple(env.get(key, "") for key in CREDENTIAL_KEYS)
    if (
        not re.fullmatch(r"ASIA[A-Z0-9]{16}", credentials[0])
        or not 16 <= len(credentials[1]) <= 128
        or not 16 <= len(credentials[2]) <= 16384
        or any("\n" in value or "\r" in value for value in credentials)
    ):
        raise ValueError("credentials")
    return env["CANARY_ROLE_NAME"], credentials


def deadline(_signum, _frame):
    raise TimeoutError("deadline")


def main():
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(90)
        role, credentials = runtime_inputs(os.environ)
        out = run_probe(role, credentials)
        out["oidc_requestable"] = True
    except Exception:
        out = {"probe_error": True}
    finally:
        signal.alarm(0)
    out["collector_cleanup_confirmed"] = True
    print(json.dumps(out, sort_keys=True))
    return int(out["probe_error"])


if __name__ == "__main__":
    raise SystemExit(preflight_main() if sys.argv[1:] == ["--preflight"] else main())
