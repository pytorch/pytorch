"""Temporary exact-repository read canary. Never prints AWS response bodies."""

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
import xml.etree.ElementTree as ET


ACCOUNT = "308535385114"
REGION = "us-east-1"
REPOSITORY = "pytorch/ci-image"
ARN = f"arn:aws:ecr:{REGION}:{ACCOUNT}:repository/{REPOSITORY}"
PREFIX = "AmazonEC2ContainerRegistry_V20150921."
LIMIT = 262144
DENIED = {"AccessDenied", "AccessDeniedException", "UnauthorizedException"}
AUTH_ERRORS = DENIED | {
    "MissingAuthenticationTokenException",
    "UnrecognizedClientException",
    "InvalidSignatureException",
}
MUTABILITY = {
    "MUTABLE",
    "IMMUTABLE",
    "IMMUTABLE_WITH_EXCLUSION",
    "MUTABLE_WITH_EXCLUSION",
}
ENCRYPTION = {"AES256", "KMS", "KMS_DSSE"}


def request_spec(operation):
    if operation == "identity":
        return (
            "sts",
            "sts.us-east-1.amazonaws.com",
            {"content-type": "application/x-www-form-urlencoded; charset=utf-8"},
            b"Action=GetCallerIdentity&Version=2011-06-15",
        )
    if operation == "describe":
        body = {"registryId": ACCOUNT, "repositoryNames": [REPOSITORY]}
        target = "DescribeRepositories"
    elif operation == "policy":
        body = {"registryId": ACCOUNT, "repositoryName": REPOSITORY}
        target = "GetRepositoryPolicy"
    else:
        raise ValueError("operation")
    return (
        "ecr",
        "api.ecr.us-east-1.amazonaws.com",
        {"content-type": "application/x-amz-json-1.1", "x-amz-target": PREFIX + target},
        json.dumps(body, separators=(",", ":")).encode(),
    )


def sign(service, host, headers, body, credentials, now):
    access, secret, token = credentials
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    day = stamp[:8]
    headers = dict(
        headers, host=host, **{"x-amz-date": stamp, "x-amz-security-token": token}
    )
    names = ";".join(sorted(headers))
    canonical = "".join(k + ":" + headers[k].strip() + "\n" for k in sorted(headers))
    digest = hashlib.sha256(body).hexdigest()
    request = "POST\n/\n\n" + canonical + "\n" + names + "\n" + digest
    scope = day + "/" + REGION + "/" + service + "/aws4_request"
    string = (
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
    signature = hmac.new(key, string.encode(), hashlib.sha256).hexdigest()
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


def call(operation, credentials, unsigned=False):
    if unsigned and operation != "describe":
        raise ValueError("unsigned operation")
    service, host, headers, body = request_spec(operation)
    if not unsigned:
        headers = sign(
            service,
            host,
            headers,
            body,
            credentials,
            datetime.datetime.now(datetime.UTC),
        )
    # HTTPSConnection uses no environment proxy, provider chain or redirects.
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


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def parse_json(data):
    if len(data) > LIMIT:
        raise ValueError("response size")
    result = json.loads(data, object_pairs_hook=object_pairs)
    if not isinstance(result, dict):
        raise ValueError("response shape")
    return result


def error_code(data):
    value = parse_json(data).get("__type", "")
    return value.rsplit("#", 1)[-1].split(":", 1)[0] if isinstance(value, str) else ""


def identity_matches(data):
    if len(data) > LIMIT or b"<!" in data:
        return False
    root = ET.fromstring(data)
    namespace = "{https://sts.amazonaws.com/doc/2011-06-15/}"
    if root.tag != namespace + "GetCallerIdentityResponse":
        return False
    rows = root.findall(namespace + "GetCallerIdentityResult")
    if len(rows) != 1:
        return False
    account = rows[0].findall(namespace + "Account")
    arn = rows[0].findall(namespace + "Arn")
    return (
        len(account) == len(arn) == 1
        and account[0].text == ACCOUNT
        and re.fullmatch(
            r"arn:aws:sts::308535385114:assumed-role/arc/[A-Za-z0-9+=,.@_-]{2,64}",
            arn[0].text or "",
        )
        is not None
    )


def repository_matches(value):
    return (
        isinstance(value, dict)
        and value.get("registryId") == ACCOUNT
        and value.get("repositoryName") == REPOSITORY
        and value.get("repositoryArn") == ARN
    )


def initial_output():
    result = dict.fromkeys(
        (
            "oidc_requestable",
            "arc_assumed",
            "identity_matches",
            "describe_allowed",
            "describe_access_denied",
            "repository_matches",
            "policy_read_allowed",
            "policy_absent",
            "policy_access_denied",
            "negative_control_passed",
            "cleanup_confirmed",
            "probe_error",
        ),
        False,
    )
    result.update(mutability_category="unknown", encryption_category="unknown")
    return result


def run_probe(credentials, invoke=call):
    out = initial_output()
    try:
        status, body = invoke("identity", credentials)
        if status != 200 or not identity_matches(body):
            raise ValueError("identity")
        out["identity_matches"] = True
        status, body = invoke("describe", credentials, unsigned=True)
        if status not in (400, 401, 403) or error_code(body) not in AUTH_ERRORS:
            raise ValueError("negative control")
        out["negative_control_passed"] = True
        status, body = invoke("describe", credentials)
        if status != 200:
            if status in (400, 401, 403) and error_code(body) in DENIED:
                out["describe_access_denied"] = True
                return out
            raise ValueError("describe")
        payload = parse_json(body)
        rows = payload.get("repositories")
        if (
            payload.get("nextToken") is not None
            or not isinstance(rows, list)
            or len(rows) != 1
            or not repository_matches(rows[0])
        ):
            raise ValueError("repository")
        mode = rows[0].get("imageTagMutability")
        encryption = rows[0].get("encryptionConfiguration", {}).get("encryptionType")
        if mode not in MUTABILITY or encryption not in ENCRYPTION:
            raise ValueError("metadata")
        out.update(
            describe_allowed=True,
            repository_matches=True,
            mutability_category=mode,
            encryption_category=encryption,
        )
        status, body = invoke("policy", credentials)
        if status != 200:
            code = error_code(body)
            if status in (400, 401, 403) and code in DENIED:
                out["policy_access_denied"] = True
                return out
            if status == 400 and code == "RepositoryPolicyNotFoundException":
                out["policy_absent"] = True
                return out
            raise ValueError("policy")
        payload = parse_json(body)
        if (
            payload.get("registryId") != ACCOUNT
            or payload.get("repositoryName") != REPOSITORY
            or not isinstance(payload.get("policyText"), str)
        ):
            raise ValueError("policy identity")
        parse_json(payload["policyText"].encode())
        out["policy_read_allowed"] = True
    except Exception:
        out["probe_error"] = True
    return out


TAG = "refs/tags/ciflow/inductor-cu126/204870"
CREDENTIAL_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
FIXED_AWS_ENV = {
    "AWS_REGION": REGION,
    "AWS_DEFAULT_REGION": REGION,
    "AWS_EC2_METADATA_DISABLED": "true",
    "AWS_CONFIG_FILE": "/dev/null",
    "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
}
TLS_OVERRIDES = {
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS", "NODE_TLS_REJECT_UNAUTHORIZED", "NODE_OPTIONS",
    "OPENSSL_CONF", "OPENSSL_MODULES",
}


def scope_valid(env):
    if env.get("GITHUB_REPOSITORY") != "pytorch/pytorch":
        return False
    if env.get("GITHUB_EVENT_NAME") == "push":
        return env.get("GITHUB_REF") == TAG
    return (
        env.get("GITHUB_EVENT_NAME") == "pull_request"
        and env.get("GITHUB_REF") == "refs/pull/204870/merge"
        and env.get("CANARY_PR_NUMBER") == "204870"
        and env.get("CANARY_PR_AUTHOR") == "ptdbot"
        and env.get("CANARY_HEAD_REPO") == "ptdbot/pytorch"
        and env.get("CANARY_HEAD_REF") == "ci-ecr-read-20261113"
    )


def validate_environment(env, after_action=False):
    if not scope_valid(env):
        raise ValueError("workflow scope")
    for key in ("AWS_EC2_METADATA_DISABLED", "AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE"):
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
    return bool(env.get("ACTIONS_ID_TOKEN_REQUEST_URL") and env.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN"))


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
    if not (env.get("ACTIONS_ID_TOKEN_REQUEST_URL") and env.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")):
        raise ValueError("OIDC unavailable")
    credentials = tuple(env.get(key, "") for key in CREDENTIAL_KEYS)
    if (
        not re.fullmatch(r"ASIA[A-Z0-9]{16}", credentials[0])
        or not 16 <= len(credentials[1]) <= 128
        or not 16 <= len(credentials[2]) <= 16384
        or any("\n" in value or "\r" in value for value in credentials)
    ):
        raise ValueError("credentials")
    return credentials


def deadline(_signum, _frame):
    raise TimeoutError("deadline")


def main():
    out = initial_output()
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(60)
        out = run_probe(runtime_inputs(os.environ))
        out.update(oidc_requestable=True, arc_assumed=out["identity_matches"])
    except Exception:
        out["probe_error"] = True
    finally:
        signal.alarm(0)
        out["cleanup_confirmed"] = True
    print(json.dumps(out, sort_keys=True))
    return int(out["probe_error"])


if __name__ == "__main__":
    raise SystemExit(preflight_main() if sys.argv[1:] == ["--preflight"] else main())
