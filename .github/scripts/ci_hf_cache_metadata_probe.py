# Owner(s): ["module: ci"]
"""Temporary, fixed-scope HF-cache metadata probe. Never writes cloud state."""

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
ROLE = "gha_workflow_hf-cache-write"
PR_NUMBER = "205603"
HEAD_REF = "gh/d4l3k/351/head"
BASE_REF = "gh/d4l3k/351/base"
TAG = f"refs/tags/ciflow/inductor-cu126/{PR_NUMBER}"
WORKFLOW = (
    ".github/workflows/ci-security-hf-cache-writer-metadata-repair-read-canary.yml"
)
BOT = "pytorch-bot[bot]"
BUCKET = "pytorch-hf-model-cache-meta-prod-aws-ue1"
OBJECT_NAME = "release-audit-hf-cache-20261117-c1e8e9b4"
S3_HOST = f"{BUCKET}.s3.{REGION}.amazonaws.com"
LIMIT = 262144
MAX_KEYS = 8
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
LOGGING_NS = "{http://doc.s3.amazonaws.com/2006-03-01}"
STS_NS = "{https://sts.amazonaws.com/doc/2011-06-15/}"
POLICY_SHA256 = "ceca8a3c3d4d0590d57988c57fe7bfa479518d9ef7d2fdfe2b7dbb3fe387b05f"
CREDENTIAL_KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
FIXED_AWS_ENV = {
    "AWS_REGION": REGION,
    "AWS_DEFAULT_REGION": REGION,
    "AWS_MAX_ATTEMPTS": "1",
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
CONFIG_QUERIES = {
    "GetBucketLocation": "location",
    "GetBucketVersioning": "versioning",
    "GetObjectLockConfiguration": "object-lock",
    "GetBucketReplication": "replication",
    "GetBucketNotificationConfiguration": "notification",
    "GetBucketLifecycleConfiguration": "lifecycle",
    "GetBucketLogging": "logging",
    "GetBucketPolicy": "policy",
    "GetBucketPolicyStatus": "policyStatus",
    "GetPublicAccessBlock": "publicAccessBlock",
    "GetBucketOwnershipControls": "ownershipControls",
    "GetBucketEncryption": "encryption",
}
OPERATIONS = (
    "HeadBucket",
    *CONFIG_QUERIES,
    "HeadObject",
    "ListObjectsV2",
    "ListObjectVersions",
)
ABSENT_CONFIG = {
    "GetObjectLockConfiguration": "ObjectLockConfigurationNotFoundError",
    "GetBucketReplication": "ReplicationConfigurationNotFoundError",
    "GetBucketLifecycleConfiguration": "NoSuchLifecycleConfiguration",
    "GetBucketPolicy": "NoSuchBucketPolicy",
    "GetPublicAccessBlock": "NoSuchPublicAccessBlockConfiguration",
    "GetBucketOwnershipControls": "OwnershipControlsNotFoundError",
    "GetBucketEncryption": "ServerSideEncryptionConfigurationNotFoundError",
}
DENIED = {"AccessDenied", "AccessDeniedException", "UnauthorizedOperation"}
AUTH_ERRORS = {
    "InvalidAccessKeyId",
    "InvalidToken",
    "ExpiredToken",
    "SignatureDoesNotMatch",
    "AuthorizationHeaderMalformed",
    "InvalidSecurity",
    "MissingAuthenticationToken",
}


def request_spec(operation):
    """Return only literal endpoints, methods, keys, queries and payloads."""
    if operation == "identity":
        return (
            "sts",
            "sts.us-east-1.amazonaws.com",
            "POST",
            "/",
            "",
            {"content-type": "application/x-www-form-urlencoded; charset=utf-8"},
            b"Action=GetCallerIdentity&Version=2011-06-15",
        )
    if operation not in OPERATIONS:
        raise ValueError("operation")
    method, path, fields = "GET", "/", []
    if operation == "HeadBucket":
        method = "HEAD"
    elif operation == "HeadObject":
        method, path = "HEAD", "/" + OBJECT_NAME
    elif operation == "ListObjectsV2":
        fields = [("list-type", "2"), ("max-keys", "8"), ("prefix", OBJECT_NAME)]
    elif operation == "ListObjectVersions":
        fields = [("versions", ""), ("max-keys", "8"), ("prefix", OBJECT_NAME)]
    else:
        fields = [(CONFIG_QUERIES[operation], "")]
    query = urllib.parse.urlencode(sorted(fields), quote_via=urllib.parse.quote)
    headers = {
        "x-amz-expected-bucket-owner": ACCOUNT,
        "x-amz-content-sha256": hashlib.sha256(b"").hexdigest(),
    }
    return "s3", S3_HOST, method, path, query, headers, b""


def sign(spec, credentials, now):
    service, host, method, path, query, initial, body = spec
    access, secret, token = credentials
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    day = stamp[:8]
    headers = dict(
        initial, host=host, **{"x-amz-date": stamp, "x-amz-security-token": token}
    )
    names = ";".join(sorted(headers))
    canonical = "".join(
        key + ":" + headers[key].strip() + "\n" for key in sorted(headers)
    )
    request = "\n".join(
        (method, path, query, canonical, names, hashlib.sha256(body).hexdigest())
    )
    scope = day + "/" + REGION + "/" + service + "/aws4_request"
    text = "\n".join(
        ("AWS4-HMAC-SHA256", stamp, scope, hashlib.sha256(request.encode()).hexdigest())
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


class Transport:
    def __init__(self, credentials):
        self.credentials = credentials
        self.counts = dict.fromkeys(("identity", *OPERATIONS), 0)

    def request(self, operation):
        spec = request_spec(operation)
        # The pinned credentials action performs the other identity read.
        if self.counts[operation] >= 1:
            raise ValueError("request budget")
        self.counts[operation] += 1
        _, host, method, path, query, _, body = spec
        headers = sign(spec, self.credentials, datetime.datetime.now(datetime.UTC))
        connection = http.client.HTTPSConnection(
            host, timeout=5, context=ssl.create_default_context()
        )
        try:
            connection.request(
                method,
                path + ("?" + query if query else ""),
                body=body,
                headers=headers,
            )
            response = connection.getresponse()
            if response.status in range(300, 400):
                raise ValueError("redirect")
            data = response.read(LIMIT + 1)
            if len(data) > LIMIT:
                raise ValueError("response size")
            result_headers = {}
            for name, value in response.getheaders():
                name = name.lower()
                if name not in {
                    "x-amz-bucket-region",
                    "x-amz-request-id",
                    "x-amz-delete-marker",
                    "x-amz-version-id",
                    "content-length",
                }:
                    continue
                if name in result_headers:
                    raise ValueError("duplicate response header")
                result_headers[name] = value
            return response.status, result_headers, data
        finally:
            connection.close()


def xml(data):
    if len(data) > LIMIT or b"<!" in data or b"\0" in data:
        raise ValueError("XML boundary")
    root = ET.fromstring(data.decode("utf-8"))
    if sum(1 for _ in root.iter()) > 4096:
        raise ValueError("XML size")
    return root


def one(parent, name):
    values = parent.findall(name)
    if len(values) != 1:
        raise ValueError("XML shape")
    return values[0]


def identity_matches(data):
    root = xml(data)
    if root.tag != STS_NS + "GetCallerIdentityResponse":
        return False
    row = one(root, STS_NS + "GetCallerIdentityResult")
    return (
        one(row, STS_NS + "Account").text == ACCOUNT
        and re.fullmatch(
            rf"arn:aws:sts::{ACCOUNT}:assumed-role/{ROLE}/[A-Za-z0-9+=,.@_-]{{2,64}}",
            one(row, STS_NS + "Arn").text or "",
        )
        is not None
    )


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def config_root(data, name, allowed, namespaces=(S3_NS,)):
    root = xml(data)
    namespace = next((ns for ns in namespaces if root.tag == ns + name), None)
    if namespace is None or any(
        child.tag not in {namespace + item for item in allowed} for child in root
    ):
        raise ValueError("S3 XML shape")
    return root


def leaf(root, name, allowed, optional=False):
    rows = root.findall(S3_NS + name)
    if optional and not rows:
        return None
    if len(rows) != 1 or list(rows[0]) or rows[0].text not in allowed:
        raise ValueError("S3 value")
    return rows[0].text


def integer(text):
    if text is None or not re.fullmatch(r"0|[1-9][0-9]{0,19}", text):
        raise ValueError("integer")
    return int(text)


def error_code(data):
    root = xml(data)
    if root.tag not in ("Error", S3_NS + "Error"):
        raise ValueError("S3 error shape")
    prefix = S3_NS if root.tag.startswith(S3_NS) else ""
    return one(root, prefix + "Code").text


def status_projection(operation, status, headers, data):
    if operation == "HeadObject":
        if headers.get("x-amz-delete-marker") not in (None, "true", "false"):
            raise ValueError("delete marker header")
        version = headers.get("x-amz-version-id")
        if version is not None and (not version or len(version) > 2048):
            raise ValueError("version header")
    if status == 200:
        return None
    if operation in ("HeadBucket", "HeadObject"):
        if data:
            raise ValueError("HEAD body")
        if operation == "HeadObject" and status == 404:
            if headers.get("x-amz-delete-marker") == "true":
                return {"state": "delete-marker-present"}
            if "x-amz-version-id" in headers:
                return {"state": "unavailable", "reason": "head-version-metadata"}
            return {"state": "current-key-not-found"}
        return {"state": "unavailable", "reason": "head-status"}
    code = error_code(data)
    if status == 404 and ABSENT_CONFIG.get(operation) == code:
        return {"state": "not-configured"}
    if status in (400, 401, 403) and code in DENIED:
        return {"state": "unavailable", "reason": "access-denied"}
    if status in (400, 401, 403) and code in AUTH_ERRORS:
        return {"state": "unavailable", "reason": "authentication"}
    return {"state": "unavailable", "reason": "service-status"}


def listing(data, versions):
    name = "ListVersionsResult" if versions else "ListBucketResult"
    rows = ("Version", "DeleteMarker") if versions else ("Contents",)
    allowed = {"Name", "Prefix", "MaxKeys", "IsTruncated", *rows}
    allowed.update(("KeyMarker", "VersionIdMarker") if versions else ("KeyCount",))
    root = config_root(data, name, allowed)
    if (
        one(root, S3_NS + "Name").text != BUCKET
        or one(root, S3_NS + "Prefix").text != OBJECT_NAME
    ):
        raise ValueError("list scope")
    if integer(one(root, S3_NS + "MaxKeys").text) != MAX_KEYS:
        raise ValueError("list limit")
    leaf(root, "IsTruncated", {"false"})
    if versions:
        for marker in ("KeyMarker", "VersionIdMarker"):
            found = root.findall(S3_NS + marker)
            if len(found) > 1 or (found and found[0].text):
                raise ValueError("list marker")
    entries = [(kind, row) for kind in rows for row in root.findall(S3_NS + kind)]
    if len(entries) > MAX_KEYS:
        raise ValueError("list count")
    if not versions and integer(one(root, S3_NS + "KeyCount").text) != len(entries):
        raise ValueError("key count")
    exact = delete_markers = 0
    for kind, row in entries:
        key = one(row, S3_NS + "Key").text
        if not isinstance(key, str) or not key.startswith(OBJECT_NAME):
            raise ValueError("listed key")
        if versions:
            version = one(row, S3_NS + "VersionId").text
            if not version or len(version) > 2048:
                raise ValueError("version shape")
            leaf(row, "IsLatest", {"true", "false"})
        exact += key == OBJECT_NAME
        delete_markers += kind == "DeleteMarker" and key == OBJECT_NAME
    return {
        "state": "complete",
        "exact_entries": exact,
        "other_prefix_entries": len(entries) - exact,
        "exact_delete_markers": delete_markers,
    }


def project_success(operation, headers, data):
    if operation in ("HeadBucket", "HeadObject"):
        if data:
            raise ValueError("HEAD body")
        if operation == "HeadBucket":
            if headers.get("x-amz-bucket-region") != REGION:
                raise ValueError("bucket region")
            return {"state": "owner-region-matched"}
        if headers.get("x-amz-delete-marker") == "true":
            raise ValueError("successful delete marker")
        return {
            "state": "current-key-present",
            "bytes": integer(headers.get("content-length")),
            "version_id_present": "x-amz-version-id" in headers,
            "delete_marker": headers.get("x-amz-delete-marker") == "true",
        }
    if operation in ("ListObjectsV2", "ListObjectVersions"):
        return listing(data, operation == "ListObjectVersions")
    if operation == "GetBucketPolicy":
        value = json.loads(data, object_pairs_hook=object_pairs)
        if not isinstance(value, dict) or not set(value) <= {
            "Version",
            "Id",
            "Statement",
        }:
            raise ValueError("policy shape")
        rows = value.get("Statement")
        rows = [rows] if isinstance(rows, dict) else rows
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("policy statements")
        return {"state": "present-unreviewed", "statements": len(rows)}
    if operation == "GetBucketLocation":
        root = config_root(data, "LocationConstraint", set())
        if root.text not in (None, "", REGION):
            raise ValueError("bucket region")
        return {"state": "region-matched"}
    if operation == "GetBucketVersioning":
        root = config_root(data, "VersioningConfiguration", {"Status", "MfaDelete"})
        value = leaf(root, "Status", {"Enabled", "Suspended"}, optional=True)
        mfa = leaf(root, "MfaDelete", {"Enabled", "Disabled"}, optional=True)
        if value is None and mfa is not None:
            raise ValueError("versioning shape")
        return {"state": value or "never-enabled", "mfa_delete": mfa or "absent"}
    if operation == "GetObjectLockConfiguration":
        root = config_root(
            data, "ObjectLockConfiguration", {"ObjectLockEnabled", "Rule"}
        )
        leaf(root, "ObjectLockEnabled", {"Enabled"})
        rules = root.findall(S3_NS + "Rule")
        if len(rules) > 1:
            raise ValueError("lock rules")
        return {"state": "enabled", "default_rule_present": bool(rules)}
    if operation == "GetBucketReplication":
        root = config_root(data, "ReplicationConfiguration", {"Role", "Rule"})
        if not one(root, S3_NS + "Role").text:
            raise ValueError("replication role")
        rules = root.findall(S3_NS + "Rule")
        if not rules:
            raise ValueError("replication rules")
        return {"state": "configured", "rules": len(rules)}
    if operation == "GetBucketNotificationConfiguration":
        root = config_root(
            data,
            "NotificationConfiguration",
            {
                "TopicConfiguration",
                "QueueConfiguration",
                "CloudFunctionConfiguration",
                "EventBridgeConfiguration",
            },
        )
        return {
            "state": "configured" if len(root) else "not-configured",
            "rules": len(root),
        }
    if operation == "GetBucketLifecycleConfiguration":
        root = config_root(data, "LifecycleConfiguration", {"Rule"})
        if not len(root):
            raise ValueError("lifecycle rules")
        return {"state": "configured", "rules": len(root)}
    if operation == "GetBucketLogging":
        root = config_root(
            data, "BucketLoggingStatus", {"LoggingEnabled"}, (S3_NS, LOGGING_NS)
        )
        namespace = root.tag.removesuffix("BucketLoggingStatus")
        if len(root) > 1 or any(
            not node.tag.startswith(namespace) for node in root.iter()
        ):
            raise ValueError("logging shape")
        return {"state": "configured" if len(root) else "not-configured"}
    if operation == "GetBucketPolicyStatus":
        root = config_root(data, "PolicyStatus", {"IsPublic"})
        return {
            "state": "present",
            "public": leaf(root, "IsPublic", {"true", "false"}) == "true",
        }
    if operation == "GetPublicAccessBlock":
        fields = (
            "BlockPublicAcls",
            "IgnorePublicAcls",
            "BlockPublicPolicy",
            "RestrictPublicBuckets",
        )
        root = config_root(data, "PublicAccessBlockConfiguration", set(fields))
        values = [leaf(root, field, {"true", "false"}) == "true" for field in fields]
        return {"state": "present", "all_enabled": all(values)}
    if operation == "GetBucketOwnershipControls":
        root = config_root(data, "OwnershipControls", {"Rule"})
        row = one(root, S3_NS + "Rule")
        value = leaf(
            row,
            "ObjectOwnership",
            {"BucketOwnerPreferred", "ObjectWriter", "BucketOwnerEnforced"},
        )
        return {"state": value}
    if operation == "GetBucketEncryption":
        root = config_root(data, "ServerSideEncryptionConfiguration", {"Rule"})
        algorithms = set()
        for row in root:
            default = one(row, S3_NS + "ApplyServerSideEncryptionByDefault")
            algorithms.add(
                leaf(default, "SSEAlgorithm", {"AES256", "aws:kms", "aws:kms:dsse"})
            )
        if not algorithms:
            raise ValueError("encryption rules")
        return {"state": "configured", "algorithms": sorted(algorithms)}
    raise ValueError("operation")


def project(operation, response):
    request_spec(operation)
    status, headers, data = response
    if operation == "identity" or len(data) > LIMIT:
        raise ValueError("projection boundary")
    out = status_projection(operation, status, headers, data)
    if out is None:
        out = project_success(operation, headers, data)
    out["response_sha256"] = hashlib.sha256(data).hexdigest()
    request_id = headers.get("x-amz-request-id", "")
    if request_id and re.fullmatch(r"[A-Za-z0-9+/=_-]{1,256}", request_id):
        out["request_id"] = request_id
    return out


def run_probe(invoke):
    """Collect redacted metadata using invoke(operation) -> (status, headers, body).

    The callback accepts only "identity" and OPERATIONS. It must make at most
    one HTTP request per operation, without retries or redirects, and return
    an integer status, lowercase allowlisted headers, and at most LIMIT bytes.
    Transport errors or invalid responses must raise. Transport enforces this
    contract. The initial exact-account/role check is the collector's only STS
    identity request; the pinned credentials action makes the other (total <=2).

    The result has identity_matches, operations, probe_error and
    write_authorized. operations maps attempted, successfully projected S3
    operation names to redacted state dictionaries. Exceptions set probe_error
    and failure_stage without exposing the exception or raw response; the
    all_metadata_requests_attempted field can then be absent. Otherwise that
    field reports whether all 16 responses were projected, not their success.
    Consumers must require identity_matches and all_metadata_requests_attempted
    to be true, then inspect every expected operation's state. Unavailable and
    present-unreviewed do not establish safe configuration. write_authorized is
    always false: it states this collector's scope, not the role's permissions.
    """
    out = {
        "identity_matches": False,
        "operations": {},
        "probe_error": False,
        "write_authorized": False,
    }
    stage = "identity"
    try:
        status, _, data = invoke("identity")
        if status != 200 or not identity_matches(data):
            raise ValueError("identity")
        out["identity_matches"] = True
        for operation in OPERATIONS:
            stage = operation
            result = project(operation, invoke(operation))
            out["operations"][operation] = result
            if operation == "HeadBucket" and result["state"] != "owner-region-matched":
                break
            if operation == "GetBucketLocation" and result["state"] != "region-matched":
                break
        out["all_metadata_requests_attempted"] = len(out["operations"]) == len(
            OPERATIONS
        )
    except Exception:
        out.update(probe_error=True, failure_stage=stage)
    return out


def scope_valid(env):
    if (
        PR_NUMBER == "0"
        or env.get("GITHUB_REPOSITORY") != "pytorch/pytorch"
        or env.get("GITHUB_REPOSITORY_ID") != "65600975"
    ):
        return False
    return (
        env.get("GITHUB_EVENT_NAME") == "push"
        and env.get("GITHUB_REF") == TAG
        and env.get("GITHUB_ACTOR") == BOT
        and env.get("GITHUB_ACTOR_ID") == "54816060"
        and env.get("GITHUB_TRIGGERING_ACTOR") == BOT
        and env.get("CANARY_SENDER_ID") == "54816060"
        and env.get("GITHUB_RUN_ATTEMPT") == "1"
        and env.get("GITHUB_WORKFLOW_REF") == f"pytorch/pytorch/{WORKFLOW}@{TAG}"
        and re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", "")) is not None
        and env.get("GITHUB_WORKFLOW_SHA") == env.get("GITHUB_SHA")
        and env.get("RUNNER_ENVIRONMENT") == "github-hosted"
        and env.get("RUNNER_OS") == "Linux"
    )


def source_matches(pr, sha):
    return (
        pr["number"] == int(PR_NUMBER)
        and pr["state"] == "open"
        and pr["draft"] is True
        and pr["user"]["login"] == "d4l3k"
        and pr["user"]["id"] == 909104
        and pr["head"]["repo"]["full_name"] == "pytorch/pytorch"
        and pr["head"]["repo"]["id"] == 65600975
        and pr["head"]["ref"] == HEAD_REF
        and pr["head"]["sha"] == sha
        and pr["base"]["repo"]["full_name"] == "pytorch/pytorch"
        and pr["base"]["repo"]["id"] == 65600975
        and pr["base"]["ref"] == BASE_REF
    )


def validate_source(env):
    """One fixed, nonredirecting GitHub read before obtaining cloud credentials."""
    if not scope_valid(env):
        raise ValueError("workflow scope")
    token = env.get("CANARY_GITHUB_TOKEN", "")
    if not token or "\r" in token or "\n" in token:
        raise ValueError("source credentials")
    connection = http.client.HTTPSConnection(
        "api.github.com", timeout=5, context=ssl.create_default_context()
    )
    try:
        connection.request(
            "GET",
            f"/repos/pytorch/pytorch/pulls/{PR_NUMBER}",
            headers={
                "authorization": "Bearer " + token,
                "accept": "application/vnd.github+json",
                "x-github-api-version": "2022-11-28",
                "user-agent": "pytorch-hf-cache-metadata-audit",
            },
        )
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("source response")
        data = response.read(LIMIT + 1)
        if len(data) > LIMIT:
            raise ValueError("source response size")
        pr = json.loads(data, object_pairs_hook=object_pairs)
        if not source_matches(pr, env["GITHUB_SHA"]):
            raise ValueError("source identity")
    finally:
        connection.close()


def validate_environment(env, after_action=False):
    if not scope_valid(env) or env.get("CANARY_ROLE_NAME") != ROLE:
        raise ValueError("workflow scope")
    for key in (
        "AWS_EC2_METADATA_DISABLED",
        "AWS_MAX_ATTEMPTS",
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
        validate_source(os.environ)
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
    return credentials


def deadline(_signum, _frame):
    raise TimeoutError("deadline")


def main():
    """Print one JSON result and exit int(probe_error).

    Exit 0 alone does not imply complete or available metadata. The outer error
    fallback can omit identity_matches, operations and completion. request_counts,
    when present, counts collector attempts only, not the credentials action's
    one identity request. collector_cleanup_confirmed records local completion;
    it is not evidence of cloud permissions or remote cleanup.
    """
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(120)
        transport = Transport(runtime_inputs(os.environ))
        out = run_probe(transport.request)
        out["request_counts"] = transport.counts
    except Exception:
        out = {"probe_error": True, "write_authorized": False}
    finally:
        signal.alarm(0)
    out["collector_cleanup_confirmed"] = True
    print(json.dumps(out, sort_keys=True))
    return int(out["probe_error"])


if __name__ == "__main__":
    if sys.argv[1:] == ["--preflight"]:
        raise SystemExit(preflight_main())
    if sys.argv[1:]:
        raise SystemExit("unsupported invocation")
    raise SystemExit(main())
