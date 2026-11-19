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
PR_NUMBER = "0"
HEAD_REF = "gh/d4l3k/0/head"
BASE_REF = "gh/d4l3k/0/base"
TAG = f"refs/tags/ciflow/inductor-cu126/{PR_NUMBER}"
WORKFLOW = ".github/workflows/ci-security-hf-cache-metadata-tail-read-canary.yml"
BOT = "pytorch-bot[bot]"
BUCKET = "pytorch-hf-model-cache-meta-prod-aws-ue1"
OBJECT_NAME = "release-audit-hf-cache-20261117-c1e8e9b4"
S3_HOST = f"{BUCKET}.s3.{REGION}.amazonaws.com"
LIMIT = 262144
MAX_KEYS = 8
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
STS_NS = "{https://sts.amazonaws.com/doc/2011-06-15/}"
POLICY_SHA256 = "c2818b33b6232e873dc911362d6e8d1ac79c7dda16d734a6d43625f747697c1a"
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
    "GetBucketPolicy": "NoSuchBucketPolicy",
    "GetPublicAccessBlock": "NoSuchPublicAccessBlockConfiguration",
    "GetBucketOwnershipControls": "OwnershipControlsNotFoundError",
    "GetBucketEncryption": "ServerSideEncryptionConfigurationNotFoundError",
}
EMPTY_HASH = hashlib.sha256(b"").hexdigest()
LOCAL_CODES = frozenset(
    (
        "response-limit",
        "xml-boundary",
        "xml-shape",
        "json-shape",
        "value-shape",
        "list-scope",
        "list-truncated",
        "list-count",
    )
)
FATAL_CODES = frozenset(
    (
        "source-boundary",
        "environment-boundary",
        "oidc-unavailable",
        "credentials",
        "identity",
        "owner-region",
        "request-boundary",
        "request-budget",
        "framing",
        "redirect",
        "tls",
        "network",
        "authentication",
        "deadline",
        "internal",
        "finalization",
    )
)


class GlobalFatalFailure(Exception):
    def __init__(self, code, stage="transport", response=None, projection=None):
        if code not in FATAL_CODES:
            raise RuntimeError("invalid fatal code")
        self.code, self.stage = code, stage
        self.response, self.projection = response, projection
        super().__init__(code)


class OperationLocalFailure(Exception):
    def __init__(self, code, stage="projection", response=None):
        if code not in LOCAL_CODES:
            raise RuntimeError("invalid local code")
        self.code, self.stage, self.response = code, stage, response
        super().__init__(code)


def require(condition, code="internal", stage="validation"):
    if not condition:
        raise GlobalFatalFailure(code, stage)


def local_require(condition, code="value-shape", stage="projection"):
    if not condition:
        raise OperationLocalFailure(code, stage)


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
        raise GlobalFatalFailure("request-boundary", "request")
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


HEADER_LIMIT = 65536
LINE_LIMIT = 8192
HEADER_COUNT = 100
TOKEN = rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+"
SAFE_HEADERS = frozenset(
    (
        "x-amz-bucket-region",
        "x-amz-request-id",
        "x-amz-delete-marker",
        "x-amz-version-id",
        "content-length",
    )
)
FORBIDDEN_TRAILERS = frozenset(
    (
        "content-length",
        "transfer-encoding",
        "host",
        "connection",
        "trailer",
        "upgrade",
        "authorization",
        "proxy-authorization",
        "content-encoding",
        *SAFE_HEADERS,
    )
)


def envelope(status, framing="unaccepted", headers=None):
    return {
        "http_status": status,
        "framing": framing,
        "response_complete": False,
        "response_bytes": None,
        "response_sha256": None,
        "request_id": (headers or {}).get("x-amz-request-id"),
        "body_limit_exceeded": False,
    }


def finish_response(metadata, body):
    metadata.update(
        response_complete=True,
        response_bytes=len(body),
        response_sha256=hashlib.sha256(body).hexdigest(),
    )
    return metadata


def wire_line(stream):
    line = stream.readline(LINE_LIMIT + 1)
    require(len(line) <= LINE_LIMIT and line.endswith(b"\r\n"), "framing")
    return line


def wire_headers(stream, trailer=False):
    total = count = 0
    values = {}
    while True:
        raw = wire_line(stream)
        total += len(raw)
        require(total <= HEADER_LIMIT, "framing")
        if raw == b"\r\n":
            return values
        count += 1
        require(count <= HEADER_COUNT, "framing")
        match = re.fullmatch(b"(" + TOKEN + rb"):([\t\x20-\x7e\x80-\xff]*)\r\n", raw)
        require(match is not None, "framing")
        name = match[1].decode("ascii").lower()
        value = match[2].decode("latin1").strip(" \t")
        if trailer:
            require(name not in FORBIDDEN_TRAILERS, "framing")
            continue
        if name in SAFE_HEADERS or name in ("transfer-encoding", "content-encoding"):
            require(name not in values, "framing")
            values[name] = value


def exact_bytes(stream, count):
    parts = []
    while count:
        part = stream.read(count)
        require(isinstance(part, bytes) and 0 < len(part) <= count, "framing")
        parts.append(part)
        count -= len(part)
    return b"".join(parts)


def read_wire(stream, method):
    """Read exactly one framed HTTP/1.x response, without library normalization."""
    metadata = None
    try:
        status_line = wire_line(stream)
        match = re.fullmatch(
            rb"HTTP/1\.[01] ([0-9]{3}) [\t\x20-\x7e\x80-\xff]*\r\n", status_line
        )
        require(match is not None, "framing")
        status = int(match[1])
        require(200 <= status <= 599, "framing")
        metadata = envelope(status)
        # A known authentication failure must never become a local body error.
        # Stop before reading headers or a HEAD/oversized response body.
        require(status != 401, "authentication", "http")
        headers = wire_headers(stream)
        length = headers.get("content-length")
        if length is not None:
            require(re.fullmatch(r"0|[1-9][0-9]{0,19}", length) is not None, "framing")
        encoding = headers.get("transfer-encoding")
        require(encoding is None or encoding.lower() == "chunked", "framing")
        require(length is None or encoding is None, "framing")
        require(
            headers.get("content-encoding", "identity").lower() == "identity", "framing"
        )
        request_id = headers.get("x-amz-request-id")
        require(
            request_id is None
            or re.fullmatch(r"[A-Za-z0-9+/=_-]{1,256}", request_id) is not None,
            "framing",
        )
        region = headers.get("x-amz-bucket-region")
        require(
            region is None or re.fullmatch(r"[a-z0-9-]{1,64}", region) is not None,
            "framing",
        )
        marker = headers.get("x-amz-delete-marker")
        require(marker is None or marker in ("true", "false"), "framing")
        version = headers.get("x-amz-version-id")
        require(
            version is None
            or re.fullmatch(r"[\x21-\x7e]{1,2048}", version) is not None,
            "framing",
        )
        metadata["request_id"] = request_id
        require(not 300 <= status <= 399, "redirect")
        if method == "HEAD":
            metadata["framing"] = "head-no-body"
            return finish_response(metadata, b""), headers, b""
        if status == 204:
            require(length is None and encoding is None, "framing")
            metadata["framing"] = "status-no-body"
            return finish_response(metadata, b""), headers, b""
        parts = []
        size = 0
        if length is not None:
            metadata["framing"] = "content-length"
            size = int(length)
            if size > LIMIT:
                raise OperationLocalFailure("response-limit", "transport", metadata)
            body = exact_bytes(stream, size)
        elif encoding is not None:
            metadata["framing"] = "chunked"
            while True:
                # These requests do not negotiate extensions; reject rather than guess.
                line = wire_line(stream)
                require(
                    re.fullmatch(rb"[0-9A-Fa-f]{1,16}\r\n", line) is not None, "framing"
                )
                count = int(line[:-2], 16)
                if count == 0:
                    wire_headers(stream, trailer=True)
                    break
                if count > LIMIT - size:
                    raise OperationLocalFailure("response-limit", "transport", metadata)
                parts.append(exact_bytes(stream, count))
                require(exact_bytes(stream, 2) == b"\r\n", "framing")
                size += count
            body = b"".join(parts)
        else:
            metadata["framing"] = "close-delimited"
            while True:
                part = stream.read(min(65536, LIMIT - size + 1))
                require(isinstance(part, bytes), "framing")
                if not part:
                    break
                size += len(part)
                if size > LIMIT:
                    raise OperationLocalFailure("response-limit", "transport", metadata)
                parts.append(part)
            body = b"".join(parts)
        return finish_response(metadata, body), headers, body
    except OperationLocalFailure as failure:
        if failure.code == "response-limit" and metadata is not None:
            metadata["body_limit_exceeded"] = True
        raise
    except GlobalFatalFailure as failure:
        if failure.response is None:
            failure.response = metadata
        raise
    except ssl.SSLError as error:
        raise GlobalFatalFailure("tls", response=metadata) from error
    except OSError as error:
        raise GlobalFatalFailure("network", response=metadata) from error


class Transport:
    def __init__(self, credentials):
        self.credentials = credentials
        self.cleanup_state = {"failed": False}
        self.counts = dict.fromkeys(("identity", *OPERATIONS), 0)

    def request(self, operation):
        spec = request_spec(operation)
        require(operation in self.counts, "request-boundary", "request")
        require(self.counts[operation] == 0, "request-budget", "request")
        _, host, method, path, query, _, body = spec
        headers = sign(spec, self.credentials, datetime.datetime.now(datetime.UTC))
        # A failed DNS, TLS or connect is still the one permitted attempt.
        self.counts[operation] += 1
        return https_request(
            host,
            method,
            path + ("?" + query if query else ""),
            body,
            headers,
            self.cleanup_state,
        )


def https_request(host, method, target, body, headers, cleanup_state=None):
    connection = stream = None
    prior_failure = None
    cleanup_state = cleanup_state if cleanup_state is not None else {"failed": False}
    try:
        context = ssl.create_default_context()
        context.set_alpn_protocols(["http/1.1"])
        connection = http.client.HTTPSConnection(host, timeout=5, context=context)
        connection.request(
            method, target, body=body, headers={**headers, "connection": "close"}
        )
        require(isinstance(connection.sock, ssl.SSLSocket), "tls")
        # Do not convert a missing TLS close_notify to a successful body EOF.
        connection.sock.suppress_ragged_eofs = False
        stream = connection.sock.makefile("rb")
        return read_wire(stream, method)
    except OperationLocalFailure:
        prior_failure = "local"
        raise
    except GlobalFatalFailure:
        prior_failure = "fatal"
        raise
    except ssl.SSLError as error:
        prior_failure = "fatal"
        raise GlobalFatalFailure("tls") from error
    except (OSError, http.client.HTTPException) as error:
        prior_failure = "fatal"
        raise GlobalFatalFailure("network") from error
    except BaseException:
        prior_failure = "fatal"
        raise
    finally:
        for owned in (stream, connection):
            if owned is not None:
                try:
                    owned.close()
                except Exception:
                    cleanup_state["failed"] = True
        if cleanup_state["failed"] and prior_failure != "fatal":
            raise GlobalFatalFailure("finalization", "finalization")


def xml(data):
    if len(data) > LIMIT or b"<!" in data or b"\0" in data:
        raise OperationLocalFailure("xml-boundary", "xml")
    try:
        root = ET.fromstring(data.decode("utf-8"))
    except (UnicodeDecodeError, ET.ParseError) as error:
        raise OperationLocalFailure("xml-shape", "xml") from error
    pending = [(root, 1)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > 4096 or depth > 64:
            raise OperationLocalFailure("xml-boundary", "xml")
        pending.extend((child, depth + 1) for child in node)
    return root


def one(parent, name):
    values = parent.findall(name)
    if len(values) != 1:
        raise OperationLocalFailure("xml-shape", "xml")
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
            raise OperationLocalFailure("json-shape", "json")
        result[key] = value
    return result


def config_root(data, name, allowed, namespaces=(S3_NS,)):
    root = xml(data)
    namespace = next((ns for ns in namespaces if root.tag == ns + name), None)
    if namespace is None or any(
        child.tag not in {namespace + item for item in allowed} for child in root
    ):
        raise OperationLocalFailure("xml-shape", "xml")
    return root


def leaf(root, name, allowed, optional=False):
    rows = root.findall(S3_NS + name)
    if optional and not rows:
        return None
    if len(rows) != 1 or list(rows[0]) or rows[0].text not in allowed:
        raise OperationLocalFailure("value-shape", "projection")
    return rows[0].text


def integer(text):
    if text is None or not re.fullmatch(r"0|[1-9][0-9]{0,19}", text):
        raise OperationLocalFailure("value-shape", "projection")
    return int(text)


def error_code(data):
    root = xml(data)
    if root.tag not in ("Error", S3_NS + "Error"):
        raise OperationLocalFailure("xml-shape", "xml")
    prefix = S3_NS if root.tag.startswith(S3_NS) else ""
    return one(root, prefix + "Code").text


ACCESS_STATUSES = {"AccessDenied": 403}
AUTH_STATUSES = {
    "InvalidAccessKeyId": 403,
    "InvalidToken": 400,
    "ExpiredToken": 400,
    "SignatureDoesNotMatch": 403,
    "AuthorizationHeaderMalformed": 400,
    "InvalidSecurity": 403,
    "MissingAuthenticationToken": 400,
}


def status_projection(operation, status, headers, data):
    if status == 401:
        raise GlobalFatalFailure("authentication", "http")
    if status == 200:
        return None
    if operation in ("HeadBucket", "HeadObject"):
        local_require(not data)
        if operation == "HeadObject" and status == 404:
            if headers.get("x-amz-delete-marker") == "true":
                return {"state": "delete-marker-present"}
            if "x-amz-version-id" in headers:
                return {"state": "unavailable", "reason": "head-version-metadata"}
            return {"state": "current-key-not-found"}
        return {"state": "unavailable", "reason": "head-status"}
    code = error_code(data)
    if code in AUTH_STATUSES and AUTH_STATUSES[code] != status:
        raise GlobalFatalFailure("authentication", "http")
    if status == 404 and ABSENT_CONFIG.get(operation) == code:
        return {"state": "not-configured"}
    if ACCESS_STATUSES.get(code) == status:
        return {"state": "unavailable", "reason": "access-denied"}
    if AUTH_STATUSES.get(code) == status:
        return {"state": "unavailable", "reason": "authentication"}
    return {"state": "unavailable", "reason": "service-status"}


def listing(data, versions):
    rows = ("Version", "DeleteMarker") if versions else ("Contents",)
    allowed = {"Name", "Prefix", "MaxKeys", "IsTruncated", *rows}
    allowed.update(("KeyMarker", "VersionIdMarker") if versions else ("KeyCount",))
    root = config_root(
        data, "ListVersionsResult" if versions else "ListBucketResult", allowed
    )
    local_require(
        one(root, S3_NS + "Name").text == BUCKET
        and one(root, S3_NS + "Prefix").text == OBJECT_NAME,
        "list-scope",
    )
    local_require(integer(one(root, S3_NS + "MaxKeys").text) == MAX_KEYS, "list-count")
    local_require(
        leaf(root, "IsTruncated", {"true", "false"}) == "false", "list-truncated"
    )
    if versions:
        for marker in ("KeyMarker", "VersionIdMarker"):
            values = root.findall(S3_NS + marker)
            local_require(
                len(values) <= 1 and (not values or not values[0].text), "list-scope"
            )
    entries = [(kind, row) for kind in rows for row in root.findall(S3_NS + kind)]
    local_require(len(entries) <= MAX_KEYS, "list-count")
    if not versions:
        local_require(
            integer(one(root, S3_NS + "KeyCount").text) == len(entries), "list-count"
        )
    seen, latest = set(), set()
    exact = current = noncurrent = markers = current_markers = 0
    for kind, row in entries:
        key = one(row, S3_NS + "Key").text
        local_require(
            isinstance(key, str) and key.startswith(OBJECT_NAME), "list-scope"
        )
        version = None
        is_latest = False
        if versions:
            version = one(row, S3_NS + "VersionId").text
            local_require(isinstance(version, str) and 0 < len(version) <= 2048)
            is_latest = leaf(row, "IsLatest", {"true", "false"}) == "true"
            local_require(not is_latest or key not in latest, "list-count")
            if is_latest:
                latest.add(key)
        local_require((key, version) not in seen, "list-count")
        seen.add((key, version))
        if key != OBJECT_NAME:
            continue
        exact += 1
        if kind == "Version":
            current += is_latest
            noncurrent += not is_latest
        if kind == "DeleteMarker":
            markers += 1
            current_markers += is_latest
    out = {
        "state": "complete",
        "exact_entries": exact,
        "other_prefix_entries": len(entries) - exact,
    }
    if versions:
        out.update(
            exact_current_versions=current,
            exact_noncurrent_versions=noncurrent,
            exact_delete_markers=markers,
            exact_current_delete_markers=current_markers,
        )
    else:
        local_require(exact <= 1, "list-count")
    return out


def project_success(operation, headers, data):
    if operation in ("HeadBucket", "HeadObject"):
        local_require(not data)
        if operation == "HeadBucket":
            local_require(headers.get("x-amz-bucket-region") == REGION)
            return {"state": "owner-region-matched"}
        local_require(headers.get("x-amz-delete-marker") != "true")
        return {
            "state": "current-key-present",
            "bytes": integer(headers.get("content-length")),
            "version_id_present": "x-amz-version-id" in headers,
            "delete_marker": False,
        }
    if operation in ("ListObjectsV2", "ListObjectVersions"):
        return listing(data, operation == "ListObjectVersions")
    if operation == "GetBucketPolicy":
        try:
            value = json.loads(data, object_pairs_hook=object_pairs)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise OperationLocalFailure("json-shape", "json") from error
        local_require(
            isinstance(value, dict) and set(value) <= {"Version", "Id", "Statement"},
            "json-shape",
            "json",
        )
        rows = value.get("Statement")
        rows = [rows] if isinstance(rows, dict) else rows
        local_require(
            isinstance(rows, list)
            and len(rows) <= 4096
            and all(isinstance(row, dict) for row in rows),
            "json-shape",
            "json",
        )
        return {"state": "present-unreviewed", "statements": len(rows)}
    if operation == "GetBucketLocation":
        root = config_root(data, "LocationConstraint", set())
        local_require(root.text in (None, "", REGION))
        return {"state": "region-matched"}
    if operation == "GetBucketPolicyStatus":
        root = config_root(data, "PolicyStatus", {"IsPublic"})
        return {
            "state": "present",
            "public": leaf(root, "IsPublic", {"true", "false"}) == "true",
        }
    if operation == "GetPublicAccessBlock":
        fields = {
            "BlockPublicAcls": "block_public_acls",
            "IgnorePublicAcls": "ignore_public_acls",
            "BlockPublicPolicy": "block_public_policy",
            "RestrictPublicBuckets": "restrict_public_buckets",
        }
        root = config_root(data, "PublicAccessBlockConfiguration", set(fields))
        values = {
            out: leaf(root, field, {"true", "false"}) == "true"
            for field, out in fields.items()
        }
        return {"state": "present", **values, "all_enabled": all(values.values())}
    if operation == "GetBucketOwnershipControls":
        root = config_root(data, "OwnershipControls", {"Rule"})
        row = one(root, S3_NS + "Rule")
        return {
            "state": leaf(
                row,
                "ObjectOwnership",
                {"BucketOwnerPreferred", "ObjectWriter", "BucketOwnerEnforced"},
            )
        }
    if operation == "GetBucketEncryption":
        root = config_root(data, "ServerSideEncryptionConfiguration", {"Rule"})
        algorithms = set()
        for row in root:
            default = one(row, S3_NS + "ApplyServerSideEncryptionByDefault")
            algorithms.add(
                leaf(default, "SSEAlgorithm", {"AES256", "aws:kms", "aws:kms:dsse"})
            )
        local_require(bool(algorithms))
        return {"state": "configured", "algorithms": sorted(algorithms)}
    raise GlobalFatalFailure("request-boundary", "projection")


def project(operation, response):
    request_spec(operation)
    metadata, headers, data = response
    require(
        operation != "identity"
        and metadata["response_complete"]
        and len(data) <= LIMIT,
        "internal",
        "projection",
    )
    out = status_projection(operation, metadata["http_status"], headers, data)
    return project_success(operation, headers, data) if out is None else out


def same_value(left, right):
    return type(left) is type(right) and left == right


def schema_matches(value, rule, definitions):
    """Validate only the fixed, checked-in schema vocabulary; no remote refs."""
    if "$ref" in rule:
        ref = rule["$ref"]
        require(ref.startswith("#/$defs/") and ref[8:] in definitions)
        return schema_matches(value, definitions[ref[8:]], definitions)
    if (
        "oneOf" in rule
        and sum(schema_matches(value, row, definitions) for row in rule["oneOf"]) != 1
    ):
        return False
    if "anyOf" in rule and not any(
        schema_matches(value, row, definitions) for row in rule["anyOf"]
    ):
        return False
    if "const" in rule and not same_value(value, rule["const"]):
        return False
    if "enum" in rule and not any(same_value(value, item) for item in rule["enum"]):
        return False
    kind = rule.get("type")
    if (
        kind is not None
        and type(value)
        is not {
            "object": dict,
            "array": list,
            "string": str,
            "integer": int,
            "boolean": bool,
            "null": type(None),
        }[kind]
    ):
        return False
    if kind == "object":
        props = rule.get("properties", {})
        if not set(rule.get("required", ())) <= set(value):
            return False
        if rule.get("additionalProperties") is False and not set(value) <= set(props):
            return False
        return all(
            schema_matches(item, props[key], definitions)
            for key, item in value.items()
            if key in props
        )
    if kind == "array":
        if (
            not rule.get("minItems", 0)
            <= len(value)
            <= rule.get("maxItems", len(value))
        ):
            return False
        if rule.get("uniqueItems") and len(
            {json.dumps(item, sort_keys=True) for item in value}
        ) != len(value):
            return False
        return "items" not in rule or all(
            schema_matches(item, rule["items"], definitions) for item in value
        )
    if kind == "integer" and not rule.get("minimum", value) <= value <= rule.get(
        "maximum", value
    ):
        return False
    return (
        kind != "string"
        or "pattern" not in rule
        or re.fullmatch(rule["pattern"], value) is not None
    )


def output_schema():
    from pathlib import Path

    return json.loads(
        Path(__file__).with_name("ci_hf_cache_metadata_tail_schema.json").read_bytes()
    )


def validate_output(out):
    schema = output_schema()
    require(schema_matches(out, schema, schema["$defs"]))
    ops, counts, fatal, code = (
        out[name]
        for name in ("operations", "request_counts", "fatal_stage", "fatal_code")
    )
    prefix = [counts[name] for name in OPERATIONS]
    require(out["probe_error"] == (fatal is not None) == (code is not None))
    require(out["collector_cleanup_confirmed"] or out["probe_error"])
    require(not out["identity_matches"] or counts["identity"] == 1)
    require(out["identity_matches"] or not any(prefix))
    if fatal in ("source", "environment", "oidc", "credentials"):
        require(not any(counts.values()))
    if fatal == "identity":
        require(not out["identity_matches"] and not any(prefix))
    if fatal == "finalization":
        require(code == "finalization" and not out["collector_cleanup_confirmed"])
    owner = ops["HeadBucket"]["projection"] == {
        "state": "owner-region-matched"
    } and ops["GetBucketLocation"]["projection"] == {"state": "region-matched"}
    require(out["owner_region_matches"] == owner)
    require(owner or not any(prefix[2:]))
    require(prefix == sorted(prefix, reverse=True))
    require(out["all_operations_attempted"] == all(prefix))
    require(
        out["all_responses_projected"]
        == (
            all(prefix)
            and all(ops[name]["projection"] is not None for name in OPERATIONS)
        )
    )
    if not out["probe_error"]:
        require(out["identity_matches"] and owner and all(prefix))
    fatal_ops = []
    for name in OPERATIONS:
        rec = ops[name]
        kind, response, projection, failure = (
            rec[key] for key in ("disposition", "response", "projection", "failure")
        )
        count = counts[name]
        if kind == "not-attempted":
            require(count == 0 and response is projection is failure is None)
            continue
        if count == 0:
            require(
                kind == "fatal"
                and fatal == name
                and response is projection is None
                and failure["stage"] == "request"
                and failure["code"] in ("request-boundary", "request-budget")
            )
        if kind == "projected":
            require(
                count == 1
                and response is not None
                and response["response_complete"]
                and projection is not None
                and failure is None
            )
        if kind == "operation-error":
            require(
                name in OPERATIONS[2:]
                and owner
                and count == 1
                and projection is None
                and failure["class"] == "operation-local"
                and response is not None
            )
            if failure["code"] == "response-limit":
                require(
                    name not in ("HeadBucket", "HeadObject")
                    and response["framing"]
                    in ("content-length", "chunked", "close-delimited")
                    and response["http_status"] != 204
                    and not response["response_complete"]
                    and response["body_limit_exceeded"]
                )
            else:
                require(response["response_complete"])
        if kind == "fatal":
            require(
                failure is not None
                and failure["class"] == "global-fatal"
                and failure["code"] == code
            )
            fatal_ops.append(name)
        if response is not None:
            status, framing, complete = (
                response[key] for key in ("http_status", "framing", "response_complete")
            )
            if status == 401:
                require(
                    kind == "fatal"
                    and failure["code"] == "authentication"
                    and projection is None
                )
            if name in ("HeadBucket", "HeadObject"):
                require(
                    framing in ("unaccepted", "head-no-body")
                    and not response["body_limit_exceeded"]
                )
            if complete:
                require(
                    framing != "unaccepted"
                    and not response["body_limit_exceeded"]
                    and response["response_bytes"] is not None
                    and response["response_sha256"] is not None
                )
                if name in ("HeadBucket", "HeadObject"):
                    require(framing == "head-no-body")
                elif status == 204:
                    require(framing == "status-no-body")
                else:
                    require(framing in ("content-length", "chunked", "close-delimited"))
            else:
                require(
                    response["response_bytes"] is response["response_sha256"] is None
                )
            if complete and framing in ("head-no-body", "status-no-body"):
                require(
                    (response["response_bytes"], response["response_sha256"])
                    == (0, EMPTY_HASH)
                )
            if 300 <= status <= 399:
                require(
                    kind == "fatal"
                    and failure["code"] == "redirect"
                    and projection is None
                    and not complete
                )
        if projection is None:
            continue
        require(response is not None and response["response_complete"])
        state, status = projection["state"], response["http_status"]
        if state == "unavailable":
            reason = projection["reason"]
            require(status != 200 and not 300 <= status <= 399)
            if reason == "access-denied":
                require(status in set(ACCESS_STATUSES.values()))
            if reason == "authentication":
                require(
                    status in set(AUTH_STATUSES.values())
                    and kind == "fatal"
                    and failure["code"] == "authentication"
                )
            if reason == "head-version-metadata":
                require(name == "HeadObject" and status == 404)
        elif state in (
            "not-configured",
            "current-key-not-found",
            "delete-marker-present",
        ):
            require(status == 404)
        else:
            require(status == 200)
    require(fatal_ops == ([fatal] if fatal in OPERATIONS else []))
    if fatal in OPERATIONS:
        index = OPERATIONS.index(fatal)
        require(all(counts[name] == 1 for name in OPERATIONS[:index]))
        require(
            all(
                counts[name] == 0 and ops[name]["disposition"] == "not-attempted"
                for name in OPERATIONS[index + 1 :]
            )
        )
    projection = ops["GetPublicAccessBlock"]["projection"]
    if projection and projection["state"] == "present":
        require(
            projection["all_enabled"]
            == all(
                projection[key]
                for key in (
                    "block_public_acls",
                    "ignore_public_acls",
                    "block_public_policy",
                    "restrict_public_buckets",
                )
            )
        )
    projection = ops["GetBucketEncryption"]["projection"]
    if projection and projection["state"] == "configured":
        require(projection["algorithms"] == sorted(projection["algorithms"]))
    projection = ops["ListObjectsV2"]["projection"]
    if projection and projection["state"] == "complete":
        require(
            projection["exact_entries"] + projection["other_prefix_entries"] <= MAX_KEYS
        )
    projection = ops["ListObjectVersions"]["projection"]
    if projection and projection["state"] == "complete":
        require(
            projection["exact_entries"]
            == projection["exact_current_versions"]
            + projection["exact_noncurrent_versions"]
            + projection["exact_delete_markers"]
        )
        require(
            projection["exact_current_delete_markers"]
            <= projection["exact_delete_markers"]
        )
        require(
            projection["exact_current_versions"]
            + projection["exact_current_delete_markers"]
            <= 1
        )
        require(
            projection["exact_entries"] + projection["other_prefix_entries"] <= MAX_KEYS
        )
    return out


def idle_record():
    return {
        "disposition": "not-attempted",
        "response": None,
        "projection": None,
        "failure": None,
    }


def result_skeleton():
    return {
        "schema": 1,
        "identity_matches": False,
        "owner_region_matches": False,
        "operations": {name: idle_record() for name in OPERATIONS},
        "request_counts": dict.fromkeys(("identity", *OPERATIONS), 0),
        "all_operations_attempted": False,
        "all_responses_projected": False,
        "probe_error": False,
        "fatal_stage": None,
        "fatal_code": None,
        "write_authorized": False,
        "collector_cleanup_confirmed": False,
    }


def record_failure(out, operation, failure):
    is_local = isinstance(failure, OperationLocalFailure)
    if operation in OPERATIONS:
        out["operations"][operation] = {
            "disposition": "operation-error" if is_local else "fatal",
            "response": failure.response,
            "projection": getattr(failure, "projection", None),
            "failure": {
                "class": "operation-local" if is_local else "global-fatal",
                "stage": failure.stage,
                "code": failure.code,
            },
        }
    if not is_local and not out["probe_error"]:
        out.update(probe_error=True, fatal_stage=operation, fatal_code=failure.code)


def collect(transport, out):
    stage, response = "identity", None
    out["request_counts"] = transport.counts
    try:
        response = transport.request(stage)
        require(
            response[0]["http_status"] == 200 and identity_matches(response[2]),
            "identity",
            "identity",
        )
        out["identity_matches"] = True
        for stage in OPERATIONS:
            response = None
            try:
                response = transport.request(stage)
                projection = project(stage, response)
                if projection.get("reason") == "authentication":
                    raise GlobalFatalFailure(
                        "authentication", "http", response[0], projection
                    )
                if stage in OPERATIONS[:2] and projection["state"] != (
                    "owner-region-matched"
                    if stage == "HeadBucket"
                    else "region-matched"
                ):
                    raise GlobalFatalFailure(
                        "owner-region", "owner-region", response[0], projection
                    )
                out["operations"][stage] = {
                    "disposition": "projected",
                    "response": response[0],
                    "projection": projection,
                    "failure": None,
                }
                if stage == "GetBucketLocation":
                    out["owner_region_matches"] = True
            except OperationLocalFailure as failure:
                if failure.response is None and response is not None:
                    failure.response = response[0]
                if stage not in OPERATIONS[2:]:
                    raise GlobalFatalFailure(
                        "owner-region", "owner-region", failure.response
                    ) from failure
                record_failure(out, stage, failure)
    except OperationLocalFailure as failure:
        record_failure(
            out, stage, GlobalFatalFailure("identity", "identity", failure.response)
        )
    except GlobalFatalFailure as failure:
        if failure.response is None and response is not None and stage in OPERATIONS:
            failure.response = response[0]
        record_failure(out, stage, failure)
    except Exception:
        record_failure(
            out,
            stage,
            GlobalFatalFailure(
                "internal",
                "internal",
                response[0] if response is not None and stage in OPERATIONS else None,
            ),
        )
    return out


def complete_result(out, cleanup_confirmed):
    if not cleanup_confirmed and not out["probe_error"]:
        record_failure(
            out, "finalization", GlobalFatalFailure("finalization", "finalization")
        )
    out["collector_cleanup_confirmed"] = cleanup_confirmed
    out["all_operations_attempted"] = all(
        out["request_counts"][name] == 1 for name in OPERATIONS
    )
    out["all_responses_projected"] = out["all_operations_attempted"] and all(
        out["operations"][name]["projection"] is not None for name in OPERATIONS
    )
    return validate_output(out)


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


def validate_environment(env, after_action=False):
    if not scope_valid(env) or env.get("CANARY_ROLE_NAME") != ROLE:
        raise GlobalFatalFailure("environment-boundary", "environment")
    for key in (
        "AWS_EC2_METADATA_DISABLED",
        "AWS_MAX_ATTEMPTS",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
    ):
        if env.get(key) != FIXED_AWS_ENV[key]:
            raise GlobalFatalFailure("environment-boundary", "environment")
    for key, value in env.items():
        if not value:
            continue
        if key in TLS_OVERRIDES or key.startswith("INPUT_"):
            raise GlobalFatalFailure("environment-boundary", "environment")
        if not key.startswith("AWS_"):
            continue
        if key in FIXED_AWS_ENV and value == FIXED_AWS_ENV[key]:
            continue
        if after_action and key in CREDENTIAL_KEYS:
            continue
        raise GlobalFatalFailure("environment-boundary", "environment")


def preflight(env):
    validate_environment(env)
    return bool(
        env.get("ACTIONS_ID_TOKEN_REQUEST_URL")
        and env.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    )


def runtime_inputs(env):
    validate_environment(env, after_action=True)
    if not (
        env.get("ACTIONS_ID_TOKEN_REQUEST_URL")
        and env.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    ):
        raise GlobalFatalFailure("oidc-unavailable", "oidc")
    credentials = tuple(env.get(key, "") for key in CREDENTIAL_KEYS)
    if (
        not re.fullmatch(r"ASIA[A-Z0-9]{16}", credentials[0])
        or not 16 <= len(credentials[1]) <= 128
        or not 16 <= len(credentials[2]) <= 16384
        or any("\n" in value or "\r" in value for value in credentials)
    ):
        raise GlobalFatalFailure("credentials", "credentials")
    return credentials


def validate_source(env):
    require(scope_valid(env), "source-boundary", "source")
    token = env.get("CANARY_GITHUB_TOKEN", "")
    require(
        bool(token) and "\r" not in token and "\n" not in token,
        "source-boundary",
        "source",
    )
    try:
        metadata, _, data = https_request(
            "api.github.com",
            "GET",
            f"/repos/pytorch/pytorch/pulls/{PR_NUMBER}",
            b"",
            {
                "authorization": "Bearer " + token,
                "accept": "application/vnd.github+json",
                "x-github-api-version": "2022-11-28",
                "user-agent": "pytorch-hf-cache-metadata-tail-audit",
            },
        )
        require(metadata["http_status"] == 200, "source-boundary", "source")
        pr = json.loads(data, object_pairs_hook=object_pairs)
        require(source_matches(pr, env["GITHUB_SHA"]), "source-boundary", "source")
    except (
        OperationLocalFailure,
        json.JSONDecodeError,
        UnicodeDecodeError,
        KeyError,
        TypeError,
    ) as error:
        raise GlobalFatalFailure("source-boundary", "source") from error


def preflight_main():
    try:
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(120)
        available = preflight(os.environ)
        validate_source(os.environ)
        line = "oidc_requestable=" + str(available).lower()
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(line + "\n")
    except Exception:
        print("provider_gate_passed=false")
        return 1
    finally:
        signal.alarm(0)
    print(line)
    return 0


def deadline(_signum, _frame):
    raise GlobalFatalFailure("deadline", "transport")


def main():
    out = result_skeleton()
    transport = None
    cleanup = True
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(120)
        transport = Transport(runtime_inputs(os.environ))
        collect(transport, out)
    except GlobalFatalFailure as failure:
        record_failure(out, failure.stage, failure)
    except Exception:
        record_failure(
            out,
            "environment",
            GlobalFatalFailure("environment-boundary", "environment"),
        )
    finally:
        try:
            signal.alarm(0)
        except Exception:
            cleanup = False
        if transport is not None and transport.cleanup_state["failed"]:
            cleanup = False
    try:
        complete_result(out, cleanup)
    except Exception:
        print("collector_output_valid=false")
        return 1
    print(json.dumps(out, sort_keys=True))
    return int(out["probe_error"])


if __name__ == "__main__":
    if sys.argv[1:] == ["--preflight"]:
        raise SystemExit(preflight_main())
    if sys.argv[1:]:
        raise SystemExit("unsupported invocation")
    raise SystemExit(main())
