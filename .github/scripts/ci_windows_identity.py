"""Test-only metadata-only own-instance diagnostic; never emit response data."""

from __future__ import annotations

import sys


BOOTSTRAP_FAILED = False
try:
    import ctypes
    import ctypes.wintypes as wintypes
    import http.client as http
    import json
    import os
    import re
    import time
except BaseException:
    # The launcher also discards stderr if the interpreter cannot start at all.
    BOOTSTRAP_FAILED = True


ACCOUNT = "308535385114"
REGION = "us-east-1"
ROLE = "gh-ci-github-action-runners-runner-role"
PROFILE = (
    f"arn:aws:iam::{ACCOUNT}:instance-profile/gh-ci/gh-ci-github-action-runners-profile"
)
PYTHON = r"C:\Jenkins\Miniconda3\python.exe"
KEYS = (
    "runtime_matched",
    "windows_administrator",
    "ambient_credential_inputs_present",
    "metadata_reachable",
    "runner_name_well_formed",
    "document_instance_matches_runner",
    "metadata_instance_matches_runner",
    "document_account_matches_expected",
    "document_region_matches_expected",
    "metadata_region_matches_expected",
    "availability_zone_matches_expected",
    "image_id_format_valid",
    "profile_metadata_success",
    "profile_arn_matches_expected",
    "profile_id_format_valid",
    "role_name_matches_expected",
    "probe_error",
)
OVERRIDES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_CONFIG_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_STS",
    "AWS_ENDPOINT_URL_EC2",
    "AWS_CA_BUNDLE",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
    "AWS_EC2_METADATA_DISABLED",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)
MODULES = (
    ("http", "http.client"),
    ("json", "json"),
    ("os", "os"),
    ("re", "re"),
    ("time", "time"),
    ("ctypes", "ctypes"),
    ("wintypes", "ctypes.wintypes"),
)


def load_stdlib(importer=__import__):
    # Fixed stdlib imports only. Missing Conda DLLs are a runtime failure.
    loaded = {alias: importer(name, fromlist=["*"]) for alias, name in MODULES}
    globals().update(loaded)


def require(condition):
    if not condition:
        raise ValueError("probe rejected")


def exact(value, pattern):
    require(isinstance(value, str) and re.fullmatch(pattern, value) is not None)
    return value


def runtime_matches(name, version, executable):
    return (
        name == "nt"
        and tuple(version[:2]) == (3, 11)
        and executable.casefold() == PYTHON.casefold()
    )


def admin_token():
    # A stack-owned well-known SID and NULL token require no handle allocation.
    advapi = ctypes.WinDLL("advapi32.dll", use_last_error=True)
    create = advapi.CreateWellKnownSid
    create.argtypes = [
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.DWORD),
    ]
    create.restype = wintypes.BOOL
    check = advapi.CheckTokenMembership
    check.argtypes = [wintypes.HANDLE, wintypes.LPVOID, ctypes.POINTER(wintypes.BOOL)]
    check.restype = wintypes.BOOL
    sid = ctypes.create_string_buffer(68)
    size = wintypes.DWORD(len(sid))
    require(bool(create(26, None, sid, ctypes.byref(size))))
    member = wintypes.BOOL()
    require(bool(check(None, sid, ctypes.byref(member))))
    return bool(member.value)


def request_spec(operation, token=""):
    paths = {
        "document": "/latest/dynamic/instance-identity/document",
        "instance": "/latest/meta-data/instance-id",
        "region": "/latest/meta-data/placement/region",
        "profile": "/latest/meta-data/iam/info",
        "role": "/latest/meta-data/iam/security-credentials/",
    }
    if operation == "token":
        method, path = "PUT", "/latest/api/token"
        headers = {"X-aws-ec2-metadata-token-ttl-seconds": "60"}
    else:
        require(operation in paths)
        method, path = "GET", paths[operation]
        headers = {
            "X-aws-ec2-metadata-token": exact(token, r"[A-Za-z0-9+/=_-]{1,4096}")
        }
    return {
        "method": method,
        "host": "169.254.169.254",
        "path": path,
        "body": b"",
        "headers": headers,
        "tls": False,
        "limit": 32768,
    }


def validate_request(request):
    require(
        set(request) == {"method", "host", "path", "body", "headers", "tls", "limit"}
    )
    require(
        request["host"] == "169.254.169.254"
        and request["tls"] is False
        and request["body"] == b""
        and request["limit"] == 32768
    )
    if request["path"] == "/latest/api/token":
        require(
            request["method"] == "PUT"
            and request["headers"] == {"X-aws-ec2-metadata-token-ttl-seconds": "60"}
        )
        return
    paths = {
        request_spec(op, "placeholder")["path"]
        for op in ("document", "instance", "region", "profile", "role")
    }
    require(
        request["method"] == "GET"
        and request["path"] in paths
        and set(request["headers"]) == {"X-aws-ec2-metadata-token"}
    )
    exact(request["headers"]["X-aws-ec2-metadata-token"], r"[A-Za-z0-9+/=_-]{1,4096}")


def remaining(deadline, clock, maximum):
    left = deadline - clock()
    require(left > 0)
    return min(maximum, left)


def http_request(request, deadline, clock=None, factory=None):
    validate_request(request)
    clock = clock or time.monotonic
    # http.client does not consult proxy environment variables or follow redirects.
    timeout = remaining(deadline, clock, 5)
    if factory is not None:
        conn = factory(request, timeout)
    else:
        conn = http.HTTPConnection(request["host"], timeout=timeout)
    response = None
    try:
        conn.set_debuglevel(0)
        conn.connect()
        # getresponse() may clear conn.sock on Connection: close. The socket's
        # makefile reference keeps this handle usable until response EOF.
        read_socket = conn.sock
        read_socket.settimeout(remaining(deadline, clock, 10))
        conn.request(
            request["method"],
            request["path"],
            body=request["body"],
            headers=request["headers"],
        )
        read_socket.settimeout(remaining(deadline, clock, 10))
        response = conn.getresponse()
        remaining(deadline, clock, 10)
        length = response.getheader("Content-Length")
        declared_length = None
        if length is not None:
            exact(length, r"[0-9]{1,12}")
            declared_length = int(length)
            require(declared_length <= request["limit"])
        chunks, total = [], 0
        while not response.isclosed():
            read_socket.settimeout(remaining(deadline, clock, 10))
            chunk = response.read1(min(4096, request["limit"] + 1 - total))
            remaining(deadline, clock, 10)
            if not chunk:
                break
            total += len(chunk)
            require(total <= request["limit"])
            chunks.append(chunk)
        require(declared_length is None or total == declared_length)
        return response.status, b"".join(chunks)
    finally:
        if response is not None:
            response.close()
        conn.close()


def call(transport, operation, *args):
    request = request_spec(operation, *args)
    validate_request(request)
    status, body = transport(request)
    require(
        type(status) is int
        and isinstance(body, bytes)
        and len(body) <= request["limit"]
    )
    return status, body


def success(response):
    status, body = response
    require(status == 200)
    return body


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result)
        result[key] = value
    return result


def validate_json_depth(value, depth=0):
    require(depth <= 16)
    if isinstance(value, dict):
        for item in value.values():
            validate_json_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            validate_json_depth(item, depth + 1)


def json_object(raw):
    require(len(raw) <= 32768)
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=unique_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("probe rejected")),
    )
    require(type(value) is dict)
    validate_json_depth(value)
    return value


def matches(value, pattern):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def probe(runner, administrator, environment, transport):
    result = dict.fromkeys(KEYS, False)
    token = ""
    try:
        # run_live checks the actual interpreter before entering this protocol.
        result["runtime_matched"] = True
        result["windows_administrator"] = bool(administrator)
        result["ambient_credential_inputs_present"] = any(
            environment(name) for name in OVERRIDES
        )
        require(not result["ambient_credential_inputs_present"])
        result["runner_name_well_formed"] = matches(runner, r"i-[0-9a-f]{17}")
        require(result["runner_name_well_formed"])
        token = exact(
            success(call(transport, "token")).decode("ascii"),
            r"[A-Za-z0-9+/=_-]{1,4096}",
        )
        result["metadata_reachable"] = True
        document = json_object(success(call(transport, "document", token)))
        instance = success(call(transport, "instance", token)).decode("ascii")
        region = success(call(transport, "region", token)).decode("ascii")
        result["document_instance_matches_runner"] = (
            document.get("instanceId") == runner
        )
        result["metadata_instance_matches_runner"] = instance == runner
        result["document_account_matches_expected"] = (
            document.get("accountId") == ACCOUNT
        )
        result["document_region_matches_expected"] = document.get("region") == REGION
        result["metadata_region_matches_expected"] = region == REGION
        result["availability_zone_matches_expected"] = matches(
            document.get("availabilityZone"), r"us-east-1[a-z]"
        )
        result["image_id_format_valid"] = matches(
            document.get("imageId"), r"ami-[0-9a-f]{17}"
        )
        require(all(result[key] for key in KEYS[4:12]))
        info = json_object(success(call(transport, "profile", token)))
        result["profile_metadata_success"] = info.get("Code") == "Success"
        result["profile_arn_matches_expected"] = (
            info.get("InstanceProfileArn") == PROFILE
        )
        result["profile_id_format_valid"] = matches(
            info.get("InstanceProfileId"), r"AIPA[A-Z0-9]{16,64}"
        )
        require(all(result[key] for key in KEYS[12:15]))
        result["role_name_matches_expected"] = (
            success(call(transport, "role", token)).decode("ascii").rstrip("\r\n")
            == ROLE
        )
        require(result["role_name_matches_expected"])
    except BaseException:
        result["probe_error"] = True
    finally:
        token = ""
    return result


def run_live():
    require(not BOOTSTRAP_FAILED)
    load_stdlib()
    require(
        runtime_matches(os.name, sys.version_info, os.path.normpath(sys.executable))
    )
    deadline = time.monotonic() + 120
    return probe(
        os.environ.get("EXPECTED_RUNNER_NAME", ""),
        admin_token(),
        os.environ.get,
        lambda request: http_request(request, deadline),
    )


def emit(result):
    # Bootstrap errors do not require json import.
    require(
        set(result) == set(KEYS)
        and all(type(value) is bool for value in result.values())
    )
    sys.stdout.write(
        "{"
        + ",".join(
            '"' + key + '":' + ("true" if result[key] else "false") for key in KEYS
        )
        + "}\n"
    )
    sys.stdout.flush()


def main(run=run_live):
    try:
        outcome = run()
    except BaseException:
        outcome = dict.fromkeys(KEYS, False)
        outcome["probe_error"] = True
    emit(outcome)
    return 1 if outcome["probe_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
