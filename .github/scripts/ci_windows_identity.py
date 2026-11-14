"""Test-only exact-own-instance probe; never emit response or credential data."""

from __future__ import annotations

import sys


BOOTSTRAP_FAILED = False
try:
    import ctypes
    import ctypes.wintypes as wintypes
    import datetime
    import hashlib
    import hmac
    import http.client as http
    import json
    import os
    import re
    import ssl
    import time
    import xml.etree.ElementTree as ET
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
    "windows_administrator",
    "ambient_credential_inputs_present",
    "metadata_reachable",
    "runner_instance_identity_matched",
    "instance_profile_matched",
    "instance_role_matched",
    "temporary_credentials_present",
    "sts_authenticated",
    "own_instance_describe_allowed",
    "own_instance_describe_denied",
    "own_instance_identity_matched",
    "own_instance_environment_matched",
    "own_instance_dynamic",
    "own_root_device_ebs",
    "own_root_delete_on_termination",
    "own_other_ebs_mapping_present",
    "own_all_ebs_delete_on_termination",
    "own_ephemeral_finished_tag_present",
    "own_root_replacement_tag_present",
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
    ("datetime", "datetime"),
    ("hashlib", "hashlib"),
    ("hmac", "hmac"),
    ("http", "http.client"),
    ("json", "json"),
    ("os", "os"),
    ("re", "re"),
    ("ssl", "ssl"),
    ("time", "time"),
    ("ET", "xml.etree.ElementTree"),
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


def signed_headers(method, host, service, region, body, headers, credentials, stamp):
    exact(stamp, r"[0-9]{8}T[0-9]{6}Z")
    values = {key.lower(): value for key, value in headers.items()}
    values.update({"host": host, "x-amz-date": stamp})
    if credentials.get("Token"):
        values["x-amz-security-token"] = credentials["Token"]
    names = ";".join(sorted(values))
    canonical_headers = "".join(f"{key}:{values[key]}\n" for key in sorted(values))
    canonical = f"{method}\n/\n\n{canonical_headers}\n{names}\n{hashlib.sha256(body).hexdigest()}"
    scope = f"{stamp[:8]}/{region}/{service}/aws4_request"
    to_sign = f"AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"
    key = ("AWS4" + credentials["SecretAccessKey"]).encode()
    for part in (stamp[:8], region, service, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    values["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={credentials['AccessKeyId']}/{scope}, SignedHeaders={names}, Signature={signature}"
    )
    return values


def request_spec(operation, instance="", token="", credentials=None, stamp=""):
    method, host, path, body, tls, limit = (
        "GET",
        "169.254.169.254",
        "",
        b"",
        False,
        32768,
    )
    headers = {}
    paths = {
        "document": "/latest/dynamic/instance-identity/document",
        "instance": "/latest/meta-data/instance-id",
        "region": "/latest/meta-data/placement/region",
        "profile": "/latest/meta-data/iam/info",
        "role": "/latest/meta-data/iam/security-credentials/",
        "credentials": "/latest/meta-data/iam/security-credentials/" + ROLE,
    }
    if operation == "token":
        method, path = "PUT", "/latest/api/token"
        headers["X-aws-ec2-metadata-token-ttl-seconds"] = "60"
    elif operation in paths:
        path = paths[operation]
        headers["X-aws-ec2-metadata-token"] = exact(token, r"[A-Za-z0-9+/=_-]{1,4096}")
    elif operation in ("sts", "ec2"):
        method, host, path, tls = (
            "POST",
            f"{operation}.us-east-1.amazonaws.com",
            "/",
            True,
        )
        if operation == "sts":
            body = b"Action=GetCallerIdentity&Version=2011-06-15"
        else:
            body = (
                "Action=DescribeInstances&Version=2016-11-15&InstanceId.1="
                + exact(instance, r"i-[0-9a-f]{17}")
            ).encode()
            limit = 262144
        headers = signed_headers(
            method,
            host,
            operation,
            REGION,
            body,
            {"content-type": "application/x-www-form-urlencoded; charset=utf-8"},
            credentials,
            stamp,
        )
    else:
        raise ValueError("probe rejected")
    return {
        "method": method,
        "host": host,
        "path": path,
        "body": body,
        "headers": headers,
        "tls": tls,
        "limit": limit,
    }


def validate_request(request):
    require(
        set(request) == {"method", "host", "path", "body", "headers", "tls", "limit"}
    )
    if request["host"] == "169.254.169.254":
        require(
            request["tls"] is False
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
            request_spec(op, token="placeholder")["path"]
            for op in (
                "document",
                "instance",
                "region",
                "profile",
                "role",
                "credentials",
            )
        }
        require(
            request["method"] == "GET"
            and request["path"] in paths
            and set(request["headers"]) == {"X-aws-ec2-metadata-token"}
        )
        exact(
            request["headers"]["X-aws-ec2-metadata-token"], r"[A-Za-z0-9+/=_-]{1,4096}"
        )
        return
    service = {
        "sts.us-east-1.amazonaws.com": "sts",
        "ec2.us-east-1.amazonaws.com": "ec2",
    }.get(request["host"])
    require(
        service is not None
        and request["tls"] is True
        and request["method"] == "POST"
        and request["path"] == "/"
    )
    require(request["limit"] == (32768 if service == "sts" else 262144))
    if service == "sts":
        require(request["body"] == b"Action=GetCallerIdentity&Version=2011-06-15")
    else:
        exact(
            request["body"].decode("ascii"),
            r"Action=DescribeInstances&Version=2016-11-15&InstanceId\.1=i-[0-9a-f]{17}",
        )
    headers = request["headers"]
    require(
        set(headers)
        == {
            "Authorization",
            "content-type",
            "host",
            "x-amz-date",
            "x-amz-security-token",
        }
    )
    require(
        headers["host"] == request["host"]
        and headers["content-type"]
        == "application/x-www-form-urlencoded; charset=utf-8"
    )
    exact(headers["x-amz-date"], r"[0-9]{8}T[0-9]{6}Z")
    exact(headers["x-amz-security-token"], r"[A-Za-z0-9/+=_-]{1,16384}")
    exact(
        headers["Authorization"],
        rf"AWS4-HMAC-SHA256 Credential=ASIA[A-Z0-9]{{16}}/[0-9]{{8}}/us-east-1/{service}/aws4_request, SignedHeaders=content-type;host;x-amz-date;x-amz-security-token, Signature=[0-9a-f]{{64}}",
    )


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
    elif request["tls"]:
        conn = http.HTTPSConnection(
            request["host"], timeout=timeout, context=ssl.create_default_context()
        )
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


def xml(raw, limit=262144):
    require(len(raw) <= limit)
    text = raw.decode("utf-8")
    require("<!DOCTYPE" not in text.upper() and "<!ENTITY" not in text.upper())
    return ET.fromstring(text)


def one(parent, tag):
    values = parent.findall(tag)
    require(len(values) == 1)
    return values[0]


def text(parent, tag):
    node = one(parent, tag)
    require(len(node) == 0 and isinstance(node.text, str))
    return node.text


def credentials(raw, now):
    value = json_object(raw)
    require(value.get("Code") == "Success" and value.get("Type") == "AWS-HMAC")
    expiry = datetime.datetime.strptime(
        exact(
            value.get("Expiration"),
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        ),
        "%Y-%m-%dT%H:%M:%SZ",
    ).replace(tzinfo=datetime.UTC)
    require(now < expiry <= now + datetime.timedelta(hours=12))
    return {
        "AccessKeyId": exact(value.get("AccessKeyId"), r"ASIA[A-Z0-9]{16}"),
        "SecretAccessKey": exact(value.get("SecretAccessKey"), r"[A-Za-z0-9/+=]{40}"),
        "Token": exact(value.get("Token"), r"[A-Za-z0-9/+=_-]{1,16384}"),
    }


def confirm_sts(raw, instance):
    ns = "{https://sts.amazonaws.com/doc/2011-06-15/}"
    root = xml(raw, 32768)
    require(root.tag == ns + "GetCallerIdentityResponse")
    result = one(root, ns + "GetCallerIdentityResult")
    require(text(result, ns + "Account") == ACCOUNT)
    require(
        text(result, ns + "Arn")
        == f"arn:aws:sts::{ACCOUNT}:assumed-role/{ROLE}/{instance}"
    )


def denied(raw):
    root = xml(raw, 32768)
    require(root.tag == "Response")
    return text(one(one(root, "Errors"), "Error"), "Code") in (
        "UnauthorizedOperation",
        "AccessDenied",
    )


def optional(parent, tag):
    items = parent.findall(tag)
    require(len(items) <= 1)
    return items[0] if items else None


def confirm_ec2(raw, instance, zone, result):
    ns = "{http://ec2.amazonaws.com/doc/2016-11-15/}"
    root = xml(raw)
    require(
        root.tag == ns + "DescribeInstancesResponse"
        and not root.findall(ns + "nextToken")
    )
    reservation = one(one(root, ns + "reservationSet"), ns + "item")
    require(text(reservation, ns + "ownerId") == ACCOUNT)
    item = one(one(reservation, ns + "instancesSet"), ns + "item")
    require(
        text(item, ns + "instanceId") == instance
        and text(one(item, ns + "placement"), ns + "availabilityZone") == zone
        and text(one(item, ns + "iamInstanceProfile"), ns + "arn") == PROFILE
    )
    result["own_instance_identity_matched"] = True
    root_type, root_name = (
        optional(item, ns + "rootDeviceType"),
        optional(item, ns + "rootDeviceName"),
    )
    result["own_root_device_ebs"] = root_type is not None and root_type.text == "ebs"
    maps = optional(item, ns + "blockDeviceMapping")
    if maps is not None and root_name is not None:
        seen, count, all_delete = set(), 0, True
        for mapping in maps.findall(ns + "item"):
            name = text(mapping, ns + "deviceName")
            require(name not in seen)
            seen.add(name)
            ebs = optional(mapping, ns + "ebs")
            if ebs is None:
                continue
            count += 1
            delete = text(ebs, ns + "deleteOnTermination")
            require(delete in ("true", "false"))
            all_delete = all_delete and delete == "true"
            if name == root_name.text:
                result["own_root_delete_on_termination"] = delete == "true"
            else:
                result["own_other_ebs_mapping_present"] = True
        result["own_all_ebs_delete_on_termination"] = count > 0 and all_delete
    tag_set = optional(item, ns + "tagSet")
    if tag_set is None:
        return
    tags = {}
    for tag in tag_set.findall(ns + "item"):
        key = text(tag, ns + "key")
        if key in (
            "Environment",
            "InstanceManagement",
            "EphemeralRunnerFinished",
            "EBSVolumeReplacementRequestTm",
        ):
            require(key not in tags)
            tags[key] = text(tag, ns + "value")
    result["own_instance_environment_matched"] = tags.get("Environment") == "gh-ci"
    result["own_instance_dynamic"] = tags.get("InstanceManagement") == "dynamic"
    result["own_ephemeral_finished_tag_present"] = "EphemeralRunnerFinished" in tags
    result["own_root_replacement_tag_present"] = "EBSVolumeReplacementRequestTm" in tags


def probe(runner, administrator, environment, transport, now):
    result = dict.fromkeys(KEYS, False)
    creds, token = {}, ""
    try:
        result["windows_administrator"] = bool(administrator)
        result["ambient_credential_inputs_present"] = any(
            environment(name) for name in OVERRIDES
        )
        require(not result["ambient_credential_inputs_present"])
        exact(runner, r"i-[0-9a-f]{17}")
        token = exact(
            success(call(transport, "token")).decode("ascii"),
            r"[A-Za-z0-9+/=_-]{1,4096}",
        )
        result["metadata_reachable"] = True
        document = json_object(success(call(transport, "document", "", token)))
        instance = success(call(transport, "instance", "", token)).decode("ascii")
        region = success(call(transport, "region", "", token)).decode("ascii")
        require(
            document.get("accountId") == ACCOUNT
            and document.get("region") == region == REGION
            and document.get("instanceId") == instance == runner
        )
        zone = exact(document.get("availabilityZone"), r"us-east-1[a-z]")
        exact(document.get("imageId"), r"ami-[0-9a-f]{17}")
        result["runner_instance_identity_matched"] = True
        info = json_object(success(call(transport, "profile", "", token)))
        require(
            info.get("Code") == "Success" and info.get("InstanceProfileArn") == PROFILE
        )
        exact(info.get("InstanceProfileId"), r"AIPA[A-Z0-9]{16,64}")
        result["instance_profile_matched"] = True
        require(
            success(call(transport, "role", "", token)).decode("ascii").rstrip("\r\n")
            == ROLE
        )
        result["instance_role_matched"] = True
        creds = credentials(success(call(transport, "credentials", "", token)), now)
        result["temporary_credentials_present"] = True
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        confirm_sts(
            success(call(transport, "sts", instance, "", creds, stamp)), instance
        )
        result["sts_authenticated"] = True
        response = call(transport, "ec2", instance, "", creds, stamp)
        if response[0] == 403 and denied(response[1]):
            result["own_instance_describe_denied"] = True
        else:
            confirm_ec2(success(response), instance, zone, result)
            result["own_instance_describe_allowed"] = True
    except BaseException:
        result["probe_error"] = True
    finally:
        creds.clear()
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
        datetime.datetime.now(datetime.UTC),
    )


def emit(result):
    # Bootstrap errors (including unavailable _ssl) do not require json import.
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
