#!/usr/bin/env python3
"""Test-only, read-only own-TPU metadata/permission audit. Never log credentials."""

import dataclasses
import http.client
import json
import os
import re
import resource
import signal
import ssl
import time
from pathlib import Path
from urllib.parse import quote, urlencode


META = "169.254.169.254"
PREFIX = "/computeMetadata/v1/"
FIELDS = "id,name,zone,selfLink,serviceAccounts(email,scopes)"
META_KEYS = (
    "project/project-id",
    "project/numeric-project-id",
    "instance/id",
    "instance/name",
    "instance/zone",
    "instance/service-accounts/default/email",
    "instance/service-accounts/default/scopes",
)
TOKEN_KEY = "instance/service-accounts/default/token"
PERMISSIONS = {
    "project": (
        "compute.instances.get",
        "compute.instances.setMetadata",
        "iam.serviceAccounts.actAs",
    ),
    "instance": (
        "compute.instances.get",
        "compute.instances.setMetadata",
        "compute.instances.setServiceAccount",
        "compute.instances.osLogin",
        "compute.instances.osAdminLogin",
    ),
    "service_account": (
        "iam.serviceAccounts.get",
        "iam.serviceAccounts.actAs",
        "iam.serviceAccounts.getAccessToken",
        "iam.serviceAccounts.signBlob",
        "iam.serviceAccounts.setIamPolicy",
    ),
}
HOSTS = {
    "compute.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
}


class Boundary(Exception):
    """A fixed-scope or schema boundary; details must never reach output."""


def require(value):
    if not value:
        raise Boundary


def integer(value):
    require(isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{0,19}", value))
    require(int(value) < 2**64)
    return value


def component(value, pattern):
    require(isinstance(value, str) and re.fullmatch(pattern, value))
    return value


def parse_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result)
            result[key] = value
        return result

    return json.loads(data.decode("utf-8"), object_pairs_hook=unique)


@dataclasses.dataclass(frozen=True)
class Identity:
    project: str
    project_number: str
    instance_id: str
    name: str
    zone: str
    email: str
    scopes: tuple

    @classmethod
    def parse(cls, values):
        project, number, iid, name, zone, email, scopes = values
        component(project, r"[a-z][a-z0-9-]{4,28}[a-z0-9]")
        integer(number)
        integer(iid)
        component(name, r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?")
        require(zone.startswith(f"projects/{number}/zones/"))
        zone = zone.removeprefix(f"projects/{number}/zones/")
        component(zone, r"[a-z][a-z0-9-]{0,61}[a-z0-9]")
        component(
            email,
            r"[A-Za-z0-9._-]+@(?:[a-z0-9-]+\.iam|developer)\.gserviceaccount\.com",
        )
        require(isinstance(scopes, str) and len(scopes) <= 8192)
        items = tuple(sorted(scopes.splitlines()))
        require(items and len(items) <= 64 and len(set(items)) == len(items))
        require(
            all(
                re.fullmatch(r"https://www\.googleapis\.com/auth/[A-Za-z0-9._/-]+", x)
                for x in items
            )
        )
        return cls(project, number, iid, name, zone, email, items)

    def specs(self):
        p, z, n, e = (
            quote(x, safe="") for x in (self.project, self.zone, self.name, self.email)
        )
        instance = f"/compute/v1/projects/{p}/zones/{z}/instances/{n}"
        return {
            "compute": (
                "compute.googleapis.com",
                "GET",
                instance + "?" + urlencode({"fields": FIELDS}),
                None,
            ),
            "project": (
                "cloudresourcemanager.googleapis.com",
                "POST",
                f"/v1/projects/{p}:testIamPermissions",
                PERMISSIONS["project"],
            ),
            "instance": (
                "compute.googleapis.com",
                "POST",
                instance + "/testIamPermissions",
                PERMISSIONS["instance"],
            ),
            "service_account": (
                "iam.googleapis.com",
                "POST",
                f"/v1/projects/{p}/serviceAccounts/{e}:testIamPermissions",
                PERMISSIONS["service_account"],
            ),
        }


@dataclasses.dataclass(frozen=True)
class Reply:
    status: int
    flavor: str
    content_type: str
    data: bytes


class Transport:
    def __init__(self):
        self.deadline = time.monotonic() + 60
        self.identity = None

    def remaining(self, maximum):
        left = self.deadline - time.monotonic()
        require(left > 0)
        return min(maximum, left)

    def bind(self, identity):
        require(self.identity is None and isinstance(identity, Identity))
        canonical = Identity.parse(
            (
                identity.project,
                identity.project_number,
                identity.instance_id,
                identity.name,
                f"projects/{identity.project_number}/zones/{identity.zone}",
                identity.email,
                "\n".join(identity.scopes),
            )
        )
        require(canonical == identity)
        self.identity = identity

    def metadata(self, key, flavor=True):
        require(key in META_KEYS or key == TOKEN_KEY)
        require(flavor or key == "instance/id")
        limit = 16384 if key == TOKEN_KEY else 8192 if key.endswith("/scopes") else 2048
        return self._perform(META, "GET", PREFIX + key, None, None, flavor, limit)

    def api(self, route, token=None):
        require(self.identity is not None and route in self.identity.specs())
        require(token is not None or route == "compute")
        host, method, path, permissions = self.identity.specs()[route]
        body = (
            None
            if permissions is None
            else json.dumps(
                {"permissions": permissions}, separators=(",", ":")
            ).encode()
        )
        return self._perform(host, method, path, body, token, False, 65536)

    def _perform(self, host, method, path, body, token, flavor, limit):
        require(host == META or host in HOSTS)
        require(
            method in {"GET", "POST"}
            and path.startswith("/")
            and not path.startswith("//")
        )
        headers = {
            "Accept": "application/json"
            if host != META
            else "text/plain, application/json"
        }
        if host == META:
            require(method == "GET" and body is None and token is None)
            if flavor:
                headers["Metadata-Flavor"] = "Google"
            conn = http.client.HTTPConnection(host, timeout=self.remaining(3))
        else:
            require(not flavor)
            # Load only the platform's compiled-in trust paths, not SSL_CERT_FILE/DIR.
            paths = ssl.get_default_verify_paths()
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.load_verify_locations(
                cafile=paths.openssl_cafile, capath=paths.openssl_capath
            )
            conn = http.client.HTTPSConnection(
                host, context=context, timeout=self.remaining(3)
            )
            if token is not None:
                require(
                    isinstance(token, str)
                    and token
                    and not any(c.isspace() for c in token)
                )
                headers["Authorization"] = "Bearer " + token
            if body is not None:
                headers["Content-Type"] = "application/json"
        try:
            conn.connect()
            require(conn.sock is not None)
            if host == META:
                require(conn.sock.getpeername()[0] == META)
            conn.sock.settimeout(self.remaining(5))
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            require(not 300 <= response.status < 400)
            data = response.read(limit + 1)
            require(len(data) <= limit)
            self.remaining(60)
            return Reply(
                response.status,
                response.getheader("Metadata-Flavor", ""),
                response.getheader("Content-Type", "").split(";", 1)[0],
                data,
            )
        finally:
            conn.close()


def text(reply):
    require(
        reply.status == 200
        and reply.flavor == "Google"
        and reply.content_type in {"text/plain", "application/text"}
    )
    value = reply.data.decode("utf-8").strip()
    require(value and "\x00" not in value and "\r" not in value)
    return value


def token_value(reply):
    require(
        reply.status == 200
        and reply.flavor == "Google"
        and reply.content_type == "application/json"
    )
    obj = parse_json(reply.data)
    require(
        isinstance(obj, dict)
        and set(obj) == {"access_token", "token_type", "expires_in"}
    )
    require(
        obj["token_type"] == "Bearer"
        and type(obj["expires_in"]) is int
        and 0 < obj["expires_in"] <= 86400
    )
    token = obj["access_token"]
    require(
        isinstance(token, str)
        and 0 < len(token) <= 8192
        and not any(c.isspace() for c in token)
    )
    return token


def resource_matches(url, identity, suffix):
    if not isinstance(url, str):
        return False
    for host in ("www.googleapis.com", "compute.googleapis.com"):
        for project in (identity.project, identity.project_number):
            if (
                url
                == f"https://{host}/compute/v1/projects/{project}/zones/{identity.zone}{suffix}"
            ):
                return True
    return False


def check_instance(reply, identity):
    require(reply.status == 200 and reply.content_type == "application/json")
    obj = parse_json(reply.data)
    require(
        isinstance(obj, dict)
        and obj.get("id") == identity.instance_id
        and obj.get("name") == identity.name
    )
    require(resource_matches(obj.get("zone"), identity, ""))
    require(
        resource_matches(obj.get("selfLink"), identity, "/instances/" + identity.name)
    )
    accounts = obj.get("serviceAccounts")
    require(
        isinstance(accounts, list)
        and len(accounts) == 1
        and isinstance(accounts[0], dict)
    )
    account = accounts[0]
    require(
        account.get("email") == identity.email
        and isinstance(account.get("scopes"), list)
    )
    require(all(isinstance(x, str) for x in account["scopes"]))
    require(tuple(sorted(account["scopes"])) == identity.scopes)


def status(reply):
    if reply.status == 200:
        return "allowed"
    if reply.status == 403:
        try:
            obj = parse_json(reply.data)
            details = obj.get("error", {}).get("details", [])
            if isinstance(details, list) and any(
                isinstance(x, dict) and x.get("reason") == "SERVICE_DISABLED"
                for x in details
            ):
                return "service_disabled"
        except (ValueError, TypeError, AttributeError, Boundary):
            pass
        return "access_denied"
    return {404: "not_found", 429: "rate_limited", 401: "unauthenticated"}.get(
        reply.status, "unexpected_status"
    )


def decisions(reply, permissions):
    require(reply.content_type == "application/json")
    obj = parse_json(reply.data)
    require(isinstance(obj, dict) and set(obj) <= {"permissions"})
    found = obj.get("permissions", [])
    require(isinstance(found, list) and all(isinstance(x, str) for x in found))
    require(len(found) == len(set(found)) and set(found) <= set(permissions))
    return {permission: permission in found for permission in permissions}


def initial_result():
    keys = (
        "scope_valid",
        "metadata_http_observed",
        "metadata_origin_matches",
        "metadata_server_cryptographically_authenticated",
        "metadata_identity_stable",
        "google_vm_indicators_match",
        "cloud_platform_scope",
        "token_received",
        "compute_request_authenticated",
        "instance_identity_matches",
        "attached_sa_matches",
        "negative_control_passed",
        "cloud_probe_skipped",
        "cleanup_confirmed",
        "probe_error",
    )
    result = dict.fromkeys(keys, False)
    result["token_expiration_category"] = "unknown"
    result["compute_status"] = "not_attempted"
    result["permissions"] = {
        key: {
            "status": "not_attempted",
            "decision_received": False,
            "allowed": dict.fromkeys(values, False),
        }
        for key, values in PERMISSIONS.items()
    }
    return result


def context_valid(env):
    return (
        env.get("GITHUB_REPOSITORY") == "pytorch/pytorch"
        and env.get("GITHUB_EVENT_NAME") == "pull_request"
        and env.get("GITHUB_ACTOR") in {"ptdbot", "d4l3k"}
        and bool(re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", "")))
        and bool(re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ATTEMPT", "")))
        and bool(re.fullmatch(r"[A-Za-z0-9._-]{1,200}", env.get("RUNNER_NAME", "")))
        and bool(re.fullmatch(r"[0-9a-f]{40}", env.get("AUDIT_HEAD_SHA", "")))
    )


def probe(client, env):
    result = initial_result()
    token = None
    try:
        require(context_valid(env))
        result["scope_valid"] = True
        # This check reads only standard nonsecret, own-host provider indicators.
        try:
            result["google_vm_indicators_match"] = (
                Path("/sys/class/dmi/id/sys_vendor").read_text().strip() == "Google"
                and Path("/sys/class/dmi/id/product_name").read_text().strip()
                == "Google Compute Engine"
            )
        except OSError:
            pass
        control = client.metadata("instance/id", False)
        require(control.status in {400, 401, 403})
        values = tuple(text(client.metadata(key)) for key in META_KEYS)
        result["metadata_http_observed"] = True
        result["metadata_origin_matches"] = True
        identity = Identity.parse(values)
        require(tuple(text(client.metadata(key)) for key in META_KEYS) == values)
        result["metadata_identity_stable"] = True
        result["cloud_platform_scope"] = (
            "https://www.googleapis.com/auth/cloud-platform" in identity.scopes
        )
        client.bind(identity)
        unsigned = client.api("compute")
        require(unsigned.status in {401, 403})
        result["negative_control_passed"] = True
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        token = token_value(client.metadata(TOKEN_KEY))
        result["token_received"] = True
        result["token_expiration_category"] = "provider_expiration"
        own = client.api("compute", token)
        result["compute_status"] = status(own)
        if own.status != 200:
            return result
        check_instance(own, identity)
        result["compute_request_authenticated"] = True
        result["instance_identity_matches"] = True
        result["attached_sa_matches"] = True
        for route, permissions in PERMISSIONS.items():
            reply = client.api(route, token)
            item = result["permissions"][route]
            item["status"] = status(reply)
            require(reply.status in {200, 401, 403, 404, 429})
            if reply.status != 200:
                continue
            try:
                item["allowed"] = decisions(reply, permissions)
                item["decision_received"] = True
            except (ValueError, TypeError, Boundary):
                item["status"] = "schema_error"
                raise Boundary from None
    except (Exception, KeyboardInterrupt):
        result["probe_error"] = True
    finally:
        token = None
        result["cloud_probe_skipped"] = not result["instance_identity_matches"]
        result["cleanup_confirmed"] = (
            True  # No token file, environment export, or created resource.
        )
    return result


def main():
    def stop(_signum, _frame):
        raise Boundary

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        result = probe(Transport(), os.environ)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return int(result["probe_error"])


if __name__ == "__main__":
    raise SystemExit(main())
