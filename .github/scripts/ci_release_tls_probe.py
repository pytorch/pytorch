"""Temporary exact-PR Windows replay of the unchanged RC10 installer stage."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import socketserver
import ssl
import subprocess
import tempfile
import threading
from pathlib import Path


REPOSITORY = "pytorch/pytorch"
PR_NUMBER = "205338"  # Exact owned draft; never merge.
HEAD_REF = "gh/d4l3k/347/head"
SOURCE_COMMIT = "d0f2af8b66b20da4baa50644c0cd3f887a2eae2e"
SOURCE_PATH = ".ci/wheel/windows/set_desired_python.sh"
SOURCE_SHA256 = "0e95c944223016625e70b45ca20a84b63074c05a480804a0d8be280d8013c627"
HOST = "www.python.org"
MARKER = b"release-tls-native-marker-20261117\n"
FORBIDDEN_ENV = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    "ACTIONS_ID_TOKEN_REQUEST_URL",
    "GH_TOKEN",
    "GITHUB_TOKEN",
}
PROXY_ENV = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "CURL_CA_BUNDLE",
    "CURL_SSL_BACKEND",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "GIT_SSL_NO_VERIFY",
    "GIT_SSL_CAINFO",
}


def validate_route(env):
    expected = {
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_REF": f"refs/pull/{PR_NUMBER}/merge",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_ACTOR": "d4l3k",
        "GITHUB_TRIGGERING_ACTOR": "d4l3k",
        "CANARY_PR_NUMBER": PR_NUMBER,
        "CANARY_PR_AUTHOR": "d4l3k",
        "CANARY_HEAD_REPO": REPOSITORY,
        "CANARY_HEAD_REF": HEAD_REF,
    }
    if any(env.get(k) != v for k, v in expected.items()):
        raise ValueError("unexpected workflow route")
    if not re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", "")):
        raise ValueError("unexpected workflow SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", env.get("CANARY_HEAD_SHA", "")):
        raise ValueError("unexpected canary SHA")
    if any(env.get(k) for k in FORBIDDEN_ENV):
        raise ValueError("unexpected credential environment")


def source_details(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("RC10 script digest mismatch")
    text = raw.decode("utf-8")
    if (
        text.count('curl --retry 3 -kL "$PYTHON_INSTALLER_URL" --output "$INSTALLER"')
        != 1
    ):
        raise ValueError("unexpected download statement")
    match = re.search(r'3\.13\) PYTHON_FULL="([0-9.]+)"', text)
    if not match:
        raise ValueError("missing Python pin")
    version = match[1]
    path = f"/ftp/python/{version}/python-{version}-amd64.exe"
    return path, raw.replace(b"curl --retry 3 -kL ", b"curl --retry 3 -L ", 1)


def read_headers(sock):
    data = bytearray()
    while not data.endswith(b"\r\n\r\n"):
        part = sock.recv(1)
        if not part:
            raise EOFError("early request EOF")
        data.extend(part)
        if len(data) > 16384:
            raise ValueError("oversized request")
    return bytes(data)


def valid_request(header, method, target):
    return header.split(b"\r\n", 1)[0] == f"{method} {target} HTTP/1.1".encode()


class RequestEvents:
    def __init__(self):
        self.condition = threading.Condition()
        self.events = []

    def append(self, event):
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()

    def require(self, expected):
        with self.condition:
            completed = self.condition.wait_for(
                lambda: len(self.events) >= len(expected), timeout=15
            )
            if not completed or self.events != expected:
                raise ValueError(
                    f"unexpected completed request sequence: {self.events!r}; "
                    f"expected {expected!r}"
                )
            return self.events.copy()


@contextlib.contextmanager
def local_proxy(context, request_path, payload):
    events = RequestEvents()

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(15)
            event = "request-error"
            try:
                header = read_headers(self.request)
                if not valid_request(header, "CONNECT", f"{HOST}:443"):
                    event = "unexpected-connect"
                    return
                self.request.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                try:
                    stream = context.wrap_socket(self.request, server_side=True)
                except (ssl.SSLError, ConnectionResetError):
                    event = "tls-rejected"
                    return
                with stream:
                    header = read_headers(stream)
                    if not valid_request(header, "GET", request_path):
                        event = "unexpected-get"
                        return
                    stream.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                        + f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                        + payload
                    )
                    event = "payload-served"
            except (OSError, EOFError, ValueError):
                event = "request-error"
            finally:
                events.append(event)

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = False
        block_on_close = True

    with Server(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}", events
        finally:
            server.shutdown()
            worker.join()


def child_environment(original, proxy, marker):
    env = {k: v for k, v in original.items() if k not in PROXY_ENV | FORBIDDEN_ENV}
    env.update(
        {
            "DESIRED_PYTHON": "3.13",
            "RLS05_MARKER": str(marker),
            "HOME": str(marker.parent),
            "USERPROFILE": str(marker.parent),
            "APPDATA": str(marker.parent),
            "XDG_CONFIG_HOME": str(marker.parent),
            "CURL_HOME": str(marker.parent),
            "HTTPS_PROXY": proxy,
            "https_proxy": proxy,
            "HTTP_PROXY": "",
            "http_proxy": "",
            "ALL_PROXY": "",
            "all_proxy": "",
            "NO_PROXY": "",
            "no_proxy": "",
        }
    )
    return env


def cmd_path(path):
    value = str(path)
    if any(char in value for char in '"%!&|<>^\r\n'):
        raise ValueError("unsupported command path")
    return f'"{value}"'


def run(argv, **kwargs):
    return subprocess.run(argv, capture_output=True, timeout=180, **kwargs)


def compile_marker(directory):
    program_files = Path(os.environ["ProgramFiles(x86)"])
    vswhere = program_files / "Microsoft Visual Studio/Installer/vswhere.exe"
    result = run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find",
            r"VC\Auxiliary\Build\vcvarsall.bat",
        ],
        check=True,
        text=True,
    )
    paths = result.stdout.strip().splitlines()
    if len(paths) != 1 or not Path(paths[0]).is_file():
        raise ValueError("unexpected MSVC installation")
    source = Path(__file__).with_name("ci_release_tls_marker.c").resolve()
    exe, obj = directory / "payload.exe", directory / "payload.obj"
    script = directory / "build-marker.cmd"
    script.write_bytes(
        (
            "@echo off\r\n"
            f"call {cmd_path(paths[0])} x64 >nul\r\n"
            "if errorlevel 1 exit /b 1\r\n"
            f"cl.exe /nologo /W4 /WX /O2 {cmd_path(source)} "
            f"/Fe:{cmd_path(exe)} /Fo:{cmd_path(obj)} /link /INCREMENTAL:NO\r\n"
            "exit /b %errorlevel%\r\n"
        ).encode()
    )
    run(["cmd.exe", "/d", "/v:off", "/c", str(script)], check=True, cwd=directory)
    payload = exe.read_bytes()
    if not payload.startswith(b"MZ") or not 512 <= len(payload) <= 1024 * 1024:
        raise ValueError("unexpected PE payload")
    return exe, payload


def find_tools():
    git_root = Path(os.environ["ProgramFiles"]) / "Git"
    bash = git_root / "bin/bash.exe"
    openssl = next(
        (
            p
            for p in (
                git_root / "usr/bin/openssl.exe",
                git_root / "mingw64/bin/openssl.exe",
            )
            if p.is_file()
        ),
        None,
    )
    if not bash.is_file() or openssl is None or shutil.which("powershell.exe") is None:
        raise ValueError("required stock Windows tools unavailable")
    return bash, openssl


def bash_path(bash, path):
    value = run(
        [
            str(bash),
            "--noprofile",
            "--norc",
            "-c",
            'cygpath -u "$1"',
            "path",
            str(path),
        ],
        check=True,
        text=True,
    ).stdout.strip()
    if not value.startswith("/") or "\n" in value or "\r" in value:
        raise ValueError("unexpected Git Bash path")
    return value


def execute(source_root):
    validate_route(os.environ)
    if os.name != "nt":
        raise ValueError("native Windows required")
    source_root = source_root.resolve(strict=True)
    source = source_root / SOURCE_PATH
    raw = source.read_bytes()
    request_path, hardened = source_details(raw)
    strict = source.with_name(".rls05-strict-installer.sh")
    downloaded = source_root / ".ci/pytorch/windows/python-amd64.exe"
    if strict.exists() or downloaded.exists():
        raise ValueError("owned test path already exists")
    bash, openssl = find_tools()
    source_ref = run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], check=True, text=True
    ).stdout.strip()
    if source_ref != SOURCE_COMMIT:
        raise ValueError("unexpected source checkout")
    canary_root = Path(__file__).resolve().parents[2]
    canary_ref = run(
        ["git", "-C", str(canary_root), "rev-parse", "HEAD"], check=True, text=True
    ).stdout.strip()
    if canary_ref != os.environ["CANARY_HEAD_SHA"]:
        raise ValueError("unexpected canary checkout")

    with tempfile.TemporaryDirectory(prefix="release-tls-native-") as temporary:
        temp = Path(temporary)
        marker = temp / "execution-marker"
        exe, payload = compile_marker(temp)
        selftest_env = child_environment(os.environ, "", marker)
        selftest = run([str(exe)], env=selftest_env)
        if selftest.returncode != 73 or marker.read_bytes() != MARKER:
            raise ValueError("native marker self-test failed")
        marker.unlink()
        cert, key = temp / "cert.pem", temp / "key.pem"
        run(
            [
                str(openssl),
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-keyout",
                str(key),
                "-out",
                str(cert),
                "-days",
                "1",
                "-subj",
                f"/CN={HOST}",
                "-addext",
                f"subjectAltName=DNS:{HOST}",
            ],
            check=True,
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        strict.write_bytes(hardened)
        try:
            strict_arg, source_arg = bash_path(bash, strict), bash_path(bash, source)
            with local_proxy(context, request_path, payload) as (proxy, events):
                env = child_environment(os.environ, proxy, marker)
                negative = run(
                    [str(bash), "--noprofile", "--norc", strict_arg], env=env
                )
                if negative.returncode != 1 or marker.exists():
                    raise ValueError("strict TLS negative control failed")
                if downloaded.exists() and downloaded.stat().st_size:
                    raise ValueError("strict TLS accepted response bytes")
                if negative.stderr.count(b"curl: (60)") != 3:
                    raise ValueError("strict TLS did not report certificate rejection")
                events.require(["tls-rejected"] * 3)
                positive = run(
                    [str(bash), "--noprofile", "--norc", source_arg], env=env
                )
                if positive.returncode != 1 or marker.read_bytes() != MARKER:
                    raise ValueError(
                        "unchanged installer did not execute native marker"
                    )
                completed_events = events.require(
                    ["tls-rejected"] * 3 + ["payload-served"] * 3
                )
                if downloaded.read_bytes() != payload:
                    raise ValueError("downloaded PE differs")
                evidence = {
                    "workflow_sha": os.environ["GITHUB_SHA"],
                    "canary_sha": canary_ref,
                    "source_commit": source_ref,
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                    "payload_bytes": len(payload),
                    "negative_exit": negative.returncode,
                    "positive_exit": positive.returncode,
                    "marker_matches": True,
                    "events": completed_events,
                    "curl": run(
                        [str(bash), "--noprofile", "--norc", "-c", "curl --version"],
                        check=True,
                        text=True,
                    ).stdout.splitlines()[0],
                    "powershell": run(
                        [
                            "powershell.exe",
                            "-NoProfile",
                            "-NonInteractive",
                            "-Command",
                            "$PSVersionTable.PSVersion.ToString()",
                        ],
                        check=True,
                        text=True,
                    ).stdout.strip(),
                }
        finally:
            strict.unlink(missing_ok=True)
            downloaded.unlink(missing_ok=True)
        if source.read_bytes() != raw:
            raise ValueError("source script changed")
    evidence["temporary_resources_removed"] = (
        not Path(temporary).exists() and not strict.exists() and not downloaded.exists()
    )
    if not evidence["temporary_resources_removed"]:
        raise ValueError("temporary cleanup failed")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    print("RLS05_RESULT=" + json.dumps(execute(args.source_root), sort_keys=True))


if __name__ == "__main__":
    main()
