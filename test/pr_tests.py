#!/usr/bin/env python3
"""List fully qualified Python tests reported by CI for a pull request.

This script does not infer tests from workflow YAML or execute them. It resolves
the pull request's current head SHA, finds its completed GitHub Actions runs and
jobs, and locates each job's ``test-reports-*.zip`` artifact. The public PyTorch
S3 mirror is preferred, with GitHub artifacts used as a fallback.

The artifacts may contain JUnit XML reports. For every Python ``<testcase>`` in
those reports, the script combines its ``file``, ``classname``, and ``name``
attributes into a node ID such as
``test/test_as_strided.py::TestAsStrided::test_subset_property``. It also
reconstructs module-level tests and nested classes, preserves parameterized test
names, deduplicates the results, and prints them in sorted order.

The collection fails instead of returning partial results when completed test
jobs are missing their report artifacts. CI artifacts are retained for a limited
time, so an older pull request may no longer have enough data to produce a
complete list.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import xml.parsers.expat as expat
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from email.message import Message


DEFAULT_REPOSITORY = "pytorch/pytorch"
GITHUB_API = "https://api.github.com"
S3_BUCKET_URL = "https://gha-artifacts.s3.amazonaws.com"
USER_AGENT = "pytorch-pr-tests"

_MAX_API_BYTES = 8 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_REPORT_BYTES = 64 * 1024 * 1024
_MAX_REPORT_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_REPORT_COLLECTION_BYTES = 4 * 1024 * 1024 * 1024
_MAX_ARTIFACT_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_ARTIFACTS = 1000
_MAX_DOWNLOAD_WORKERS = 4
_MAX_XML_DEPTH = 64
_MAX_REPORTS_PER_ARTIFACT = 10_000
_MAX_REPORTS = 100_000
_MAX_TESTCASES_PER_ARTIFACT = 1_000_000
_MAX_TESTCASES = 10_000_000
_MAX_ZIP_MEMBERS = 20_000
_ARTIFACT_UPLOAD_GRACE = timedelta(minutes=5)
_ARTIFACT_JOB_ID = re.compile(r"_(\d+)\.zip$")
_TEST_JOB_NAME = re.compile(r" / test(?:-osdc)? \((?P<matrix>[^/]*)\)$")
_TEST_FILE = re.compile(r"[A-Za-z0-9_./-]+\.py")
_TEST_FILE_ALIASES = {
    "test_cpp_extensions_aot_ninja.py": "test_cpp_extensions_aot.py",
    "test_cpp_extensions_aot_no_ninja.py": "test_cpp_extensions_aot.py",
}
_IGNORED_RUN_CONCLUSIONS = {"action_required", "skipped"}
_IGNORED_JOB_CONCLUSIONS = {"action_required", "cancelled", "skipped"}
_TRANSIENT_HTTP_ERRORS = {429, 500, 502, 503, 504}


class PRTestsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    source: str
    job_id: int | None
    expired: bool = False
    updated_at: datetime | None = None


@dataclass(frozen=True)
class ExpectedJob:
    job_id: int
    run_attempt: int
    latest_run_attempt: int
    started_at: datetime
    completed_at: datetime


@dataclass(frozen=True)
class JobIdentity:
    key: tuple[int, str]
    run_attempt: int | None


class _Budget:
    def __init__(self, limit: int, error: str) -> None:
        self.limit = limit
        self.error = error
        self.used = 0
        self.lock = threading.Lock()

    def reserve(self, size: int) -> None:
        with self.lock:
            if self.used + size > self.limit:
                raise PRTestsError(self.error)
            self.used += size


@dataclass(frozen=True)
class CollectionResult:
    tests: frozenset[str]
    workflow_runs: int
    artifacts: int
    reports: int


def _parse_timestamp(value: Any, source: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PRTestsError(f"{source} contained an invalid timestamp")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise PRTestsError(f"{source} contained an invalid timestamp") from error
    if timestamp.tzinfo is None:
        raise PRTestsError(f"{source} contained a timestamp without a timezone")
    return timestamp


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        old_host = urllib.parse.urlsplit(req.full_url).netloc
        new_url = urllib.parse.urlsplit(newurl)
        if new_url.scheme != "https":
            raise PRTestsError("Refusing to follow a non-HTTPS redirect")
        new_host = new_url.netloc
        if old_host != new_host:
            redirected.remove_header("Authorization")
        return redirected


def _open_request(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.build_opener(_SafeRedirectHandler()).open(
        request, timeout=timeout
    )


def _retry_delay(headers: Message | None, attempt: int) -> float:
    if headers is not None:
        retry_after = headers.get("Retry-After")
        if retry_after is not None:
            try:
                return min(float(retry_after), 30.0)
            except ValueError:
                pass
    return min(0.5 * (2**attempt), 8.0)


def _request_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    retries: int = 3,
    timeout: float = 60.0,
    max_bytes: int = _MAX_API_BYTES,
) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    for attempt in range(retries + 1):
        try:
            with _open_request(request, timeout) as response:
                content_length = getattr(response, "headers", {}).get("Content-Length")
                if content_length is not None:
                    try:
                        length = int(content_length)
                    except ValueError as error:
                        raise PRTestsError(
                            f"Invalid Content-Length while requesting {url}"
                        ) from error
                    if length > max_bytes:
                        raise PRTestsError(
                            f"Response exceeded {max_bytes} bytes while requesting {url}"
                        )
                data = response.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise PRTestsError(
                        f"Response exceeded {max_bytes} bytes while requesting {url}"
                    )
                return data
        except urllib.error.HTTPError as error:
            if error.code not in _TRANSIENT_HTTP_ERRORS or attempt == retries:
                raise PRTestsError(
                    f"HTTP {error.code} while requesting {url}"
                ) from error
            time.sleep(_retry_delay(error.headers, attempt))
        except urllib.error.URLError as error:
            if attempt == retries:
                raise PRTestsError(
                    f"Unable to request {url}: {error.reason}"
                ) from error
            time.sleep(_retry_delay(None, attempt))
        except (http.client.IncompleteRead, OSError) as error:
            if attempt == retries:
                raise PRTestsError(
                    f"Unable to read response from {url}: {error}"
                ) from error
            time.sleep(_retry_delay(None, attempt))
    raise AssertionError("unreachable")


def _github_headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_json(url: str, token: str | None) -> dict[str, Any]:
    try:
        value = json.loads(_request_bytes(url, headers=_github_headers(token)))
    except json.JSONDecodeError as error:
        raise PRTestsError(f"GitHub returned invalid JSON for {url}") from error
    if not isinstance(value, dict):
        raise PRTestsError(f"GitHub returned an unexpected response for {url}")
    return value


def _github_token() -> str | None:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token", "--hostname", "github.com"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    token = result.stdout.strip()
    return token if result.returncode == 0 and token else None


def parse_pr(value: str) -> tuple[str, int]:
    if value.isdecimal():
        return DEFAULT_REPOSITORY, int(value)

    parsed = urllib.parse.urlsplit(value)
    match = re.fullmatch(r"/([^/]+)/([^/]+)/pull/(\d+)(?:/.*)?", parsed.path)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.netloc != "github.com"
        or not match
    ):
        raise PRTestsError(
            "PR must be a number or a URL such as "
            "https://github.com/pytorch/pytorch/pull/190437"
        )
    owner, repository, number = match.groups()
    return f"{owner}/{repository}", int(number)


def _pr_info(repository: str, number: int, token: str | None) -> tuple[str, str]:
    value = _github_json(f"{GITHUB_API}/repos/{repository}/pulls/{number}", token)
    try:
        sha = value["head"]["sha"]
        canonical_repository = value["base"]["repo"]["full_name"]
    except (KeyError, TypeError) as error:
        raise PRTestsError(
            "GitHub's PR response did not contain repository information"
        ) from error
    if not isinstance(sha, str) or not sha:
        raise PRTestsError("GitHub's PR response contained an invalid head SHA")
    if not isinstance(canonical_repository, str) or "/" not in canonical_repository:
        raise PRTestsError("GitHub's PR response contained an invalid repository")
    return canonical_repository, sha


def _workflow_runs(
    repository: str, head_sha: str, token: str | None
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    total_count: int | None = None
    page = 1
    while True:
        query = urllib.parse.urlencode(
            {"head_sha": head_sha, "per_page": 100, "page": page}
        )
        value = _github_json(
            f"{GITHUB_API}/repos/{repository}/actions/runs?{query}", token
        )
        if total_count is None:
            total_count = value.get("total_count")
            if not isinstance(total_count, int):
                raise PRTestsError(
                    "GitHub's workflow response did not contain a total count"
                )
            if total_count > 1000:
                raise PRTestsError(
                    "GitHub found more than 1,000 workflow runs for the PR head"
                )
        page_runs = value.get("workflow_runs")
        if not isinstance(page_runs, list):
            raise PRTestsError("GitHub's workflow response did not contain a run list")
        runs.extend(page_runs)
        if len(page_runs) < 100:
            break
        page += 1
    if len(runs) != total_count:
        raise PRTestsError("GitHub returned an incomplete workflow run list")
    return runs


def _workflow_jobs(
    repository: str, run_id: int, token: str | None
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"filter": "all", "per_page": 100, "page": page})
        value = _github_json(
            f"{GITHUB_API}/repos/{repository}/actions/runs/{run_id}/jobs?{query}",
            token,
        )
        page_jobs = value.get("jobs")
        if not isinstance(page_jobs, list):
            raise PRTestsError("GitHub's workflow response did not contain a job list")
        jobs.extend(page_jobs)
        if len(page_jobs) < 100:
            break
        page += 1
    return jobs


def _artifact_job_id(name: str) -> int | None:
    match = _ARTIFACT_JOB_ID.search(name)
    return int(match.group(1)) if match else None


def _s3_artifacts(repository: str, run_id: int) -> list[Artifact]:
    prefix = f"{repository}/{run_id}/"
    continuation_token: str | None = None
    artifacts: list[Artifact] = []
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": 1000}
        if continuation_token is not None:
            params["continuation-token"] = continuation_token
        url = f"{S3_BUCKET_URL}/?{urllib.parse.urlencode(params)}"
        try:
            root = ET.fromstring(_request_bytes(url))
        except ET.ParseError as error:
            raise PRTestsError("S3 returned an invalid artifact listing") from error

        for contents in root.findall("s3:Contents", namespace):
            key = contents.findtext("s3:Key", namespaces=namespace)
            if key is None:
                continue
            relative = key.removeprefix(prefix)
            parts = relative.split("/", 2)
            if (
                len(parts) != 3
                or not parts[0].isdecimal()
                or parts[1] != "artifact"
                or not parts[2].startswith("test-reports-")
                or not parts[2].endswith(".zip")
            ):
                continue
            artifacts.append(
                Artifact(
                    name=parts[2],
                    url=f"{S3_BUCKET_URL}/{urllib.parse.quote(key, safe='/')}",
                    source="s3",
                    job_id=_artifact_job_id(parts[2]),
                    updated_at=_parse_timestamp(
                        contents.findtext("s3:LastModified", namespaces=namespace),
                        "S3",
                    ),
                )
            )

        is_truncated = root.findtext("s3:IsTruncated", namespaces=namespace)
        if is_truncated != "true":
            break
        continuation_token = root.findtext(
            "s3:NextContinuationToken", namespaces=namespace
        )
        if continuation_token is None:
            raise PRTestsError("S3 truncated an artifact listing without a next page")
    return artifacts


def _github_artifacts(
    repository: str, run_id: int, token: str | None
) -> list[Artifact]:
    artifacts: list[Artifact] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"per_page": 100, "page": page})
        value = _github_json(
            f"{GITHUB_API}/repos/{repository}/actions/runs/{run_id}/artifacts?{query}",
            token,
        )
        page_artifacts = value.get("artifacts")
        if not isinstance(page_artifacts, list):
            raise PRTestsError("GitHub's artifact response did not contain a list")
        for item in page_artifacts:
            if not isinstance(item, dict):
                raise PRTestsError("GitHub returned an invalid artifact entry")
            name = item.get("name")
            artifact_id = item.get("id")
            if (
                not isinstance(name, str)
                or not name.startswith("test-reports-")
                or not isinstance(artifact_id, int)
            ):
                continue
            artifacts.append(
                Artifact(
                    name=name,
                    url=f"{GITHUB_API}/repos/{repository}/actions/artifacts/{artifact_id}/zip",
                    source="github",
                    job_id=_artifact_job_id(name),
                    expired=item.get("expired") is True,
                    updated_at=_parse_timestamp(item.get("updated_at"), "GitHub"),
                )
            )
        if len(page_artifacts) < 100:
            break
        page += 1
    return artifacts


def _deduplicate_artifacts(
    s3_artifacts: Iterable[Artifact], github_artifacts: Iterable[Artifact]
) -> tuple[list[Artifact], list[Artifact]]:
    selected: dict[str, Artifact] = {}
    expired: list[Artifact] = []
    for artifact in s3_artifacts:
        selected[artifact.url] = artifact
    for artifact in github_artifacts:
        if artifact.expired:
            expired.append(artifact)
            continue
        selected[artifact.url] = artifact
    return list(selected.values()), expired


def _qualified_test_name(
    testcase: ET.Element,
    artifact_name: str,
    report_module: str | None = None,
    report_directory: str | None = None,
) -> str | None:
    file = testcase.get("file")
    if file is None:
        return None
    file = file.replace("\\", "/")
    if len(file) > 4096 or not file.isprintable():
        raise PRTestsError(f"Artifact {artifact_name} contains an invalid test file")
    if not file.endswith(".py"):
        return None
    if report_directory and not file.startswith(f"{report_directory}/"):
        file = f"{report_directory}/{file}"
        if len(file) > 4096 or not file.isprintable():
            raise PRTestsError(
                f"Artifact {artifact_name} contains an invalid test file"
            )
    parent, separator, basename = file.rpartition("/")
    if basename in _TEST_FILE_ALIASES:
        file = f"{parent}{separator}{_TEST_FILE_ALIASES[basename]}"

    name = testcase.get("name")
    if name is None or not name or len(name) > 4096 or not name.isprintable():
        raise PRTestsError(f"Artifact {artifact_name} contains an invalid test name")

    classname = testcase.get("classname", "")
    if len(classname) > 4096 or (classname and not classname.isprintable()):
        raise PRTestsError(f"Artifact {artifact_name} contains an invalid classname")

    class_components: list[str] = []
    if classname:
        normalized_classname = classname.removeprefix("test.")
        candidate = normalized_classname.rsplit(".", 1)[-1]
        raw_module = file[:-3].replace("/", ".")
        module_names = {
            raw_module,
            raw_module.removeprefix("test."),
            raw_module.rsplit(".", 1)[-1],
        }
        if normalized_classname in module_names or candidate in module_names:
            pass
        elif report_module and normalized_classname.startswith(f"{report_module}."):
            file = f"{report_module.replace('.', '/')}.py"
            class_components = normalized_classname[len(report_module) + 1 :].split(".")
        elif report_module and report_module == f"{raw_module}.{candidate}":
            file = f"{report_module.replace('.', '/')}.py"
        elif report_module and raw_module.startswith(f"{report_module}."):
            nested_classes = raw_module[len(report_module) + 1 :].split(".")
            if all(
                part[:1].isupper() or part.startswith("_") for part in nested_classes
            ):
                file = f"{report_module.replace('.', '/')}.py"
                class_components = [*nested_classes, candidate]
            else:
                class_components = [candidate]
        elif normalized_classname.startswith(f"{raw_module}."):
            class_components = normalized_classname[len(raw_module) + 1 :].split(".")
        else:
            if candidate[:1].islower():
                file = f"{file[:-3]}/{candidate}.py"
            else:
                class_components = [candidate]

    path = file if file.startswith("test/") else f"test/{file}"
    if (
        not _TEST_FILE.fullmatch(path)
        or path.startswith("/")
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise PRTestsError(f"Artifact {artifact_name} contains an invalid test file")

    components = [path]
    components.extend(class_components)
    components.append(name)
    return "::".join(components)


def _report_context(member: str) -> tuple[str | None, str | None]:
    parts = member.replace("\\", "/").split("/")
    try:
        reports_index = parts.index("test-reports")
    except ValueError:
        return None, None
    directory_parts = parts[:reports_index]
    if directory_parts[:1] == ["test"]:
        directory_parts = directory_parts[1:]
    directory = "/".join(directory_parts) or None
    module_index = reports_index + 2
    if module_index >= len(parts) - 1:
        return None, directory
    module = parts[module_index]
    if not module or len(module) > 4096 or not module.isprintable():
        module = None
    return module, directory


def _tests_from_zip(
    data: bytes,
    name: str,
    *,
    reserve_report_bytes: Callable[[int], None] | None = None,
    reserve_reports: Callable[[int], None] | None = None,
    reserve_testcases: Callable[[int], None] | None = None,
) -> tuple[set[str], int]:
    tests: set[str] = set()
    report_bytes = 0
    testcases = 0
    reserved_testcases = 0
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise PRTestsError(f"Artifact {name} is not a valid ZIP file") from error

    try:
        with archive:
            report_members: list[zipfile.ZipInfo] = []
            members = archive.infolist()
            if len(members) > _MAX_ZIP_MEMBERS:
                raise PRTestsError(f"Artifact {name} contains too many files")
            for member in members:
                if not member.filename.endswith(".xml"):
                    continue
                if member.compress_type not in {
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                }:
                    raise PRTestsError(
                        f"Artifact {name} contains an unsupported test report compression"
                    )
                if member.file_size > _MAX_REPORT_BYTES:
                    raise PRTestsError(
                        f"Artifact {name} contains an oversized test report"
                    )
                report_bytes += member.file_size
                if report_bytes > _MAX_REPORT_TOTAL_BYTES:
                    raise PRTestsError(
                        f"Artifact {name} contains too much test report data"
                    )
                report_members.append(member)
                if len(report_members) > _MAX_REPORTS_PER_ARTIFACT:
                    raise PRTestsError(
                        f"Artifact {name} contains too many test reports"
                    )
            if reserve_report_bytes is not None:
                reserve_report_bytes(report_bytes)
            if reserve_reports is not None:
                reserve_reports(len(report_members))

            for member in report_members:
                report_module, report_directory = _report_context(member.filename)
                try:
                    with archive.open(member) as report_file:
                        parser = expat.ParserCreate()
                        depth = 0

                        def start_element(
                            element_name: str, attributes: dict[str, str]
                        ) -> None:
                            nonlocal depth, reserved_testcases, testcases
                            depth += 1
                            if depth > _MAX_XML_DEPTH:
                                raise PRTestsError(
                                    f"Artifact {name} contains an excessively "
                                    "nested test report"
                                )
                            if element_name == "testcase":
                                testcases += 1
                                if testcases > _MAX_TESTCASES_PER_ARTIFACT:
                                    raise PRTestsError(
                                        f"Artifact {name} contains too many test cases"
                                    )
                                if (
                                    reserve_testcases is not None
                                    and testcases - reserved_testcases >= 1000
                                ):
                                    reserve_testcases(testcases - reserved_testcases)
                                    reserved_testcases = testcases
                                qualified_name = _qualified_test_name(
                                    ET.Element("testcase", attributes),
                                    name,
                                    report_module,
                                    report_directory,
                                )
                                if qualified_name is not None:
                                    tests.add(qualified_name)

                        def end_element(element_name: str) -> None:
                            nonlocal depth
                            depth -= 1

                        def reject_declaration(*args: Any) -> None:
                            raise PRTestsError(
                                f"Artifact {name} contains an unsafe XML declaration"
                            )

                        parser.StartElementHandler = start_element
                        parser.EndElementHandler = end_element
                        parser.StartDoctypeDeclHandler = reject_declaration
                        parser.EntityDeclHandler = reject_declaration
                        while chunk := report_file.read(64 * 1024):
                            parser.Parse(chunk, False)
                        parser.Parse(b"", True)
                except expat.ExpatError as error:
                    raise PRTestsError(
                        f"Artifact {name} contains an invalid test report"
                    ) from error
            if reserve_testcases is not None and testcases > reserved_testcases:
                reserve_testcases(testcases - reserved_testcases)
    except PRTestsError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise PRTestsError(f"Artifact {name} is not a valid ZIP file") from error
    return tests, len(report_members)


def _download_artifact(artifact: Artifact, token: str | None) -> bytes:
    if artifact.source == "github" and token is None:
        raise PRTestsError(
            f"Artifact {artifact.name} requires GitHub authentication; set "
            "GITHUB_TOKEN, GH_TOKEN, or authenticate with gh"
        )
    headers = _github_headers(token) if artifact.source == "github" else None
    return _request_bytes(artifact.url, headers=headers, max_bytes=_MAX_ARTIFACT_BYTES)


def _active_runs(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for run in runs:
        status = run.get("status")
        conclusion = run.get("conclusion")
        if status != "completed":
            name = run.get("name", run.get("id", "unknown"))
            raise PRTestsError(f"Workflow {name} has not completed")
        if conclusion in _IGNORED_RUN_CONCLUSIONS:
            continue
        if not isinstance(run.get("id"), int):
            raise PRTestsError("GitHub returned a workflow with an invalid ID")
        active.append(run)
    if not active:
        raise PRTestsError("The PR head has no completed GitHub Actions runs")
    return active


def _logical_test_job_name(name: str) -> str:
    match = _TEST_JOB_NAME.search(name)
    if match is None:
        return name
    matrix = match.group("matrix").split(", ")
    if len(matrix) < 4 or not matrix[3].startswith(("lf-", "mt-")):
        return name
    # The same job can move between Meta and LF pools across reruns.
    matrix[3] = matrix[3][3:]
    return f"{name[: match.start('matrix')]}{', '.join(matrix)})"


def _workflow_job_keys(
    run_id: int, jobs: Iterable[dict[str, Any]]
) -> tuple[dict[int, JobIdentity], dict[tuple[int, str], ExpectedJob]]:
    job_keys: dict[int, JobIdentity] = {}
    expected: dict[tuple[int, str], ExpectedJob] = {}
    seen_attempts: set[tuple[int, str]] = set()
    latest_attempts: dict[tuple[int, str], int] = {}
    for job in jobs:
        name = job.get("name")
        job_id = job.get("id")
        if not isinstance(name, str) or not isinstance(job_id, int):
            raise PRTestsError("GitHub returned a workflow job with invalid metadata")
        key = (run_id, name)
        job_keys[job_id] = JobIdentity(key=key, run_attempt=None)
        if not _TEST_JOB_NAME.search(name):
            continue

        key = (run_id, _logical_test_job_name(name))
        run_attempt = job.get("run_attempt")
        if not isinstance(run_attempt, int) or run_attempt < 1:
            raise PRTestsError(f"Test job {name} has an invalid run attempt")
        job_keys[job_id] = JobIdentity(key=key, run_attempt=run_attempt)
        attempt_key = (run_attempt, key[1])
        if attempt_key in seen_attempts:
            raise PRTestsError(
                f"Workflow run {run_id} contains duplicate test jobs named {name}"
            )
        seen_attempts.add(attempt_key)
        latest_attempts[key] = max(latest_attempts.get(key, 0), run_attempt)
        if job.get("status") != "completed":
            raise PRTestsError(f"Test job {name} has not completed")
        if job.get("conclusion") in _IGNORED_JOB_CONCLUSIONS:
            continue
        previous = expected.get(key)
        if previous is None or run_attempt > previous.run_attempt:
            started_at = _parse_timestamp(job.get("started_at"), "GitHub")
            completed_at = _parse_timestamp(job.get("completed_at"), "GitHub")
            if started_at is None or completed_at is None:
                raise PRTestsError(f"Test job {name} is missing timing metadata")
            expected[key] = ExpectedJob(
                job_id=job_id,
                run_attempt=run_attempt,
                latest_run_attempt=run_attempt,
                started_at=started_at,
                completed_at=completed_at,
            )
    expected = {
        key: replace(job, latest_run_attempt=latest_attempts[key])
        for key, job in expected.items()
    }
    return job_keys, expected


def _covered_job_keys(
    artifacts: Iterable[Artifact],
    job_keys: dict[int, JobIdentity],
    expected_jobs: dict[tuple[int, str], ExpectedJob],
    *,
    exact: bool = False,
) -> set[tuple[int, str]]:
    covered: set[tuple[int, str]] = set()
    for artifact in artifacts:
        identity = (
            job_keys.get(artifact.job_id) if artifact.job_id is not None else None
        )
        expected = expected_jobs.get(identity.key) if identity is not None else None
        if expected is None:
            continue
        covered_by_artifact = _artifact_covers_job(artifact, identity, expected)
        if exact:
            covered_by_artifact = (
                artifact.job_id == expected.job_id
                and expected.run_attempt == expected.latest_run_attempt
            )
        if covered_by_artifact:
            covered.add(identity.key)
    return covered


def _artifact_covers_job(
    artifact: Artifact, identity: JobIdentity, expected: ExpectedJob
) -> bool:
    if (
        artifact.job_id == expected.job_id
        and expected.run_attempt == expected.latest_run_attempt
    ):
        return True

    # Reruns can upload under a prior job ID or overwrite its S3 object, so use
    # the write time whenever the job ID alone cannot identify the attempt.
    in_time_window = artifact.updated_at is not None and (
        artifact.updated_at >= expected.started_at
        and artifact.updated_at <= expected.completed_at + _ARTIFACT_UPLOAD_GRACE
    )
    if artifact.job_id == expected.job_id:
        return in_time_window
    return (
        identity.run_attempt is not None
        and identity.run_attempt < expected.run_attempt
        and in_time_window
    )


def _select_artifacts(
    artifacts: Iterable[Artifact],
    job_keys: dict[int, JobIdentity],
    expected_jobs: dict[tuple[int, str], ExpectedJob],
    *,
    github_authenticated: bool = True,
) -> list[Artifact]:
    selected: dict[
        tuple[int, str], tuple[tuple[bool, bool, bool, float], Artifact]
    ] = {}
    extras: list[Artifact] = []
    for artifact in artifacts:
        identity = (
            job_keys.get(artifact.job_id) if artifact.job_id is not None else None
        )
        expected = expected_jobs.get(identity.key) if identity is not None else None
        if expected is None:
            extras.append(artifact)
            continue
        if not _artifact_covers_job(artifact, identity, expected):
            continue
        preferred_source = (
            "s3" if expected.run_attempt == expected.latest_run_attempt else "github"
        )
        score = (
            artifact.source != "github" or github_authenticated,
            artifact.job_id == expected.job_id,
            artifact.source == preferred_source,
            artifact.updated_at.timestamp() if artifact.updated_at is not None else 0.0,
        )
        previous = selected.get(identity.key)
        if previous is None or score > previous[0]:
            selected[identity.key] = score, artifact
    return [*extras, *(artifact for _, artifact in selected.values())]


def collect_pr_tests(value: str, token: str | None = None) -> CollectionResult:
    repository, number = parse_pr(value)
    if token is None:
        token = _github_token()
    repository, head_sha = _pr_info(repository, number, token)
    runs = _active_runs(_workflow_runs(repository, head_sha, token))

    expected_jobs: dict[tuple[int, str], ExpectedJob] = {}
    workflow_jobs: dict[int, JobIdentity] = {}
    s3_artifacts: list[Artifact] = []
    github_artifacts: list[Artifact] = []
    for run in runs:
        run_id = run["id"]
        jobs = _workflow_jobs(repository, run_id, token)
        run_job_keys, run_jobs = _workflow_job_keys(run_id, jobs)
        workflow_jobs.update(run_job_keys)
        expected_jobs.update(run_jobs)
        run_s3_artifacts = _s3_artifacts(repository, run_id)
        s3_artifacts.extend(run_s3_artifacts)
        s3_jobs = _covered_job_keys(
            run_s3_artifacts, run_job_keys, run_jobs, exact=True
        )
        missing_s3_jobs = run_jobs.keys() - s3_jobs
        if missing_s3_jobs:
            run_github_artifacts = _github_artifacts(repository, run_id, token)
            github_artifacts.extend(
                artifact
                for artifact in run_github_artifacts
                if artifact.job_id in run_job_keys
                and run_job_keys[artifact.job_id].key in missing_s3_jobs
            )

    artifacts, expired = _deduplicate_artifacts(s3_artifacts, github_artifacts)
    if len(artifacts) > _MAX_ARTIFACTS:
        raise PRTestsError(
            f"Found more than {_MAX_ARTIFACTS} test report artifacts; "
            "refusing to download an unbounded result"
        )
    unassociated = [
        artifact for artifact in artifacts if artifact.job_id not in workflow_jobs
    ]
    if unassociated:
        raise PRTestsError(
            f"Unable to associate {len(unassociated)} test report artifacts "
            "with workflow jobs"
        )
    available_jobs = _covered_job_keys(artifacts, workflow_jobs, expected_jobs)
    missing_jobs = expected_jobs.keys() - available_jobs
    if missing_jobs:
        missing_count = len(missing_jobs)
        expired_jobs = _covered_job_keys(expired, workflow_jobs, expected_jobs)
        expired_count = len(missing_jobs & expired_jobs)
        detail = f" ({expired_count} expired)" if expired_count else ""
        job_label = "job" if missing_count == 1 else "jobs"
        raise PRTestsError(
            f"Missing test report artifacts for {missing_count} completed "
            f"test {job_label}{detail}; CI artifacts may have expired"
        )
    artifacts = _select_artifacts(
        artifacts,
        workflow_jobs,
        expected_jobs,
        github_authenticated=token is not None,
    )

    tests: set[str] = set()
    reports = 0
    artifact_budget = _Budget(
        _MAX_ARTIFACT_TOTAL_BYTES,
        "Test report artifacts exceeded the aggregate download limit",
    )
    report_budget = _Budget(
        _MAX_REPORT_COLLECTION_BYTES,
        "Test reports exceeded the aggregate expanded-data limit",
    )
    report_count_budget = _Budget(
        _MAX_REPORTS, "Test artifacts contained too many test reports"
    )
    testcase_budget = _Budget(
        _MAX_TESTCASES, "Test reports contained too many test cases"
    )

    def download_and_parse(artifact: Artifact) -> tuple[set[str], int]:
        data = _download_artifact(artifact, token)
        artifact_budget.reserve(len(data))
        return _tests_from_zip(
            data,
            artifact.name,
            reserve_report_bytes=report_budget.reserve,
            reserve_reports=report_count_budget.reserve,
            reserve_testcases=testcase_budget.reserve,
        )

    if artifacts:
        workers = min(_MAX_DOWNLOAD_WORKERS, len(artifacts))
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [
                executor.submit(download_and_parse, artifact) for artifact in artifacts
            ]
            for future in concurrent.futures.as_completed(futures):
                artifact_tests, artifact_reports = future.result()
                tests.update(artifact_tests)
                reports += artifact_reports
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    if expected_jobs and reports == 0:
        raise PRTestsError(
            "Test artifacts contained no JUnit XML reports; "
            "the result may be incomplete"
        )

    return CollectionResult(
        tests=frozenset(tests),
        workflow_runs=len(runs),
        artifacts=len(artifacts),
        reports=reports,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List fully qualified Python tests reported by CI for a PR."
    )
    parser.add_argument(
        "pr",
        help="PyTorch PR number or GitHub PR URL",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    try:
        result = collect_pr_tests(options.pr)
    except PRTestsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for test in sorted(result.tests):
        print(test)
    print(
        f"Found {len(result.tests)} tests in {result.reports} reports "
        f"from {result.artifacts} artifacts across {result.workflow_runs} workflow runs.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
