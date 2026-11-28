#!/usr/bin/env python3
"""
Set up a local ROCm CI debugging environment: clone PyTorch at a commit, parse ROCm workflow
YAML, resolve GHCR image refs, download GitHub Actions build artifacts, ``docker pull`` the CI
image, then ``docker run`` + ``docker exec`` to install the wheel inside the container.

The container is named ``<user>_pytorch_rocm_ci_debug_env_<short-sha>_<workflow>``, where
``<user>`` is the launching user's login name (``getpass.getuser()``), ``<short-sha>`` is the
first 12 chars of the resolved head sha, and ``<workflow>`` is the workflow stem. Because the
name is deterministic, re-running for the same commit+workflow fails fast if that container
still exists (remove it with ``docker rm -f <name>``).

Forking: change ``REPO_OWNER`` / ``REPO_NAME`` below (clone URL, GitHub API, S3 paths follow).

Repo root is ``/var/lib/jenkins/pytorch`` in every mode (wheel at ``dist/``, tests at ``test/``).
This is a subdir of the image's jenkins HOME (``/var/lib/jenkins``), so the image workspace (pip/
sccache caches, dotfiles, etc.) is never shadowed. A ``/workspace/pytorch`` symlink pointing at the
repo root is also created inside the container for convenience.

Acquisition modes (how the PyTorch checkout reaches the container; pick at most one):
* Default (no flag): do NO host clone and NO bind mount. The workflow metadata that the
  host-mapped modes derive from a local checkout is instead fetched from GitHub for the commit
  (workflow YAML, ``.ci/docker`` tree sha, and full head sha) via the contents/commits APIs. The
  image is pulled/detected as usual, then the container is started without a bind mount and ALL
  remaining work happens inside it via ``docker exec``: ``git clone`` PyTorch to
  ``/var/lib/jenkins/pytorch`` and ``git checkout --detach`` the commit, ``curl`` the public S3
  ``artifacts.zip``, unzip, ``pip install`` the wheel, and end at the test dir.
* ``--host-mapped``: clone PyTorch to a host temp directory outside this repo and bind-mount that
  directory to ``/var/lib/jenkins/pytorch`` in the debug container; resolve metadata from the local
  checkout, stage artifacts on the host, and install the wheel via ``docker exec``.
* ``--repo-dir DIR``: host-mapped mode reusing an existing host clone (fetch/checkout the commit),
  bind-mounted to ``/var/lib/jenkins/pytorch``.

Dry run (inspect only):
* ``--workflow-parse-only``: applies on top of any acquisition mode. It resolves the
  ``(commit, workflow)`` to the ROCm build job label, docker-image stem, and ``.ci/docker`` tree
  sha, prints them, and exits — WITHOUT downloading CI artifacts, pulling or ``docker run``-ing the
  image, or installing the wheel. The default mode resolves via the GitHub API (no clone), while
  ``--host-mapped``/``--repo-dir`` resolve from a local checkout (and additionally print
  ``clone-path``).

  Dry-run examples (no artifacts/image/container; see the GH_TOKEN note under Examples):

    # Default mode: resolve via the GitHub API, no clone.
    python3 scripts/pytorch_rocm_ci_debug_env.py <commit> trunk --workflow-parse-only

    # Via a host temp clone (also prints clone-path).
    python3 scripts/pytorch_rocm_ci_debug_env.py <commit> trunk --host-mapped --workflow-parse-only

    # Via an existing host clone (also prints clone-path).
    python3 scripts/pytorch_rocm_ci_debug_env.py <commit> trunk \
        --repo-dir /path/to/pytorch --workflow-parse-only

Auth: optional ``GITHUB_TOKEN`` or ``GH_TOKEN`` (required in practice for the GitHub API calls
used by the default in-container mode).

Requires: ``pip install pyyaml`` (or toolkit ``requirements.txt``).

Examples — real runs that build the debug container and install the wheel (a ``GITHUB_TOKEN``/
``GH_TOKEN`` is recommended, and required in practice for the default in-container mode's GitHub API
calls; e.g. ``export GH_TOKEN=$(gh auth token)``). For inspect-only invocations see the Dry-run
examples above.

    # Default: clone + install inside the container (no host clone, no bind mount).
    python3 scripts/pytorch_rocm_ci_debug_env.py <commit> trunk

    # Host-mapped: clone to a host temp dir and bind-mount it into the container.
    python3 scripts/pytorch_rocm_ci_debug_env.py <commit> trunk --host-mapped

    # Host-mapped reusing an existing clone (fetch/checkout the commit, then bind-mount it).
    python3 scripts/pytorch_rocm_ci_debug_env.py <commit> trunk --repo-dir /path/to/pytorch
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from cli_common import CliError, cli_fail


# ---------------------------------------------------------------------------
# Repository identity (edit for a fork)
# ---------------------------------------------------------------------------

REPO_OWNER = "pytorch"
REPO_NAME = "pytorch"
PYTORCH_CLONE_URL = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------


def git_stdout(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def clone_shallow_then_checkout(repo_url: str, dest: Path, commit: str) -> None:
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", repo_url, str(dest)],
        check=True,
    )
    shallow = subprocess.run(
        ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit],
        capture_output=True,
        text=True,
    )
    if shallow.returncode != 0:
        full = subprocess.run(
            ["git", "-C", str(dest), "fetch", "origin", commit],
            capture_output=True,
            text=True,
        )
        if full.returncode != 0:
            raise RuntimeError(
                f"git fetch failed for {commit!r}:\n{shallow.stderr}\n{full.stderr}"
            )
    subprocess.run(["git", "-C", str(dest), "checkout", "--detach", commit], check=True)


def verify_detached_head_matches(repo: Path, commit: str) -> None:
    head = git_stdout(repo, "rev-parse", "HEAD")
    target = git_stdout(repo, "rev-parse", commit)
    if head != target:
        raise RuntimeError(f"HEAD {head!r} does not match expected {target!r}")


def branch_label_for_container_env(repo: Path) -> str:
    """Value for docker ``BRANCH=``; detached HEAD uses env ``BRANCH`` or ``master``."""
    try:
        abbrev = git_stdout(repo, "rev-parse", "--abbrev-ref", "HEAD")
    except subprocess.CalledProcessError:
        return os.environ.get("BRANCH", "master")
    if abbrev == "HEAD":
        return os.environ.get("BRANCH", "master")
    return abbrev


def checkout_user_repo(repo: Path, commit: str) -> None:
    fetch = subprocess.run(
        ["git", "-C", str(repo), "fetch", "--quiet", "origin", commit],
        capture_output=True,
        text=True,
    )
    if fetch.returncode != 0:
        raise RuntimeError(f"git fetch failed for {commit!r}:\n{fetch.stderr}")
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--detach", "--quiet", commit],
        check=True,
    )


def _safe_name_segment(value: str, *, fallback: str) -> str:
    """One segment restricted to ``[A-Za-z0-9._-]`` (unsafe chars -> ``_``).

    This character set is safe for both a single filesystem path segment and a Docker
    container-name segment, so path and container-name helpers share it.
    """
    cleaned = "".join(c if (c.isalnum() or c in "._-") else "_" for c in value.strip())
    return cleaned or fallback


def filesystem_safe_commit_segment(ref: str) -> str:
    """Commit-ish string safe for a single path segment (no slashes)."""
    return _safe_name_segment(ref, fallback="unknown")


def acquire_repository(
    commit: str,
    repo_dir: Path | None,
) -> tuple[Path, Path | None]:
    """
    Return ``(repo_root, managed_clone_dir_or_none)`` for the host-clone modes.
    ``managed_clone_dir_or_none`` is set when this script created the clone (temp dir prefix
    ``pytorch-ghcr-``); that directory is kept on disk after exit. The default in-container mode
    does not use this function (it performs no host clone).
    """
    if repo_dir is not None:
        root = repo_dir.resolve()
        if not root.is_dir():
            raise CliError(f"--repo-dir is not a directory: {root}")
        checkout_user_repo(root, commit)
        return root, None

    tmp = Path(tempfile.mkdtemp(prefix="pytorch-ghcr-"))
    try:
        clone_shallow_then_checkout(PYTORCH_CLONE_URL, tmp, commit)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        raise CliError(f"clone/checkout failed: {exc}") from exc
    return tmp, tmp


# ---------------------------------------------------------------------------
# Workflow YAML (docker-image-name + build job labels)
# ---------------------------------------------------------------------------


def is_static_github_actions_string(value: str) -> bool:
    """False for empty strings and ``${{ }}`` expressions we cannot evaluate locally."""
    s = value.strip()
    if not s:
        return False
    return not (s.startswith("${{") and s.endswith("}}"))


def _job_text_has_rocm_not_cuda(*parts: str) -> bool:
    combined = " ".join(p.lower() for p in parts if p)
    return "rocm" in combined and "cuda" not in combined


def find_rocm_build_jobs(workflow: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Jobs whose id and display name match ROCm build (``rocm``, not ``cuda``)."""
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return []
    matched: list[tuple[str, dict[str, Any]]] = []
    for job_id, job_def in jobs.items():
        if not isinstance(job_def, dict):
            continue
        jid = str(job_id).strip()
        if not jid or "build" not in jid.lower():
            continue
        raw_name = job_def.get("name")
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        if _job_text_has_rocm_not_cuda(jid, name):
            matched.append((jid, job_def))
    return matched


def rocm_build_job_label(job_id: str, job_def: dict[str, Any]) -> str:
    """Artifact / Actions label: static ``job.name`` if set, else the YAML job key."""
    raw_name = job_def.get("name")
    if isinstance(raw_name, str):
        n = raw_name.strip()
        if n and is_static_github_actions_string(n):
            return n
    return job_id.strip()


def _require_single_rocm_build_job(
    workflow: dict[str, Any], *, workflow_path: Path
) -> tuple[str, dict[str, Any]]:
    matched = find_rocm_build_jobs(workflow)
    if not matched:
        raise CliError(
            f"no ROCm build job in {workflow_path} "
            "(job id/name must contain 'rocm', not 'cuda', and id must contain 'build')."
        )
    if len(matched) > 1:
        ids = [job_id for job_id, _ in matched]
        raise CliError(
            f"expected exactly one ROCm build job in {workflow_path}, found {len(matched)}: {ids}"
        )
    return matched[0]


def collect_build_job_label(workflow: dict[str, Any], *, workflow_path: Path) -> str:
    """Exactly one ROCm build job; excludes CUDA and other build jobs."""
    job_id, job_def = _require_single_rocm_build_job(workflow, workflow_path=workflow_path)
    return rocm_build_job_label(job_id, job_def)


def collect_docker_image_stem(workflow: dict[str, Any], *, workflow_path: Path) -> str:
    """``docker-image-name`` from the single ROCm build job ``with:`` block."""
    _, job_def = _require_single_rocm_build_job(workflow, workflow_path=workflow_path)
    with_block = job_def.get("with")
    if not isinstance(with_block, dict):
        raise CliError(f"ROCm build job has no 'with:' mapping in {workflow_path}.")
    raw = with_block.get("docker-image-name")
    if not isinstance(raw, str):
        raise CliError(f"ROCm build job missing docker-image-name in {workflow_path}.")
    if not is_static_github_actions_string(raw):
        raise CliError(
            f"ROCm build job docker-image-name is not a static string in {workflow_path}: {raw!r}"
        )
    return raw.strip()


def load_workflow_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"workflow root must be a mapping: {path}")
    return data


def workflow_file_path(repo: Path, workflow_stem: str) -> Path:
    return repo / ".github" / "workflows" / f"{workflow_stem}.yml"


@dataclass(frozen=True)
class WorkflowResolution:
    """Checkout-derived workflow metadata used for artifacts, pulls, and ``docker run``."""

    workflow_path: Path
    docker_directory_sha: str  # ``git rev-parse HEAD:.ci/docker``
    image_stem: str
    build_label: str
    head_sha: str


def resolve_workflow(repo: Path, commit: str, workflow_stem: str) -> WorkflowResolution:
    verify_detached_head_matches(repo, commit)

    path = workflow_file_path(repo, workflow_stem)
    if not path.is_file():
        raise CliError(f"workflow file not found: {path}")

    try:
        workflow = load_workflow_yaml(path)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise CliError(f"failed to read workflow YAML: {exc}") from exc

    build_label = collect_build_job_label(workflow, workflow_path=path)
    image_stem = collect_docker_image_stem(workflow, workflow_path=path)

    try:
        docker_directory_sha = git_stdout(repo, "rev-parse", "HEAD:.ci/docker")
    except subprocess.CalledProcessError as exc:
        raise CliError(f"git rev-parse HEAD:.ci/docker failed:\n{exc.stderr}") from exc

    try:
        head_sha = git_stdout(repo, "rev-parse", commit)
    except subprocess.CalledProcessError as exc:
        raise CliError(f"git rev-parse failed for {commit!r}:\n{exc.stderr}") from exc

    return WorkflowResolution(
        workflow_path=path,
        docker_directory_sha=docker_directory_sha,
        image_stem=image_stem,
        build_label=build_label,
        head_sha=head_sha,
    )


def resolve_workflow_via_github(
    commit: str, workflow_stem: str, token: str | None
) -> WorkflowResolution:
    """Resolve workflow metadata for the default in-container mode with no host checkout.

    Reuses the same workflow-parsing functions as ``resolve_workflow`` on a dict parsed from the
    raw workflow YAML fetched from GitHub, and derives the ``.ci/docker`` tree sha and full head
    sha from the contents/commits APIs. ``workflow_path`` is a non-filesystem stand-in used only
    for ``.name`` (the run lookup) and error messages.
    """
    workflow_repo_path = f".github/workflows/{workflow_stem}.yml"
    path = Path(workflow_repo_path)

    text = github_file_text(workflow_repo_path, commit, token)
    try:
        workflow = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CliError(f"failed to parse workflow YAML from GitHub: {exc}") from exc
    if not isinstance(workflow, dict):
        raise CliError(f"workflow root must be a mapping: {workflow_repo_path}")

    build_label = collect_build_job_label(workflow, workflow_path=path)
    image_stem = collect_docker_image_stem(workflow, workflow_path=path)
    docker_directory_sha = github_tree_entry_sha(".ci", "docker", commit, token)
    head_sha = github_commit_full_sha(commit, token)

    return WorkflowResolution(
        workflow_path=path,
        docker_directory_sha=docker_directory_sha,
        image_stem=image_stem,
        build_label=build_label,
        head_sha=head_sha,
    )


# ---------------------------------------------------------------------------
# GitHub Actions API
# ---------------------------------------------------------------------------


def github_api_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def github_get_json(url: str, token: str | None) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "pytorch-rocm-ci-debug-env",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def github_contents_url(path: str, commit: str) -> str:
    encoded_path = urllib.parse.quote(path, safe="/")
    query = urllib.parse.urlencode({"ref": commit})
    return f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{encoded_path}?{query}"


def github_get_json_or_fail(url: str, token: str | None, *, what: str) -> Any:
    """Wrap ``github_get_json`` and convert network/HTTP errors into ``CliError``."""
    try:
        return github_get_json(url, token)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise CliError(f"GitHub API HTTP {exc.code} fetching {what}: {body.strip()}") from exc
    except urllib.error.URLError as exc:
        raise CliError(f"GitHub API request failed fetching {what}: {exc}") from exc


def github_file_text(path: str, commit: str, token: str | None) -> str:
    """Decoded text of a repo file at ``commit`` via the contents API (base64 ``content``)."""
    payload = github_get_json_or_fail(github_contents_url(path, commit), token, what=path)
    if not isinstance(payload, dict):
        raise CliError(f"unexpected contents response for {path} (not an object)")
    content = payload.get("content")
    encoding = payload.get("encoding")
    if encoding != "base64" or not isinstance(content, str):
        raise CliError(f"contents response for {path} is not base64-encoded")
    return base64.b64decode(content).decode("utf-8")


def github_tree_entry_sha(parent_path: str, name: str, commit: str, token: str | None) -> str:
    """Git tree sha of entry ``name`` under directory ``parent_path`` at ``commit``.

    For a directory entry this equals ``git rev-parse <commit>:<parent_path>/<name>``.
    """
    payload = github_get_json_or_fail(
        github_contents_url(parent_path, commit), token, what=parent_path
    )
    if not isinstance(payload, list):
        raise CliError(f"contents response for {parent_path} is not a directory listing")
    for entry in payload:
        if isinstance(entry, dict) and entry.get("name") == name:
            sha = entry.get("sha")
            if isinstance(sha, str) and sha:
                return sha
            raise CliError(f"entry {name!r} under {parent_path} has no sha")
    raise CliError(f"no entry named {name!r} under {parent_path} at {commit}")


def github_commit_full_sha(commit: str, token: str | None) -> str:
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/{commit}"
    payload = github_get_json_or_fail(url, token, what=f"commit {commit}")
    if not isinstance(payload, dict):
        raise CliError(f"unexpected commit response for {commit} (not an object)")
    sha = payload.get("sha")
    if not isinstance(sha, str) or not sha:
        raise CliError(f"commit response for {commit} missing sha")
    return sha


def parse_github_datetime(raw: str) -> datetime:
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw)


def latest_workflow_run_id(workflow_filename: str, head_sha: str, token: str | None) -> int:
    encoded_wf = urllib.parse.quote(workflow_filename, safe="")
    query = urllib.parse.urlencode({"head_sha": head_sha, "per_page": "100"})
    url = (
        f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/"
        f"actions/workflows/{encoded_wf}/runs?{query}"
    )
    payload = github_get_json_or_fail(url, token, what=f"runs for {workflow_filename}")

    runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list) or not runs:
        raise CliError("GitHub API returned no workflow_runs for this workflow and head_sha")

    def sort_key(run: Any) -> datetime:
        created = run.get("created_at") if isinstance(run, dict) else None
        if not isinstance(created, str):
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            return parse_github_datetime(created)
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)

    newest = max(runs, key=sort_key)
    if not isinstance(newest, dict):
        raise CliError("GitHub API workflow run entry is not an object")
    run_id = newest.get("id")
    if not isinstance(run_id, int):
        raise CliError("GitHub API workflow run entry missing integer id")
    return run_id


# ---------------------------------------------------------------------------
# Artifact downloads (gha-artifacts S3)
# ---------------------------------------------------------------------------

_FS_UNSAFE = str.maketrans({c: "_" for c in '<>:"/\\|?*\x00'})


def filesystem_safe_label(label: str) -> str:
    return label.translate(_FS_UNSAFE).strip() or "job"


def artifact_zip_url(workflow_run_id: int, build_job_label: str) -> str:
    o = urllib.parse.quote(REPO_OWNER, safe="")
    r = urllib.parse.quote(REPO_NAME, safe="")
    segment = urllib.parse.quote(build_job_label, safe="")
    return (
        f"https://gha-artifacts.s3.amazonaws.com/{o}/{r}/"
        f"{workflow_run_id}/{segment}/artifacts.zip"
    )


def format_byte_size(num_bytes: int | float) -> str:
    value = float(max(num_bytes, 0))
    for scale, suffix in ((1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")):
        if value >= scale:
            return f"{value / scale:.2f} {suffix}"
    return f"{int(num_bytes)} B"


def download_with_progress(url: str, destination: Path, *, stderr_prefix: str = "") -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "pytorch-rocm-ci-debug-env"})
    chunk_bytes = 256 * 1024
    prefix = f"{stderr_prefix} " if stderr_prefix else ""

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            status = getattr(response, "status", None) or response.getcode()
            if status != 200:
                raise RuntimeError(f"unexpected HTTP status {status}")

            length_header = response.headers.get("Content-Length")
            try:
                total_bytes = int(length_header) if length_header else None
            except (TypeError, ValueError):
                total_bytes = None

            downloaded = 0
            with destination.open("wb") as outfile:
                while True:
                    chunk = response.read(chunk_bytes)
                    if not chunk:
                        break
                    outfile.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes and total_bytes > 0:
                        pct = 100.0 * downloaded / total_bytes
                        line = (
                            f"{prefix}artifacts.zip {format_byte_size(downloaded)} / "
                            f"{format_byte_size(total_bytes)} ({pct:.1f}%)"
                        )
                    else:
                        line = f"{prefix}artifacts.zip {format_byte_size(downloaded)}"
                    sys.stderr.write(f"\r{line}\x1b[K")
                    sys.stderr.flush()

            sys.stderr.write("\r\x1b[2K")
            sys.stderr.flush()
            print(file=sys.stderr)

    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} when downloading {url}") from exc


def download_build_artifact(repo: Path, build_label: str, workflow_run_id: int) -> None:
    url = artifact_zip_url(workflow_run_id, build_label)
    dest = repo / "gha-artifacts" / filesystem_safe_label(build_label) / "artifacts.zip"
    print(url)
    download_with_progress(url, dest, stderr_prefix=f"[{build_label}]")
    print(f"artifact-zip: {dest}", file=sys.stderr)


def copy_build_artifact_to_repo_root(repo: Path, build_label: str) -> None:
    src = repo / "gha-artifacts" / filesystem_safe_label(build_label) / "artifacts.zip"
    if not src.is_file():
        raise RuntimeError(f"missing artifact zip for {build_label!r}: {src}")
    shutil.copy2(src, repo / "artifacts.zip")
    print(f"staged-artifacts-zip-from: {build_label}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

DOCKER_RUN_ENV_LITERAL: tuple[tuple[str, str], ...] = (
    ("GITHUB_ACTIONS", "true"),
    ("PYTORCH_RETRY_TEST_CASES", "1"),
    ("PYTORCH_OVERRIDE_FLAKY_SIGNAL", "1"),
    ("SCCACHE_BUCKET", "ossci-compiler-cache-circleci-v2"),
    ("XLA_CLANG_CACHE_S3_BUCKET_NAME", "ossci-compiler-clang-cache-circleci-xla"),
    ("PYTORCH_TEST_CUDA_MEM_LEAK_CHECK", "0"),
    ("PYTORCH_TEST_RERUN_DISABLED_TESTS", "0"),
    ("CI", "True"),
)

DOCKER_RUN_ENV_FROM_HOST = (
    "AWS_DEFAULT_REGION",
    "IN_WHEEL_TEST",
    "PR_BODY",
    "COMMIT_MESSAGES",
)

# Jenkins HOME shipped by the CI image (pip/sccache caches, dotfiles, etc.). It is never used as
# a bind-mount target, so the image's own contents are preserved (not shadowed).
CONTAINER_HOME_DIR = "/var/lib/jenkins"

# Repo root for ALL modes: a child dir of the jenkins HOME. The host-mapped modes bind-mount the
# checkout here; the default in-container mode clones here. Using a subdir (rather than the HOME
# itself) avoids shadowing the image workspace and keeps the chown scope small.
CONTAINER_REPO_DIR = f"{CONTAINER_HOME_DIR}/pytorch"


def container_setup_script(
    work_dir: str,
    *,
    clone_url: str | None = None,
    commit: str | None = None,
    artifact_url: str | None = None,
) -> str:
    """Bash that ends at the test dir: (optional clone + artifact download), unzip, pip install.

    ``work_dir`` is the repo root (``CONTAINER_REPO_DIR`` for every mode). With ``clone_url``/
    ``commit`` it first ``git clone``s + ``checkout --detach``s into ``work_dir`` (used by the
    default in-container mode). With ``artifact_url`` it downloads ``artifacts.zip`` into
    ``work_dir`` (the in-container path; the host-mapped modes stage the zip via the bind mount
    instead). The unzip/``pip install``/``cd test`` tail is shared by all modes, as is a
    ``/workspace/pytorch`` -> ``work_dir`` convenience symlink.
    """
    lines = ["set -e"]
    if clone_url and commit:
        lines += [
            f"rm -rf {work_dir}",
            f"git clone --filter=blob:none --no-checkout {clone_url} {work_dir}",
            f"cd {work_dir}",
            f"git fetch --depth 1 origin {commit} || git fetch origin {commit}",
            f"git checkout --detach {commit}",
        ]
    lines += [
        f"sudo chown -R jenkins:jenkins {work_dir}",
        # Convenience symlink so /workspace/pytorch resolves to the unified repo root.
        "sudo mkdir -p /workspace",
        f"sudo ln -sfn {work_dir} /workspace/pytorch",
        f"cd {work_dir}",
        "if ! command -v unzip >/dev/null 2>&1; then",
        "  export DEBIAN_FRONTEND=noninteractive",
        "  apt-get update -qq && apt-get install -y -qq unzip",
        "fi",
    ]
    if artifact_url:
        lines += [
            f'if command -v curl >/dev/null 2>&1; then curl -fSL -o artifacts.zip "{artifact_url}";',
            f'else wget -O artifacts.zip "{artifact_url}"; fi',
        ]
    lines += [
        "test -f artifacts.zip",
        "unzip -o -q artifacts.zip",
        "cd dist",
        "pip install *.whl",
        f"cd {work_dir}/test",
        "pwd >&2",
    ]
    return "\n".join(lines)


DEBUG_ENV_NAME_INFIX = "pytorch_rocm_ci_debug_env"


def launching_user_id() -> str:
    """Login name of the user running the script, for the container name.

    Prefers ``getpass.getuser()`` (honors ``LOGNAME``/``USER``/``LNAME``/``USERNAME``,
    then the password database for the current uid); falls back to the numeric uid,
    then the literal ``user`` if neither is resolvable.
    """
    try:
        name = getpass.getuser().strip()
    except (OSError, KeyError):
        name = ""
    if name:
        return name
    getuid = getattr(os, "getuid", None)
    if getuid is not None:
        return str(getuid())
    return "user"


def debug_env_container_name(user: str, head_sha: str, workflow: str) -> str:
    """``<user>_pytorch_rocm_ci_debug_env_<short-sha>_<workflow>``, Docker-name safe.

    Each segment is restricted to ``[A-Za-z0-9._-]``; the commit uses the first 12
    chars of the resolved head sha. Docker requires the first char to be alphanumeric,
    so a leading ``x_`` is prepended if the assembled name would not start with one.
    """
    user_seg = _safe_name_segment(user, fallback="user")
    sha_seg = _safe_name_segment(head_sha[:12], fallback="commit")
    workflow_seg = _safe_name_segment(workflow, fallback="workflow")
    name = f"{user_seg}_{DEBUG_ENV_NAME_INFIX}_{sha_seg}_{workflow_seg}"
    if not name[:1].isalnum():
        name = f"x_{name}"
    return name


def docker_container_name_in_use(name: str) -> bool:
    """True if a container (running or stopped) with exactly ``name`` already exists."""
    completed = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name=^{name}$"],
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def host_render_gid() -> str | None:
    """GID of the host ``render`` group (setup-rocm action.yml pattern), if present."""
    try:
        for line in Path("/etc/group").read_text(encoding="utf-8").splitlines():
            if line.startswith("render:"):
                parts = line.split(":")
                if len(parts) >= 3 and parts[2].isdigit():
                    return parts[2]
    except OSError:
        pass
    return None


def docker_run_pytorch_container(
    image_ref: str,
    repo: Path | None,
    head_sha: str,
    branch: str,
    container_name: str,
) -> str:
    """Start a detached container; ``-i`` without ``-t`` so stdin is not required.

    ``repo`` is bind-mounted to ``CONTAINER_REPO_DIR`` for the host-clone modes (docker auto-creates
    the target dir, so the image's jenkins HOME is preserved, not shadowed). When ``repo`` is
    ``None`` (the default in-container mode) no bind mount is added and the clone happens inside the
    container. Either way the repo root is ``CONTAINER_REPO_DIR``.

    ``container_name`` is the required ``docker run --name`` (see ``debug_env_container_name``).
    """
    command: list[str] = ["docker", "run", "-i", "--name", container_name]
    for device in ("/dev/mem", "/dev/kfd", "/dev/dri"):
        command.extend(["--device", device])
    command.extend(
        ["--user", "jenkins", "--group-add", "bin", "--group-add", "video", "--group-add", "daemon"]
    )
    render_gid = host_render_gid()
    if render_gid:
        command.extend(["--group-add", render_gid])
    else:
        print("note: host has no render group; skipping --group-add for render", file=sys.stderr)
    command.extend(["-e", f"BRANCH={branch}", "-e", f"SHA1={head_sha}"])
    for key, val in DOCKER_RUN_ENV_LITERAL:
        command.extend(["-e", f"{key}={val}"])
    for key in DOCKER_RUN_ENV_FROM_HOST:
        command.extend(["-e", key])
    command.extend(
        [
            "--ulimit",
            "stack=10485760:83886080",
            "--security-opt",
            "seccomp=unconfined",
            "--cap-add=SYS_PTRACE",
            "--shm-size=8g",
            "--detach",
            "--network=host",
        ]
    )
    if repo is not None:
        command.extend(["-v", f"{repo.resolve()}:{CONTAINER_REPO_DIR}"])
        work_dir = CONTAINER_REPO_DIR
    else:
        work_dir = CONTAINER_HOME_DIR
    command.extend(["-w", work_dir, image_ref])
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"docker run failed ({completed.returncode}): "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    container_id = (completed.stdout or "").strip()
    if not container_id:
        raise RuntimeError("docker run produced no container id on stdout")
    return container_id


def docker_inspect_container_name(container_id: str) -> str:
    completed = subprocess.run(
        ["docker", "inspect", "-f", "{{.Name}}", container_id],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    return completed.stdout.strip().lstrip("/")


def docker_exec_chown_workspace(container_id: str, target_dir: str) -> None:
    """Root chown of ``target_dir`` so the jenkins user can write to it."""
    completed = subprocess.run(
        [
            "docker",
            "exec",
            "-u",
            "root",
            container_id,
            "chown",
            "-R",
            "jenkins:jenkins",
            target_dir,
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"docker exec -u root chown failed ({completed.returncode}): {detail}"
        )


def _docker_exec_bash(container_id: str, script: str) -> None:
    completed = subprocess.run(["docker", "exec", container_id, "bash", "-lc", script])
    if completed.returncode != 0:
        raise RuntimeError(f"docker exec failed with exit code {completed.returncode}")


def docker_exec_install_wheel(container_id: str) -> None:
    """Host-clone modes: the bind mount at ``CONTAINER_REPO_DIR`` already has ``artifacts.zip``."""
    docker_exec_chown_workspace(container_id, CONTAINER_REPO_DIR)
    _docker_exec_bash(container_id, container_setup_script(CONTAINER_REPO_DIR))


def docker_exec_clone_and_install_wheel(
    container_id: str, commit: str, artifact_url: str
) -> None:
    """Default in-container mode: clone, download artifacts, and install the wheel in-container."""
    # The repo subdir does not exist yet, so chown the jenkins HOME so the clone can create it.
    docker_exec_chown_workspace(container_id, CONTAINER_HOME_DIR)
    script = container_setup_script(
        CONTAINER_REPO_DIR,
        clone_url=PYTORCH_CLONE_URL,
        commit=commit,
        artifact_url=artifact_url,
    )
    _docker_exec_bash(container_id, script)


def docker_pull_with_progress(image_reference: str) -> int:
    """
    Run ``docker pull`` and print JSON progress as layer lines on stderr.

    Docker's plain progress mode lacks per-layer percentages; we disable hints and parse JSON lines.
    """
    process = subprocess.Popen(
        ["docker", "pull", image_reference],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "DOCKER_CLI_HINTS": "false"},
    )

    layers: dict[str, str] = {}
    lines_on_screen = 0

    def clear_layers() -> None:
        nonlocal lines_on_screen
        if lines_on_screen:
            sys.stderr.write(f"\x1b[{lines_on_screen}A\x1b[0J")
            lines_on_screen = 0

    def redraw_layers() -> None:
        nonlocal lines_on_screen
        clear_layers()
        if layers:
            block = "\n".join(f"  {layer_id[:12]}  {status}" for layer_id, status in layers.items())
            sys.stderr.write(block + "\n")
            sys.stderr.flush()
            lines_on_screen = len(layers)

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            clear_layers()
            print(line, file=sys.stderr)
            redraw_layers()
            continue

        error_message = event.get("error")
        if error_message:
            clear_layers()
            print(f"error: {error_message}", file=sys.stderr)
            continue

        status = event.get("status")
        if not status:
            continue

        layer_id = event.get("id")
        detail = event.get("progressDetail") or {}
        current = detail.get("current")
        total = detail.get("total")

        if layer_id:
            if current is not None and total and total > 0:
                pct = 100.0 * current / total
                layers[layer_id] = (
                    f"{status}: {format_byte_size(current)} / "
                    f"{format_byte_size(total)} ({pct:.1f}%)"
                )
            elif current is not None:
                layers[layer_id] = f"{status}: {format_byte_size(current)}"
            else:
                layers[layer_id] = status
            redraw_layers()
        else:
            clear_layers()
            print(status, file=sys.stderr)
            redraw_layers()

    clear_layers()
    process.wait()
    return process.returncode


def ghcr_image_ref(image_stem: str, docker_directory_sha: str) -> str:
    return f"ghcr.io/pytorch/{image_stem}-{docker_directory_sha}"


def docker_image_present_locally(image_reference: str) -> bool:
    completed = subprocess.run(
        ["docker", "image", "inspect", image_reference],
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def pull_workflow_image(image_stem: str, docker_directory_sha: str) -> None:
    reference = ghcr_image_ref(image_stem, docker_directory_sha)
    print(reference)
    if docker_image_present_locally(reference):
        print(f"docker-image-present: {reference}", file=sys.stderr)
        return
    if docker_pull_with_progress(reference) != 0:
        raise RuntimeError(f"docker pull failed for {reference}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone PyTorch at a commit, resolve a ROCm workflow, download CI artifacts, "
            "pull GHCR images, start a container, and install the wheel inside it."
        ),
    )
    parser.add_argument("commit", help="Git commit hash to check out")
    parser.add_argument(
        "workflow",
        help=(
            "Workflow file stem without .yml (e.g. trunk → .github/workflows/trunk.yml); "
            "ROCm build job is selected from jobs in that file."
        ),
    )
    modes = parser.add_argument_group(
        "acquisition modes",
        "How the PyTorch checkout reaches the container (pick at most one; default is in-container).",
    )
    modes.add_argument(
        "--host-mapped",
        action="store_true",
        help=(
            "Host-mapped mode: clone PyTorch to a host temp dir, bind-mount it to "
            "/var/lib/jenkins/pytorch, resolve metadata from the local checkout, stage artifacts "
            "on the host, and install the wheel via docker exec. Prints clone-path. Default "
            "(without this flag or --repo-dir): everything happens inside the container."
        ),
    )
    modes.add_argument(
        "--repo-dir",
        type=Path,
        help=(
            "Host-mapped mode using this existing clone (fetch/checkout the commit, bind-mount it) "
            "instead of creating a temporary one"
        ),
    )
    dry_run = parser.add_argument_group("dry run")
    dry_run.add_argument(
        "--workflow-parse-only",
        action="store_true",
        help=(
            "Dry run: resolve (commit, workflow) to the ROCm build job label, docker-image stem, "
            "and .ci/docker tree sha, print them, and exit. No artifact download, no image pull, "
            "no docker run, no wheel install. Works with any mode: the default resolves via the "
            "GitHub API (no clone); --host-mapped/--repo-dir resolve from a local checkout."
        ),
    )
    return parser.parse_args()


def use_in_container_mode(args: argparse.Namespace) -> bool:
    """True for the default in-container mode (no host clone / no bind mount).

    Host-mapped mode is selected by ``--host-mapped`` (temp clone) or ``--repo-dir`` (existing
    clone); anything else defaults to cloning + installing inside the container.
    """
    return not args.host_mapped and args.repo_dir is None


def resolve_for_mode(
    args: argparse.Namespace, token: str | None
) -> tuple[Path | None, Path | None, WorkflowResolution]:
    """Resolve workflow metadata for the selected mode.

    Returns ``(repo_root, temporary_clone_dir, resolution)``. In the default in-container mode both
    paths are ``None`` (no host clone) and metadata comes from the GitHub API; for the host-mapped
    modes the repo is acquired (temp clone or ``--repo-dir``) and parsed from the local checkout.
    """
    if use_in_container_mode(args):
        return None, None, resolve_workflow_via_github(args.commit, args.workflow, token)
    repo_root, temporary_clone_dir = acquire_repository(args.commit, args.repo_dir)
    return repo_root, temporary_clone_dir, resolve_workflow(repo_root, args.commit, args.workflow)


def main() -> int:
    args = parse_arguments()

    if args.repo_dir is not None and args.host_mapped:
        return cli_fail("--repo-dir and --host-mapped cannot be used together")

    in_container = use_in_container_mode(args)
    token = github_api_token()

    try:
        repo_root, temporary_clone_dir, resolved = resolve_for_mode(args, token)
    except CliError as exc:
        return cli_fail(str(exc))

    print(f"build-job-name: {resolved.build_label}", file=sys.stderr)
    print(f"docker-image-stem: {resolved.image_stem}", file=sys.stderr)
    print(f"docker-directory-sha: {resolved.docker_directory_sha}", file=sys.stderr)

    if args.workflow_parse_only:
        if temporary_clone_dir is not None:
            print(f"clone-path: {temporary_clone_dir}", file=sys.stderr)
        return 0

    try:
        run_id = latest_workflow_run_id(resolved.workflow_path.name, resolved.head_sha, token)
        print(f"workflow-run-id: {run_id}", file=sys.stderr)
        if not in_container:
            download_build_artifact(repo_root, resolved.build_label, run_id)
        pull_workflow_image(resolved.image_stem, resolved.docker_directory_sha)
    except (CliError, RuntimeError) as exc:
        return cli_fail(str(exc))

    primary_image = ghcr_image_ref(resolved.image_stem, resolved.docker_directory_sha)
    print(f"docker-run-image: {primary_image}", file=sys.stderr)

    container_name = debug_env_container_name(
        launching_user_id(), resolved.head_sha, args.workflow
    )
    if docker_container_name_in_use(container_name):
        return cli_fail(
            f"a container named {container_name!r} already exists; remove it with "
            f"`docker rm -f {container_name}` (or rename it) and re-run"
        )

    container_id: str | None = None
    try:
        if in_container:
            container_id = docker_run_pytorch_container(
                primary_image,
                None,
                resolved.head_sha,
                os.environ.get("BRANCH", "master"),
                container_name,
            )
            artifact_url = artifact_zip_url(run_id, resolved.build_label)
            print(artifact_url, file=sys.stderr)
            docker_exec_clone_and_install_wheel(container_id, args.commit, artifact_url)
        else:
            copy_build_artifact_to_repo_root(repo_root, resolved.build_label)
            container_id = docker_run_pytorch_container(
                primary_image,
                repo_root,
                resolved.head_sha,
                branch_label_for_container_env(repo_root),
                container_name,
            )
            docker_exec_install_wheel(container_id)
    except (CliError, RuntimeError) as exc:
        if container_id:
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
        return cli_fail(str(exc))

    print(f"docker-container-id: {container_id}", file=sys.stderr)
    print("docker exec: unzip, pip install wheel, cd test - done.", file=sys.stderr)

    if temporary_clone_dir is not None:
        print(f"clone-path: {temporary_clone_dir}", file=sys.stderr)
    if in_container:
        print(f"container-clone-path: {CONTAINER_REPO_DIR}", file=sys.stderr)

    try:
        print(f"docker-container-name: {docker_inspect_container_name(container_id)}", file=sys.stderr)
    except (CliError, RuntimeError) as exc:
        return cli_fail(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
