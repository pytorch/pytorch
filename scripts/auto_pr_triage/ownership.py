"""Trusted team ownership configuration for Auto PR Triage."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from codepath_owners import OWNER_ID_RE


TARGET_BASE_REF = "main"
EXTRA_OWNERSHIP_METADATA_PATH = (
    ".github/auto-pr-triage/extra_ownership_metadata.json"
)
TEAM_MEMBERS_PATH = ".github/auto-pr-triage/team_members.json"
MAX_CONFIG_BYTES = 1_000_000
SHA_RE = re.compile(r"[0-9a-f]{40}")
USER_HANDLE_PATTERN = r"@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
USER_HANDLE_RE = re.compile(USER_HANDLE_PATTERN)
OWNER_LABEL_PREFIX = "owner: "


def _decode_document(text: str, path: str) -> dict[str, Any]:
    """Decode one strict JSON configuration document."""

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _validate_source(source: Any, repo: str, ref: str, path: str) -> None:
    """Validate one immutable repository-file identity."""

    if not isinstance(source, dict) or set(source) != {
        "repository",
        "path",
        "ref",
        "blob_sha",
    }:
        raise RuntimeError("ownership source is invalid")
    if (
        source["repository"] != repo
        or source["path"] != path
        or source["ref"] != ref
        or not SHA_RE.fullmatch(source["blob_sha"])
    ):
        raise RuntimeError("ownership source does not match the workflow revision")


def owner_label(owner_id: str) -> str:
    """Return the deterministic routing label for one internal owner ID."""

    if not OWNER_ID_RE.fullmatch(owner_id):
        raise ValueError("owner ID is invalid")
    label = f"{OWNER_LABEL_PREFIX}{owner_id}"
    if len(label) > 50:
        raise ValueError("owner label exceeds GitHub's length limit")
    return label


def validate_extra_ownership_metadata(
    metadata: dict[str, Any],
    repo: str,
    ref: str | None = None,
) -> None:
    """Validate the semantic owner descriptions used during analysis."""

    if not isinstance(metadata, dict) or set(metadata) != {"source", "owners"}:
        raise RuntimeError("extra ownership metadata configuration is invalid")
    source = metadata["source"]
    if not isinstance(source, dict):
        raise RuntimeError("extra ownership metadata source is invalid")
    effective_ref = ref or source.get("ref")
    if not isinstance(effective_ref, str) or not SHA_RE.fullmatch(effective_ref):
        raise RuntimeError("extra ownership metadata ref is invalid")
    _validate_source(source, repo, effective_ref, EXTRA_OWNERSHIP_METADATA_PATH)
    owners = metadata["owners"]
    if not isinstance(owners, dict) or not owners:
        raise RuntimeError("extra ownership metadata owners are invalid")
    for owner, description in owners.items():
        if (
            not isinstance(owner, str)
            or not OWNER_ID_RE.fullmatch(owner)
            or not isinstance(description, str)
        ):
            raise RuntimeError("extra ownership metadata entry is invalid")
        if not description.strip() or len(description) > 2_000:
            raise RuntimeError("extra ownership metadata description is invalid")


def validate_team_members(
    members: dict[str, Any],
    repo: str,
    ref: str | None = None,
) -> None:
    """Validate the reviewer-roster configuration used only during apply."""

    if not isinstance(members, dict) or set(members) != {"source", "members"}:
        raise RuntimeError("team members configuration is invalid")
    source = members["source"]
    if not isinstance(source, dict):
        raise RuntimeError("team members source is invalid")
    effective_ref = ref or source.get("ref")
    if not isinstance(effective_ref, str) or not SHA_RE.fullmatch(effective_ref):
        raise RuntimeError("team members ref is invalid")
    _validate_source(members["source"], repo, effective_ref, TEAM_MEMBERS_PATH)

    rosters = members["members"]
    if not isinstance(rosters, dict) or not rosters:
        raise RuntimeError("team member rosters are invalid")

    canonical_members: dict[str, str] = {}
    for owner_id, roster in rosters.items():
        if not isinstance(owner_id, str) or not OWNER_ID_RE.fullmatch(owner_id):
            raise RuntimeError("team member owner ID is invalid")
        try:
            owner_label(owner_id)
        except ValueError as exc:
            raise RuntimeError("team member owner ID is invalid") from exc
        if not isinstance(roster, (list, tuple)) or not roster:
            raise RuntimeError("team reviewer roster is invalid")
        seen_roster: set[str] = set()
        for reviewer in roster:
            if not isinstance(reviewer, str) or not USER_HANDLE_RE.fullmatch(reviewer):
                raise RuntimeError("team reviewer handle is invalid")
            key = reviewer.casefold()
            if key in seen_roster:
                raise RuntimeError("team reviewer roster contains a duplicate")
            seen_roster.add(key)
            prior = canonical_members.get(key)
            if prior is not None and prior != reviewer:
                raise RuntimeError("team reviewer casing is inconsistent")
            canonical_members[key] = reviewer


def parse_extra_ownership_metadata(
    document: str,
    source: dict[str, Any],
    repo: str,
    ref: str,
) -> dict[str, Any]:
    """Parse one semantic-owner metadata document."""

    decoded = _decode_document(document, EXTRA_OWNERSHIP_METADATA_PATH)
    metadata = {
        "source": source,
        "owners": dict(decoded),
    }
    validate_extra_ownership_metadata(metadata, repo, ref)
    return metadata


def parse_team_members(
    document: str,
    source: dict[str, Any],
    repo: str,
    ref: str,
) -> dict[str, Any]:
    """Parse one reviewer-roster document."""

    members_document = _decode_document(document, TEAM_MEMBERS_PATH)
    members = {
        "source": source,
        "members": dict(members_document),
    }
    validate_team_members(members, repo, ref)
    return members


def _load_document(
    repository_root: Path,
    repo: str,
    ref: str,
    path: str,
) -> tuple[str, dict[str, Any]]:
    """Load one policy document from the trusted checkout.

    The caller must provide the repository checkout created from ``ref``. This
    accepted stale-result tradeoff avoids a second GitHub fetch or revision
    check: workflow checkout is the trust boundary, and stale metadata can only
    affect bounded reviewer routing.
    """

    if not SHA_RE.fullmatch(ref):
        raise ValueError("ownership ref must be an immutable commit SHA")

    root = repository_root.resolve()
    file_path = repository_root / path
    resolved_path = file_path.resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"ownership file escapes the checkout: {path}") from exc
    if file_path.is_symlink() or not resolved_path.is_file():
        raise RuntimeError(f"ownership file is unavailable: {path}")
    try:
        with resolved_path.open("rb") as stream:
            content = stream.read(MAX_CONFIG_BYTES + 1)
    except OSError as exc:
        raise RuntimeError(f"ownership file is unavailable: {path}") from exc
    if len(content) > MAX_CONFIG_BYTES:
        raise RuntimeError(f"ownership file size is invalid: {path}")
    try:
        text = content.decode()
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"ownership file content is invalid: {path}") from exc
    header = f"blob {len(content)}\0".encode()
    return text, {
        "repository": repo,
        "path": path,
        "ref": ref,
        "blob_sha": hashlib.sha1(header + content).hexdigest(),
    }


def load_extra_ownership_metadata(
    repository_root: Path,
    repo: str,
    ref: str,
) -> dict[str, Any]:
    """Load semantic owner descriptions from the trusted checkout."""

    document, source = _load_document(
        repository_root, repo, ref, EXTRA_OWNERSHIP_METADATA_PATH
    )
    return parse_extra_ownership_metadata(
        document,
        source,
        repo,
        ref,
    )


def load_team_members(
    repository_root: Path,
    repo: str,
    ref: str,
) -> dict[str, Any]:
    """Load reviewer rosters from the trusted checkout."""

    document, source = _load_document(
        repository_root, repo, ref, TEAM_MEMBERS_PATH
    )
    return parse_team_members(
        document,
        source,
        repo,
        ref,
    )


def all_team_members(members: dict[str, Any]) -> set[str]:
    """Return every configured reviewer handle."""

    return {
        member
        for roster in members["members"].values()
        for member in roster
    }
