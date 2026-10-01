"""Load the checked-in Auto PR Triage configuration from the trusted checkout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from codepath_owners import parse_rules
from identifiers import owner_label, TEAM_OWNER_ID_RE, USER_HANDLE_RE


CODEPATH_OWNERS_PATH = ".github/auto-pr-triage/codepath_owners.txt"
EXTRA_OWNERSHIP_METADATA_PATH = ".github/auto-pr-triage/extra_ownership_metadata.json"
TEAM_MEMBERS_PATH = ".github/auto-pr-triage/team_members.json"
MAX_CODEPATH_OWNERS_BYTES = 3_000_000
MAX_CONFIG_BYTES = 1_000_000


def _load_document(
    *,
    repository_root: Path,
    repo: str,
    ref: str,
    path: str,
    max_bytes: int,
) -> tuple[str, dict[str, Any]]:
    """Load one configuration file from the trusted checkout.

    The caller must provide the repository checkout created from ``ref``. This
    accepted stale-result tradeoff avoids a second GitHub fetch or revision
    check: workflow checkout is the trust boundary, and stale configuration can
    only affect bounded reviewer routing.
    """

    root = repository_root.resolve()
    file_path = repository_root / path
    resolved_path = file_path.resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"config file escapes the checkout: {path}") from exc
    if file_path.is_symlink() or not resolved_path.is_file():
        raise RuntimeError(f"config file is unavailable: {path}")
    try:
        with resolved_path.open("rb") as stream:
            content = stream.read(max_bytes + 1)
    except OSError as exc:
        raise RuntimeError(f"config file is unavailable: {path}") from exc
    if len(content) > max_bytes:
        raise RuntimeError(f"config file size is invalid: {path}")
    try:
        text = content.decode()
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"config file is not valid UTF-8: {path}") from exc
    header = f"blob {len(content)}\0".encode()
    return text, {
        "repository": repo,
        "path": path,
        "ref": ref,
        "blob_sha": hashlib.sha1(header + content).hexdigest(),
    }


def _decode_document(*, text: str, path: str) -> dict[str, Any]:
    """Decode one strict JSON configuration document."""

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def load_codepath_owners(
    *, repository_root: Path, repo: str, ref: str
) -> dict[str, Any]:
    """Load and strictly parse the codepath-owner rules."""

    text, source = _load_document(
        repository_root=repository_root,
        repo=repo,
        ref=ref,
        path=CODEPATH_OWNERS_PATH,
        max_bytes=MAX_CODEPATH_OWNERS_BYTES,
    )
    diagnostics: list[dict[str, Any]] = []
    rules = parse_rules(
        contents=text, blob_sha=source["blob_sha"], diagnostics=diagnostics, strict=True
    )
    target_org = repo.split("/", 1)[0].casefold()
    for rule in rules:
        for owner in rule["owners"]:
            if (
                owner.startswith("@")
                and "/" in owner
                and owner[1:].split("/", 1)[0].casefold() != target_org
            ):
                raise RuntimeError(
                    "codepath-owner policy names a team outside the target organization"
                )
    return {"source": source, "rules": rules, "parse_diagnostics": diagnostics}


def load_extra_ownership_metadata(
    *, repository_root: Path, repo: str, ref: str
) -> dict[str, Any]:
    """Load owner descriptions and optional bypass-intake criteria for the LLM."""

    text, source = _load_document(
        repository_root=repository_root,
        repo=repo,
        ref=ref,
        path=EXTRA_OWNERSHIP_METADATA_PATH,
        max_bytes=MAX_CONFIG_BYTES,
    )
    document = _decode_document(text=text, path=EXTRA_OWNERSHIP_METADATA_PATH)
    if not document:
        raise RuntimeError("extra ownership metadata owners are invalid")
    owners: dict[str, dict[str, str | None]] = {}
    for owner, entry in document.items():
        if (
            not TEAM_OWNER_ID_RE.fullmatch(owner)
            or not isinstance(entry, dict)
            or "description" not in entry
            or not set(entry) <= {"description", "bypass_intake_criteria"}
        ):
            raise RuntimeError(f"extra ownership metadata entry is invalid: {owner}")
        for key in ("description", "bypass_intake_criteria"):
            value = entry.get(key)
            if key in entry and (
                not isinstance(value, str) or not value.strip() or len(value) > 2_000
            ):
                raise RuntimeError(f"ownership metadata {key} is invalid: {owner}")
        owners[owner] = {
            "description": entry["description"],
            "bypass_intake_criteria": entry.get("bypass_intake_criteria"),
        }
    return {"source": source, "owners": owners}


def load_team_members(*, repository_root: Path, repo: str, ref: str) -> dict[str, Any]:
    """Load the reviewer rosters used by planning and live apply."""

    text, source = _load_document(
        repository_root=repository_root,
        repo=repo,
        ref=ref,
        path=TEAM_MEMBERS_PATH,
        max_bytes=MAX_CONFIG_BYTES,
    )
    rosters = _decode_document(text=text, path=TEAM_MEMBERS_PATH)
    if not rosters:
        raise RuntimeError("team member rosters are invalid")
    canonical_members: dict[str, str] = {}
    for owner_id, roster in rosters.items():
        try:
            owner_label(owner_id)
        except ValueError as exc:
            raise RuntimeError(
                "team_members.json has an invalid team owner ID"
            ) from exc
        if not isinstance(roster, list) or not roster:
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
    return {"source": source, "members": rosters}
