"""Resolve the codepath-owner policy and build compact model input."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
from pathlib import Path
from typing import Any


CODEPATH_OWNERS_PATH = ".github/auto-pr-triage/codepath_owners.txt"
MAX_CODEPATH_OWNERS_BYTES = 3_000_000
SHA_RE = re.compile(r"[0-9a-f]{40}")
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
ACCOUNT_PATTERN = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
TEAM_SLUG_PATTERN = r"[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?"
OWNER_HANDLE_PATTERN = rf"(?:@{ACCOUNT_PATTERN}|@{ACCOUNT_PATTERN}/{TEAM_SLUG_PATTERN})"
OWNER_ID_PATTERN = r"[a-z][a-z0-9_-]{0,63}"
OWNER_HANDLE_RE = re.compile(OWNER_HANDLE_PATTERN)
OWNER_ID_RE = re.compile(OWNER_ID_PATTERN)
OWNER_RE = re.compile(rf"(?:{OWNER_HANDLE_PATTERN}|{OWNER_ID_PATTERN})")
PATTERN_PUNCTUATION = set("*?./@_+-:\\()|{}[]~^")


@functools.lru_cache(maxsize=None)
def glob_regex(pattern: str) -> re.Pattern[str]:
    """Compile a codepath-owner pattern using the supported glob semantics."""

    if "***" in pattern:
        raise ValueError(
            "codepath-owner pattern cannot contain three consecutive asterisks"
        )
    if not pattern:
        raise ValueError("empty codepath-owner pattern")
    if pattern == "/":
        return re.compile(r"\A\Z")

    segments = pattern.split("/")
    if segments[0] == "":
        segments = segments[1:]
    elif len(segments) == 1 or (len(segments) == 2 and segments[1] == ""):
        if segments[0] != "**":
            segments.insert(0, "**")
    if len(segments) > 1 and segments[-1] == "":
        segments[-1] = "**"

    last_index = len(segments) - 1
    need_slash = False
    expression = [r"\A"]
    for index, segment in enumerate(segments):
        if segment == "**":
            if index == 0 and index == last_index:
                expression.append(r".+")
            elif index == 0:
                expression.append(r"(?:.+/)?")
                need_slash = False
            elif index == last_index:
                expression.append(r"/.*")
            else:
                expression.append(r"(?:/.+)?")
                need_slash = True
            continue

        if need_slash:
            expression.append("/")
        if segment == "*":
            expression.append(r"[^/]+")
            need_slash = True
            continue

        escaped = False
        for character in segment:
            if escaped:
                expression.append(re.escape(character))
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "*":
                expression.append(r"[^/]*")
            elif character == "?":
                expression.append(r"[^/]")
            else:
                expression.append(re.escape(character))
        if index == last_index:
            expression.append(r"(?:/.*)?")
        need_slash = True

    expression.append(r"\Z")
    return re.compile("".join(expression))


def matches(pattern: str, path: str) -> bool:
    """Return whether one repository-relative path matches a pattern."""

    if pattern.startswith("/") and not set(pattern) & set("*?\\"):
        prefix = pattern[1:]
        return bool(prefix) and (
            path.startswith(prefix)
            if prefix.endswith("/")
            else path == prefix or path.startswith(prefix + "/")
        )
    return bool(glob_regex(pattern).match(path))


def _valid_pattern_character(character: str) -> bool:
    return character.isascii() and (
        character.isalnum() or character in PATTERN_PUNCTUATION
    )


def parse_rule(raw: str, line_number: int) -> dict[str, Any]:
    """Parse one non-comment codepath-owner line."""

    line = raw.strip()
    line, marker, comment = line.partition("#")
    inline_comment = comment.strip() if marker else ""
    if not line:
        raise ValueError(f"invalid codepath-owner line {line_number}: {raw}")

    pattern_characters = []
    escaped = False
    owner_start = len(line)
    for index, character in enumerate(line):
        if character in " \t\n" and not escaped:
            owner_start = index
            break
        if character == "\\":
            escaped = True
            pattern_characters.append(character)
            continue
        if not escaped and not _valid_pattern_character(character):
            raise ValueError(
                f"invalid codepath-owner pattern character {character!r} "
                f"on line {line_number}"
            )
        pattern_characters.append(character)
        escaped = False

    pattern = "".join(pattern_characters)
    glob_regex(pattern)
    owners = line[owner_start:].split()
    invalid_owner = next(
        (owner for owner in owners if not OWNER_RE.fullmatch(owner)), None
    )
    if invalid_owner:
        raise ValueError(
            f"invalid codepath owner {invalid_owner!r} on line {line_number}"
        )
    return {
        "line": line_number,
        "pattern": pattern,
        "owners": owners,
        "comment": inline_comment,
    }


def parse_rules(
    contents: str,
    blob_sha: str | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Parse valid rules and optionally record invalid-line diagnostics."""

    rules = []
    preceding_comments = []
    for line_number, raw in enumerate(contents.splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            preceding_comments = []
            continue
        if stripped.startswith("#"):
            preceding_comments.append(stripped.removeprefix("#").strip())
            continue
        try:
            rule = parse_rule(raw, line_number)
        except ValueError as error:
            if strict:
                raise
            if diagnostics is not None:
                diagnostics.append(
                    {"line": line_number, "raw": raw, "error": str(error)}
                )
            preceding_comments = []
            continue
        rule["preceding_comments"] = preceding_comments
        if blob_sha:
            rule["rule_id"] = f"{blob_sha}:L{line_number}"
        rules.append(rule)
        preceding_comments = []
    return rules


def resolve_rule(
    path: str, rules: list[dict[str, Any]] | tuple[dict[str, Any], ...]
) -> dict[str, Any] | None:
    """Return the last matching rule, including an ownerless override."""

    return next(
        (rule for rule in reversed(rules) if matches(rule["pattern"], path)), None
    )


def resolve_paths(
    paths: list[str] | tuple[str, ...],
    rules: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Resolve unique paths in their first-seen order."""

    resolutions = []
    for path in dict.fromkeys(paths):
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "\0" in path
        ):
            raise ValueError(f"invalid repository-relative path: {path!r}")
        rule = resolve_rule(path, rules)
        resolutions.append(
            {
                "path": path,
                "owners": [] if rule is None else list(rule["owners"]),
                "matched_rule": rule,
            }
        )
    return resolutions


def build_llm_artifact(resolutions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the exact compact codepath-owner projection shown to the model."""

    groups: dict[tuple[str, ...], list[str]] = {}
    paths_without_owners = []
    owners = set()
    for resolution in resolutions:
        path = resolution["path"]
        resolved_owners = tuple(resolution["owners"])
        if resolved_owners:
            owners.update(resolved_owners)
            groups.setdefault(resolved_owners, []).append(path)
        else:
            paths_without_owners.append(
                {
                    "path": path,
                    "reason": (
                        "no_matching_rule"
                        if resolution["matched_rule"] is None
                        else "ownerless_override"
                    ),
                }
            )
    return {
        "owners": sorted(owners, key=str.casefold),
        "matched_path_groups": [
            {"owners": list(group_owners), "paths": paths}
            for group_owners, paths in groups.items()
        ],
        "paths_without_owners": paths_without_owners,
    }


def resolve_for_llm(
    paths: list[str] | tuple[str, ...], snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Resolve paths against a loaded snapshot and return the model projection."""

    return build_llm_artifact(resolve_paths(paths, snapshot["rules"]))


def build_codepath_owners(
    paths: list[str] | tuple[str, ...], snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Build the trusted codepath-owner section stored in TriageInput."""

    return {"source": dict(snapshot["source"]), **resolve_for_llm(paths, snapshot)}


def load_codepath_owners(
    path: Path,
    repo: str,
    ref: str,
) -> dict[str, Any]:
    """Load the repository's single trusted codepath-owner policy file."""

    if not REPOSITORY_RE.fullmatch(repo):
        raise ValueError("repository must be an owner/name pair")
    if not SHA_RE.fullmatch(ref):
        raise ValueError("codepath-owner ref must be an immutable commit SHA")
    expected_parts = Path(CODEPATH_OWNERS_PATH).parts
    if path.parts[-len(expected_parts) :] != expected_parts or path.is_symlink():
        raise ValueError(f"codepath-owner path must name {CODEPATH_OWNERS_PATH}")
    text = _read_local_codepath_owners(path)
    content = text.encode("utf-8")
    header = f"blob {len(content)}\0".encode()
    blob_sha = hashlib.sha1(header + content).hexdigest()

    diagnostics: list[dict[str, Any]] = []
    rules = parse_rules(text, blob_sha, diagnostics, strict=True)
    target_org = repo.split("/", 1)[0].casefold()
    canonical_owners: dict[str, str] = {}
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
            key = owner.casefold()
            previous = canonical_owners.get(key)
            if previous is not None and previous != owner:
                raise RuntimeError("codepath-owner casing is inconsistent")
            canonical_owners[key] = owner
    return {
        "source": {
            "repository": repo,
            "path": CODEPATH_OWNERS_PATH,
            "ref": ref,
            "blob_sha": blob_sha,
        },
        "rules": rules,
        "parse_diagnostics": diagnostics,
    }


def _read_local_codepath_owners(path: Path) -> str:
    content = path.read_bytes()
    if len(content) >= MAX_CODEPATH_OWNERS_BYTES:
        raise ValueError("codepath-owner file exceeds the 3 MB limit")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("codepath-owner file is not valid UTF-8") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codepath-owners", required=True, type=Path)
    parser.add_argument(
        "--owners-only",
        action="store_true",
        help="emit only the sorted JSON list of codepath owners",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    diagnostics: list[dict[str, Any]] = []
    rules = parse_rules(
        _read_local_codepath_owners(args.codepath_owners),
        diagnostics=diagnostics,
        strict=args.strict,
    )
    artifact = build_llm_artifact(resolve_paths(args.paths, rules))
    result = artifact["owners"] if args.owners_only else artifact
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
