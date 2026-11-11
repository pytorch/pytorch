"""Resolve the codepath-owner policy and build compact LLM input."""

from __future__ import annotations

import argparse
import functools
import json
import re
from pathlib import Path
from typing import Any

from identifiers import CODEPATH_OWNER_RE


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


def matches(*, pattern: str, path: str) -> bool:
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


def parse_rule(*, raw: str, line_number: int) -> dict[str, Any]:
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
                f"invalid codepath-owner pattern character {character!r} on line {line_number}"
            )
        pattern_characters.append(character)
        escaped = False

    pattern = "".join(pattern_characters)
    glob_regex(pattern)
    raw_owners = line[owner_start:].split()
    invalid_owner = next(
        (owner for owner in raw_owners if not CODEPATH_OWNER_RE.fullmatch(owner)), None
    )
    if invalid_owner:
        raise ValueError(
            f"invalid codepath owner {invalid_owner!r} on line {line_number}"
        )
    owners = list(dict.fromkeys(owner.casefold() for owner in raw_owners))
    return {
        "line": line_number,
        "pattern": pattern,
        "owners": owners,
        "comment": inline_comment,
    }


def parse_rules(
    *,
    contents: str,
    blob_sha: str | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
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
            rule = parse_rule(raw=raw, line_number=line_number)
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
    *, path: str, rules: list[dict[str, Any]] | tuple[dict[str, Any], ...]
) -> dict[str, Any] | None:
    """Return the last matching rule, including an ownerless override."""

    return next(
        (
            rule
            for rule in reversed(rules)
            if matches(pattern=rule["pattern"], path=path)
        ),
        None,
    )


def resolve_paths(
    *,
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
        rule = resolve_rule(path=path, rules=rules)
        resolutions.append(
            {
                "path": path,
                "owners": [] if rule is None else list(rule["owners"]),
                "matched_rule": rule,
            }
        )
    return resolutions


def build_llm_artifact(resolutions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the exact compact codepath-owner projection shown to the LLM."""

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
            # No matching rule and an ownerless override (a last matching rule
            # with no owners) are treated the same: the file has no codepath owner.
            paths_without_owners.append(path)
    return {
        "owners": sorted(owners, key=str.casefold),
        "matched_path_groups": [
            {"owners": list(group_owners), "paths": paths}
            for group_owners, paths in groups.items()
        ],
        "paths_without_owners": paths_without_owners,
    }


def resolve_for_llm(
    *, paths: list[str] | tuple[str, ...], snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Resolve paths against a loaded snapshot and return the LLM projection."""

    return build_llm_artifact(resolve_paths(paths=paths, rules=snapshot["rules"]))


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
        contents=args.codepath_owners.read_text(),
        diagnostics=diagnostics,
        strict=args.strict,
    )
    artifact = build_llm_artifact(resolve_paths(paths=args.paths, rules=rules))
    result = artifact["owners"] if args.owners_only else artifact
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
