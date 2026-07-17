# Owner(s): ["module: ci"]

"""Select tests for a commit using observed file dependencies.

TD tracer outputs can be large, so this module reads ``coverage_by_test``
directly from a memory-mapped file instead of loading the full JSON document.
The result contains positive matches from the trace; an incomplete trace may
omit other affected tests. Tests whose source file is affected are also selected
when the trace contains a source-qualified test ID for that file.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import posixpath
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_VERSION = 4
_WHITESPACE = re.compile(rb"[ \t\r\n]*")
_STRUCTURAL = re.compile(rb'["{}\[\]]')
_VALUE_END = re.compile(rb"[,}\]]")
_NON_PLAIN_ASCII = re.compile(rb"[\x00-\x1f\x7f-\xff]")
_REQUIRED_MEMBERS = {
    "schema_version",
    "run_id",
    "complete",
    "successful",
    "usable",
    "running_participants",
    "participants",
    "environments",
    "coverage_by_test",
}


class TDSelectionError(ValueError):
    pass


def _git_output(args: Sequence[str], repo_root: Path) -> bytes:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        message = f": {detail}" if detail else ""
        raise TDSelectionError(f"Git command failed{message}") from error
    except OSError as error:
        raise TDSelectionError(f"Unable to run Git: {error}") from error
    return result.stdout


def changed_files_for_commit(commit: str, repo_root: Path = REPO_ROOT) -> list[str]:
    """Return repository-relative paths changed by a commit.

    Merge commits are compared with their first parent. Rename detection is
    disabled so both the deleted and added paths are returned.
    """

    if not commit or not commit.isprintable():
        raise TDSelectionError("Commit must be a nonempty printable revision")
    if commit.startswith("-"):
        raise TDSelectionError("Commit must not start with '-'")
    revision = f"{commit}^{{commit}}"
    resolved = _git_output(["rev-parse", "--verify", revision], repo_root)
    try:
        oid = resolved.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise TDSelectionError("Git returned an invalid commit ID") from error
    if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", oid) is None:
        raise TDSelectionError("Git returned an invalid commit ID")

    parents_output = _git_output(
        ["rev-list", "--parents", "-n", "1", oid, "--"], repo_root
    )
    try:
        commit_and_parents = parents_output.decode("ascii").strip().split()
    except UnicodeDecodeError as error:
        raise TDSelectionError("Git returned an invalid parent list") from error
    if (
        not commit_and_parents
        or commit_and_parents[0] != oid
        or any(
            re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", value) is None
            for value in commit_and_parents
        )
    ):
        raise TDSelectionError("Git returned an invalid parent list")

    diff_args = [
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        "-z",
        "--no-renames",
    ]
    if len(commit_and_parents) == 1:
        diff_args.extend(["--root", oid])
    else:
        diff_args.extend([commit_and_parents[1], oid])
    output = _git_output([*diff_args, "--"], repo_root)
    try:
        paths = [path.decode("utf-8") for path in output.split(b"\0") if path]
    except UnicodeDecodeError as error:
        raise TDSelectionError("Git returned a non-UTF-8 path") from error
    return sorted(set(paths))


def _normalize_affected_file(path: str | os.PathLike[str]) -> str:
    value = os.fspath(path)
    if not isinstance(value, str) or not value:
        raise TDSelectionError("Affected file paths must be nonempty strings")
    value = value.replace("\\", "/")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise TDSelectionError(f"Affected file path must be relative: {value}")
    if ".." in value.split("/"):
        raise TDSelectionError(f"Affected file path must not contain '..': {value}")
    value = posixpath.normpath(value)
    if value == ".":
        raise TDSelectionError("Affected file paths must name files")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise TDSelectionError("Affected file paths must be valid UTF-8") from error
    if not value.isprintable():
        raise TDSelectionError("Affected file paths must be printable")
    return value


@dataclass(frozen=True)
class TraceMetadata:
    schema_version: int
    run_id: str
    complete: bool
    successful: bool
    usable: bool
    revisions: tuple[str, ...]


@dataclass(frozen=True)
class SelectionResult:
    tests: list[str]
    matches_by_test: dict[str, list[str]]
    affected: frozenset[str]
    matched: frozenset[str]
    metadata: TraceMetadata

    @property
    def unsupported(self) -> list[str]:
        return sorted(
            path
            for path in self.affected
            if not path.endswith(".py") or not path.startswith(("torch/", "test/"))
        )

    @property
    def unmatched(self) -> list[str]:
        unsupported = set(self.unsupported)
        return sorted(self.affected - self.matched - unsupported)


class _MappedJSON:
    def __init__(self, data: mmap.mmap, path: Path) -> None:
        self.data = data
        self.path = path
        self._discarded = 0
        self._can_discard = hasattr(data, "madvise") and hasattr(mmap, "MADV_DONTNEED")
        if hasattr(data, "madvise") and hasattr(mmap, "MADV_SEQUENTIAL"):
            try:
                data.madvise(mmap.MADV_SEQUENTIAL)
            except (OSError, ValueError):
                pass

    def _error(self, message: str) -> TDSelectionError:
        return TDSelectionError(f"Invalid TD tracer JSON in {self.path}: {message}")

    def skip_whitespace(self, position: int) -> int:
        match = _WHITESPACE.match(self.data, position)
        assert match is not None
        return match.end()

    def discard_before(self, position: int) -> None:
        if not self._can_discard:
            return
        end = position - position % mmap.PAGESIZE
        if end - self._discarded < 64 * 1024 * 1024:
            return
        try:
            self.data.madvise(
                mmap.MADV_DONTNEED, self._discarded, end - self._discarded
            )
        except (OSError, ValueError):
            self._can_discard = False
        else:
            self._discarded = end

    def expect(self, position: int, token: int) -> int:
        position = self.skip_whitespace(position)
        if position >= len(self.data) or self.data[position] != token:
            raise self._error(f"expected {chr(token)!r}")
        return position + 1

    def string_span(self, position: int) -> tuple[int, int, int, bool]:
        position = self.skip_whitespace(position)
        if position >= len(self.data) or self.data[position] != ord('"'):
            raise self._error("expected a string")

        start = position + 1
        search_from = start
        while True:
            end = self.data.find(b'"', search_from)
            if end < 0:
                raise self._error("unterminated string")
            backslash = end - 1
            while backslash >= start and self.data[backslash] == ord("\\"):
                backslash -= 1
            if (end - backslash - 1) % 2 == 0:
                escaped = self.data.find(b"\\", start, end) >= 0
                return start, end, end + 1, escaped
            search_from = end + 1

    def string(self, position: int) -> tuple[str, int]:
        start, end, position, _ = self.string_span(position)
        try:
            value = json.loads(self.data[start - 1 : end + 1])
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise self._error("invalid string") from error
        if not isinstance(value, str):
            raise self._error("expected a string")
        return value, position

    def scalar(self, position: int) -> tuple[object, int]:
        position = self.skip_whitespace(position)
        match = _VALUE_END.search(self.data, position)
        end = len(self.data) if match is None else match.start()
        try:
            return json.loads(self.data[position:end]), end
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise self._error("invalid value") from error

    def value(self, position: int) -> tuple[object, int]:
        position = self.skip_whitespace(position)
        if position >= len(self.data):
            raise self._error("missing value")
        token = self.data[position]
        if token == ord('"'):
            return self.string(position)
        if token not in (ord("{"), ord("[")):
            return self.scalar(position)

        start = position
        closing_tokens = [ord("}") if token == ord("{") else ord("]")]
        position += 1
        while closing_tokens:
            match = _STRUCTURAL.search(self.data, position)
            if match is None:
                raise self._error("unterminated value")
            position = match.start()
            token = self.data[position]
            if token == ord('"'):
                position = self.string_span(position)[2]
            elif token == ord("{"):
                closing_tokens.append(ord("}"))
                position += 1
            elif token == ord("["):
                closing_tokens.append(ord("]"))
                position += 1
            elif token != closing_tokens.pop():
                raise self._error("mismatched container")
            else:
                position += 1
        try:
            return json.loads(self.data[start:position]), position
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise self._error("invalid value") from error


def _dependency_matches(
    parser: _MappedJSON,
    position: int,
    affected: set[str],
    affected_bytes: dict[bytes, str],
) -> tuple[str | None, int]:
    start, end, position, escaped = parser.string_span(position)
    if escaped or _NON_PLAIN_ASCII.search(parser.data, start, end) is not None:
        try:
            dependency = json.loads(parser.data[start - 1 : end + 1])
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise parser._error("invalid dependency path") from error
        if (
            not isinstance(dependency, str)
            or not dependency
            or not dependency.isprintable()
        ):
            raise parser._error("invalid dependency path")
        try:
            dependency.encode("utf-8")
        except UnicodeEncodeError as error:
            raise parser._error("invalid dependency path") from error
        return dependency if dependency in affected else None, position
    return affected_bytes.get(parser.data[start:end]), position


def _parse_coverage(
    parser: _MappedJSON,
    position: int,
    affected: set[str],
) -> tuple[dict[str, set[str]], set[str], int]:
    matches_by_test: dict[str, set[str]] = {}
    matched_affected: set[str] = set()
    affected_bytes = {path.encode("utf-8"): path for path in affected}
    source_affected: dict[str, set[str]] = {}
    for path in affected:
        source_affected.setdefault(path, set()).add(path)
        if path.startswith("torch/"):
            source_affected.setdefault(path.removeprefix("torch/"), set()).add(path)
    position = parser.expect(position, ord("{"))
    position = parser.skip_whitespace(position)
    if position < len(parser.data) and parser.data[position] == ord("}"):
        return matches_by_test, matched_affected, position + 1

    while True:
        test, position = parser.string(position)
        if not test or not test.isprintable():
            raise parser._error("invalid test ID")
        position = parser.expect(position, ord(":"))
        position = parser.expect(position, ord("["))
        source = test.split("::", 1)[0].replace("\\", "/")
        direct_matches = (
            source_affected.get(source, set()) if source.endswith(".py") else set()
        )
        matched_paths = set(direct_matches)

        position = parser.skip_whitespace(position)
        if position < len(parser.data) and parser.data[position] == ord("]"):
            position += 1
        else:
            while True:
                dependency_match, position = _dependency_matches(
                    parser, position, affected, affected_bytes
                )
                if dependency_match is not None:
                    matched_paths.add(dependency_match)
                position = parser.skip_whitespace(position)
                if position >= len(parser.data):
                    raise parser._error("unterminated dependency list")
                token = parser.data[position]
                if token == ord("]"):
                    position += 1
                    break
                if token != ord(","):
                    raise parser._error("expected ',' or ']' in dependency list")
                position += 1

        if matched_paths:
            matches_by_test.setdefault(test, set()).update(matched_paths)
            matched_affected.update(matched_paths)

        parser.discard_before(position)
        position = parser.skip_whitespace(position)
        if position >= len(parser.data):
            raise parser._error("unterminated coverage object")
        token = parser.data[position]
        if token == ord("}"):
            return matches_by_test, matched_affected, position + 1
        if token != ord(","):
            raise parser._error("expected ',' or '}' in coverage object")
        position += 1


_PARTICIPANT_TYPES = {
    "complete": bool,
    "exit_status": int,
    "participant_id": str,
    "pid": int,
    "session_id": str,
    "worker_id": str,
}


def _validate_member(key: str, value: object, path: Path) -> None:
    if key == "schema_version":
        if type(value) is not int or value != _SCHEMA_VERSION:
            raise TDSelectionError(f"Unsupported TD tracer schema in {path}: {value!r}")
    elif key == "run_id":
        if not isinstance(value, str) or not value:
            raise TDSelectionError(f"TD tracer output has no run ID in {path}")
    elif key in {"complete", "successful", "usable"}:
        if type(value) is not bool:
            raise TDSelectionError(f"Invalid TD tracer {key} in {path}")
    elif key == "running_participants":
        if not isinstance(value, list) or not all(
            isinstance(participant, str) for participant in value
        ):
            raise TDSelectionError(f"Invalid TD tracer running participants in {path}")
    elif key == "participants":
        if not isinstance(value, list) or any(
            not isinstance(participant, dict)
            or any(
                type(participant.get(field)) is not expected_type
                for field, expected_type in _PARTICIPANT_TYPES.items()
            )
            for participant in value
        ):
            raise TDSelectionError(f"Invalid TD tracer participants in {path}")
    elif key == "environments":
        if not isinstance(value, list) or any(
            not isinstance(environment, dict)
            or (
                environment.get("revision") is not None
                and not isinstance(environment.get("revision"), str)
            )
            for environment in value
        ):
            raise TDSelectionError(f"Invalid TD tracer environments in {path}")


def _select_tests(
    trace_path: Path, affected_files: Iterable[str | os.PathLike[str]]
) -> SelectionResult:
    affected = {_normalize_affected_file(path) for path in affected_files}
    with trace_path.open("rb") as input_file:
        if os.fstat(input_file.fileno()).st_size == 0:
            raise TDSelectionError(f"TD tracer JSON is empty: {trace_path}")
        with mmap.mmap(input_file.fileno(), 0, access=mmap.ACCESS_READ) as data:
            parser = _MappedJSON(data, trace_path)
            position = parser.expect(0, ord("{"))
            matches_by_test: dict[str, set[str]] | None = None
            matched: set[str] = set()
            members: dict[str, object] = {}

            position = parser.skip_whitespace(position)
            if position < len(data) and data[position] == ord("}"):
                position += 1
            else:
                while True:
                    key, position = parser.string(position)
                    position = parser.expect(position, ord(":"))
                    if key == "coverage_by_test":
                        if matches_by_test is not None:
                            raise parser._error("duplicate coverage_by_test")
                        matches_by_test, matched, position = _parse_coverage(
                            parser, position, affected
                        )
                        members[key] = True
                    else:
                        value, position = parser.value(position)
                        if key in _REQUIRED_MEMBERS:
                            if key in members:
                                raise parser._error(f"duplicate {key}")
                            _validate_member(key, value, trace_path)
                            members[key] = value

                    position = parser.skip_whitespace(position)
                    if position >= len(data):
                        raise parser._error("unterminated top-level object")
                    token = data[position]
                    if token == ord("}"):
                        position += 1
                        break
                    if token != ord(","):
                        raise parser._error("expected ',' or '}'")
                    position += 1

            if parser.skip_whitespace(position) != len(data):
                raise parser._error("unexpected content after top-level object")
            missing = _REQUIRED_MEMBERS - members.keys()
            if missing:
                raise TDSelectionError(
                    f"TD tracer output is missing {', '.join(sorted(missing))} "
                    f"in {trace_path}"
                )
            assert matches_by_test is not None
            schema_version = cast(int, members["schema_version"])
            run_id = cast(str, members["run_id"])
            complete = members["complete"]
            successful = members["successful"]
            usable = members["usable"]
            environments = cast(list[dict[str, str | None]], members["environments"])
            assert isinstance(complete, bool)
            assert isinstance(successful, bool)
            assert isinstance(usable, bool)
            if usable != (complete and successful):
                raise TDSelectionError(f"Inconsistent TD tracer status in {trace_path}")
            running_participants = members["running_participants"]
            assert isinstance(running_participants, list)
            if complete and running_participants:
                raise TDSelectionError(
                    f"Completed TD tracer output has running participants in "
                    f"{trace_path}"
                )
            metadata = TraceMetadata(
                schema_version=schema_version,
                run_id=run_id,
                complete=complete,
                successful=successful,
                usable=usable,
                revisions=tuple(
                    sorted(
                        {
                            revision
                            for environment in environments
                            if (revision := environment.get("revision")) is not None
                        }
                    )
                ),
            )
            return SelectionResult(
                tests=sorted(matches_by_test),
                matches_by_test={
                    test: sorted(matches_by_test[test])
                    for test in sorted(matches_by_test)
                },
                affected=frozenset(affected),
                matched=frozenset(matched),
                metadata=metadata,
            )


def select_tests_with_details(
    trace_path: str | os.PathLike[str],
    affected_files: Iterable[str | os.PathLike[str]],
) -> SelectionResult:
    """Return selected tests, their matching paths, and trace metadata."""

    return _select_tests(Path(trace_path), affected_files)


def select_tests(
    trace_path: str | os.PathLike[str],
    affected_files: Iterable[str | os.PathLike[str]],
) -> list[str]:
    """Return observed dependency and direct test-source matches.

    Affected paths outside ``torch/*.py`` and ``test/*.py``, and paths absent
    from the trace, require a separate conservative fallback.
    """

    return select_tests_with_details(trace_path, affected_files).tests


def select_tests_for_commit(
    trace_path: str | os.PathLike[str],
    commit: str,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Return tests affected by files changed in a commit."""

    return select_tests(trace_path, changed_files_for_commit(commit, repo_root))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select tests for a commit using dependencies observed by the TD tracer."
        )
    )
    parser.add_argument("trace", type=Path, help="TD tracer JSON file")
    parser.add_argument("commit", help="commit or revision to inspect")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    try:
        affected_files = changed_files_for_commit(options.commit)
        result = select_tests_with_details(options.trace, affected_files)
    except (OSError, TDSelectionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for test in sorted(result.tests):
        print(test)
    if not result.metadata.usable:
        print(
            "warning: TD tracer output is not usable; matches only reflect "
            "observed dependencies",
            file=sys.stderr,
        )
    if result.unsupported:
        print(
            f"warning: {len(result.unsupported)} affected "
            f"{'file is' if len(result.unsupported) == 1 else 'files are'} outside "
            f"TD tracer coverage: {', '.join(result.unsupported)}; use a "
            "conservative fallback",
            file=sys.stderr,
        )
    if result.unmatched:
        print(
            f"warning: {len(result.unmatched)} traceable affected "
            f"{'file was' if len(result.unmatched) == 1 else 'files were'} not "
            f"observed in the trace: {', '.join(result.unmatched)}; use a "
            "conservative fallback",
            file=sys.stderr,
        )
    test_count = len(result.tests)
    affected_count = len(result.affected)
    print(
        f"Selected {test_count} {'test' if test_count == 1 else 'tests'} for "
        f"{affected_count} affected "
        f"{'file' if affected_count == 1 else 'files'}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
