# mypy: ignore-errors
from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dynamo.gb_id_mapping import find_unimplemented_v2_calls, load_registry


LINTER_CODE = "GB_REGISTRY"


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def _collect_calls(dynamo_dir: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    """Return mapping *gb_type → (call_info, file_path)* for first occurrence."""
    seen: dict[str, tuple[dict[str, Any], Path]] = {}
    for py_file in dynamo_dir.rglob("*.py"):
        for call in find_unimplemented_v2_calls(py_file, dynamo_dir):
            gbt = call["gb_type"]
            if gbt not in seen:
                seen[gbt] = (call, py_file)
    return seen


def _detect_duplicate_gb_types_in_source(dynamo_dir: Path) -> list[dict[str, Any]]:
    """Detect cases where the same gb_type is used multiple times with different content."""
    gb_type_calls: dict[str, list[tuple[dict[str, Any], Path]]] = {}

    for py_file in dynamo_dir.rglob("*.py"):
        for call in find_unimplemented_v2_calls(py_file, dynamo_dir):
            gb_type = call["gb_type"]
            if gb_type not in gb_type_calls:
                gb_type_calls[gb_type] = []
            gb_type_calls[gb_type].append((call, py_file))

    duplicates = []
    for gb_type, call_list in gb_type_calls.items():
        if len(call_list) > 1:
            first_call = call_list[0][0]
            for call, file_path in call_list[1:]:
                if (
                    call["context"] != first_call["context"]
                    or call["explanation"] != first_call["explanation"]
                    or sorted(call["hints"]) != sorted(first_call["hints"])
                ):
                    duplicates.append({"gb_type": gb_type, "calls": call_list})
                    break

    return duplicates


def _update_registry_with_changes(
    registry: dict, calls: dict[str, tuple[dict[str, Any], Path]]
) -> dict:
    """Calculate what the updated registry should look like."""
    updated_registry = dict(registry)

    latest_entry: dict[str, Any] = {
        entries[0]["Gb_type"]: entries[0] for entries in registry.values()
    }

    for gb_type, (call, file_path) in calls.items():
        if gb_type in latest_entry:
            existing_entry = latest_entry[gb_type]

            if not (
                call["context"] == existing_entry["Context"]
                and call["explanation"] == existing_entry["Explanation"]
                and sorted(call["hints"]) == sorted(existing_entry["Hints"])
            ):
                for key, entries in updated_registry.items():
                    if entries[0]["Gb_type"] == gb_type:
                        new_entry = {
                            "Gb_type": gb_type,
                            "Context": call["context"],
                            "Explanation": call["explanation"],
                            "Hints": call["hints"],
                        }
                        updated_registry[key] = [new_entry] + entries
                        break
        else:
            gb_keys = []
            numeric_keys = []

            for k in updated_registry.keys():
                if k.startswith("GB") and k[2:].isdigit():
                    gb_keys.append(int(k[2:]))
                elif k.isdigit():
                    numeric_keys.append(int(k))

            if gb_keys:
                new_key_num = max(gb_keys) + 1
                new_key = f"GB{new_key_num:04d}"
            elif numeric_keys:
                new_key_num = max(numeric_keys) + 1
                new_key = str(new_key_num)
            else:
                new_key = "GB0000"

            updated_registry[new_key] = [
                {
                    "Gb_type": gb_type,
                    "Context": call["context"],
                    "Explanation": call["explanation"],
                    "Hints": call["hints"],
                }
            ]

    return updated_registry


def check_registry_sync() -> list[LintMessage]:
    """Check registry sync and return lint messages."""
    lint_messages = []

    script_dir = Path(__file__).resolve()
    repo_root = script_dir.parents[3]
    dynamo_dir = repo_root / "torch" / "_dynamo"
    registry_path = dynamo_dir / "graph_break_registry.json"

    source_duplicates = _detect_duplicate_gb_types_in_source(dynamo_dir)
    for dup in source_duplicates:
        gb_type = dup["gb_type"]
        calls = dup["calls"]

        description = f"The gb_type '{gb_type}' is used {len(calls)} times with different content. "
        description += "Each gb_type must be unique across your entire codebase."

        lint_messages.append(
            LintMessage(
                path=str(calls[0][1]),
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="Duplicate gb_type",
                original=None,
                replacement=None,
                description=description,
            )
        )

    if source_duplicates:
        return lint_messages

    calls = _collect_calls(dynamo_dir)
    registry = load_registry(registry_path)
    latest_entry: dict[str, Any] = {
        entries[0]["Gb_type"]: entries[0] for entries in registry.values()
    }

    gb_type_changes = []
    for gb_type, (call, file_path) in calls.items():
        if gb_type not in latest_entry:
            for existing_gb_type, existing_entry in latest_entry.items():
                if (
                    call["context"] == existing_entry["Context"]
                    and call["explanation"] == existing_entry["Explanation"]
                    and sorted(call["hints"]) == sorted(existing_entry["Hints"])
                ):
                    gb_type_changes.append((existing_gb_type, gb_type, file_path))
                    break

    for old_gb_type, new_gb_type, file_path in gb_type_changes:
        description = f"Detected gb_type rename: '{old_gb_type}' -> '{new_gb_type}'. Please manually update:\n"
        command = (
            f"'python tools/dynamo/gb_id_mapping.py update "
            f"{json.dumps(old_gb_type)} {file_path} --new_gb_type {json.dumps(new_gb_type)}'"
        )
        description += command

        lint_messages.append(
            LintMessage(
                path=str(file_path),
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="GB_TYPE rename detected",
                original=None,
                replacement=None,
                description=description,
            )
        )

    if gb_type_changes:
        return lint_messages

    needs_update = False
    for gb_type, (call, file_path) in calls.items():
        if gb_type in latest_entry:
            existing_entry = latest_entry[gb_type]
            if not (
                call["context"] == existing_entry["Context"]
                and call["explanation"] == existing_entry["Explanation"]
                and sorted(call["hints"]) == sorted(existing_entry["Hints"])
            ):
                needs_update = True
                break
        else:
            needs_update = True
            break

    if needs_update:
        updated_registry = _update_registry_with_changes(registry, calls)

        original_content = registry_path.read_text(encoding="utf-8")

        replacement_content = (
            json.dumps(updated_registry, indent=2, ensure_ascii=False) + "\n"
        )

        lint_messages.append(
            LintMessage(
                path=str(registry_path),
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.WARNING,
                name="Registry sync needed",
                original=original_content,
                replacement=replacement_content,
                description="Registry is out of sync with source code. Run `lintrunner -a` to apply changes.",
            )
        )

    return lint_messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-sync graph break registry with source code",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="paths to lint",
    )

    args = parser.parse_args()

    lint_messages = check_registry_sync()

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
