# mypy: ignore-errors

from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


from tools.dynamo.gb_id_mapping import (
    find_unimplemented_v2_calls,
    load_registry,
    next_gb_id,
)


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


def _collect_all_calls(
    dynamo_dir: Path,
) -> dict[str, list[tuple[dict[str, Any], Path]]]:
    """Return mapping *gb_type → list[(call_info, file_path)]* for all occurrences."""
    gb_type_calls: dict[str, list[tuple[dict[str, Any], Path]]] = {}

    for py_file in dynamo_dir.rglob("*.py"):
        for call in find_unimplemented_v2_calls(py_file, dynamo_dir):
            gb_type = call["gb_type"]
            if gb_type not in gb_type_calls:
                gb_type_calls[gb_type] = []
            gb_type_calls[gb_type].append((call, py_file))

    return gb_type_calls


def _create_registry_entry(
    gb_type: str, context: str, explanation: str, hints: list[str]
) -> dict[str, Any]:
    """Create a registry entry with consistent format."""
    return {
        "Gb_type": gb_type,
        "Context": context,
        "Explanation": explanation,
        "Hints": hints or [],
    }


def _update_registry_with_changes(
    registry: dict,
    calls: dict[str, tuple[dict[str, Any], Path]],
    renames: dict[str, str] | None = None,
) -> dict:
    """Calculate what the updated registry should look like."""
    renames = renames or {}
    updated_registry = dict(registry)

    latest_entry: dict[str, Any] = {
        entries[0]["Gb_type"]: entries[0] for entries in registry.values()
    }
    gb_type_to_key: dict[str, str] = {
        entries[0]["Gb_type"]: key for key, entries in registry.items()
    }

    # Method for determining add vs. update:
    # - If gb_type exists in registry but content differs: UPDATE (append new entry to preserve history)
    # - If gb_type is new but content matches existing entry: RENAME (append new entry with new gb_type)
    # - If gb_type is completely new: ADD (create new registry entry with a new GBID)

    for old_gb_type, new_gb_type in renames.items():
        registry_key = gb_type_to_key[old_gb_type]
        old_entry = updated_registry[registry_key][0]

        new_entry = _create_registry_entry(
            new_gb_type,
            old_entry["Context"],
            old_entry["Explanation"],
            old_entry["Hints"],
        )
        updated_registry[registry_key] = [new_entry] + updated_registry[registry_key]

        latest_entry[new_gb_type] = new_entry
        gb_type_to_key[new_gb_type] = registry_key
        del latest_entry[old_gb_type]
        del gb_type_to_key[old_gb_type]

    for gb_type, (call, file_path) in calls.items():
        if gb_type in latest_entry:
            existing_entry = latest_entry[gb_type]

            if not (
                call["context"] == existing_entry["Context"]
                and call["explanation"] == existing_entry["Explanation"]
                and sorted(call["hints"]) == sorted(existing_entry["Hints"])
            ):
                registry_key = gb_type_to_key[gb_type]
                new_entry = _create_registry_entry(
                    gb_type, call["context"], call["explanation"], call["hints"]
                )
                updated_registry[registry_key] = [new_entry] + updated_registry[
                    registry_key
                ]
        else:
            new_key = next_gb_id(updated_registry)
            new_entry = _create_registry_entry(
                gb_type, call["context"], call["explanation"], call["hints"]
            )
            updated_registry[new_key] = [new_entry]

    return updated_registry


def check_registry_sync(dynamo_dir: Path, registry_path: Path) -> list[LintMessage]:
    """Check registry sync and return lint messages."""
    lint_messages = []

    all_calls = _collect_all_calls(dynamo_dir)

    duplicates = []
    for gb_type, call_list in all_calls.items():
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

    for dup in duplicates:
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

    if duplicates:
        return lint_messages

    calls = {gb_type: calls[0] for gb_type, calls in all_calls.items()}

    registry = load_registry(registry_path)
    latest_entry: dict[str, Any] = {
        entries[0]["Gb_type"]: entries[0] for entries in registry.values()
    }

    renames: dict[str, str] = {}
    remaining_calls = dict(calls)

    for gb_type, (call, file_path) in calls.items():
        if gb_type not in latest_entry:
            for existing_gb_type, existing_entry in latest_entry.items():
                if (
                    call["context"] == existing_entry["Context"]
                    and call["explanation"] == existing_entry["Explanation"]
                    and sorted(call["hints"]) == sorted(existing_entry["Hints"])
                ):
                    renames[existing_gb_type] = gb_type
                    del remaining_calls[gb_type]
                    break

    needs_update = bool(renames)

    for gb_type, (call, file_path) in remaining_calls.items():
        if gb_type in latest_entry:
            existing_entry = latest_entry[gb_type]

            if not (
                call["context"] == existing_entry["Context"]
                and call["explanation"] == existing_entry["Explanation"]
                and sorted(call["hints"] or []) == sorted(existing_entry["Hints"] or [])
            ):
                needs_update = True
                break
        else:
            needs_update = True
            break

    if needs_update:
        updated_registry = _update_registry_with_changes(
            registry, remaining_calls, renames
        )

        original_content = registry_path.read_text(encoding="utf-8")

        replacement_content = (
            json.dumps(updated_registry, indent=2, ensure_ascii=False) + "\n"
        )

        changes = []
        if renames:
            for old, new in renames.items():
                changes.append(f"renamed '{old}' → '{new}'")
        if remaining_calls:
            new_count = sum(
                1 for gb_type in remaining_calls if gb_type not in latest_entry
            )
            if new_count:
                changes.append(f"added {new_count} new gb_types")

        description = f"Registry sync needed ({', '.join(changes)}). Run `lintrunner -a` to apply changes."

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
                description=description,
            )
        )

    return lint_messages


if __name__ == "__main__":
    script_dir = Path(__file__).resolve()
    repo_root = script_dir.parents[3]
    default_registry_path = (
        repo_root / "torch" / "_dynamo" / "graph_break_registry.json"
    )

    default_dynamo_dir = repo_root / "torch" / "_dynamo"

    parser = argparse.ArgumentParser(
        description="Auto-sync graph break registry with source code"
    )
    parser.add_argument(
        "--dynamo-dir",
        type=Path,
        default=default_dynamo_dir,
        help=f"Path to the dynamo directory (default: {default_dynamo_dir})",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=default_registry_path,
        help=f"Path to the registry file (default: {default_registry_path})",
    )

    args = parser.parse_args()

    lint_messages = check_registry_sync(
        dynamo_dir=args.dynamo_dir, registry_path=args.registry_path
    )

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
