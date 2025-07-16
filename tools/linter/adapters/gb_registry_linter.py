from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, NamedTuple
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dynamo.gb_id_mapping import (
    find_unimplemented_v2_calls,
    load_registry,
    cmd_add_new_gb_type,
    cmd_update_gb_type,
    test_verify_gb_id_mapping,
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

def _collect_calls(dynamo_dir: Path) -> Dict[str, Tuple[dict, Path]]:
    """Return mapping *gb_type → (call_info, file_path)* for first occurrence."""
    seen: Dict[str, Tuple[dict, Path]] = {}
    for py_file in dynamo_dir.rglob("*.py"):
        for call in find_unimplemented_v2_calls(py_file, dynamo_dir):
            gbt = call["gb_type"]
            if gbt not in seen:
                seen[gbt] = (call, py_file)
    return seen

def _detect_duplicate_gb_types_in_source(dynamo_dir: Path) -> list:
    """Detect cases where the same gb_type is used multiple times with different content."""
    gb_type_calls: Dict[str, list] = {}

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
                    duplicates.append({
                        "gb_type": gb_type,
                        "calls": call_list
                    })
                    break

    return duplicates

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

        lint_messages.append(LintMessage(
            path=str(calls[0][1]),
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="Duplicate gb_type",
            original=None,
            replacement=None,
            description=description
        ))

    if source_duplicates:
        return lint_messages

    calls = _collect_calls(dynamo_dir)
    registry = load_registry(registry_path)
    latest_entry: Dict[str, dict] = {
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
        description = f"Detected gb_type rename: '{old_gb_type}' -> '{new_gb_type}'. "
        description += f"Please manually update: python tools/dynamo/gb_id_mapping.py update \"{old_gb_type}\" {file_path} --new_gb_type \"{new_gb_type}\""

        lint_messages.append(LintMessage(
            path=str(file_path),
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="GB_TYPE rename detected",
            original=None,
            replacement=None,
            description=description
        ))

    if gb_type_changes:
        return lint_messages

    # Auto-fix: add new gb_types and update existing ones
    changes_made = []

    for gb_type, (call, file_path) in calls.items():
        if gb_type in latest_entry:
            existing_entry = latest_entry[gb_type]

            if not (
                call["context"] == existing_entry["Context"]
                and call["explanation"] == existing_entry["Explanation"]
                and sorted(call["hints"]) == sorted(existing_entry["Hints"])
            ):
                if cmd_update_gb_type(gb_type, str(file_path), str(registry_path)):
                    changes_made.append(f"Updated gb_type '{gb_type}' in registry")
        else:
            if cmd_add_new_gb_type(gb_type, str(file_path), str(registry_path)):
                changes_made.append(f"Added gb_type '{gb_type}' to registry")

    for change in changes_made:
        lint_messages.append(LintMessage(
            path=None,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ADVICE,
            name="Registry updated",
            original=None,
            replacement=None,
            description=change
        ))

    if changes_made and not test_verify_gb_id_mapping(str(dynamo_dir), str(registry_path)):
        lint_messages.append(LintMessage(
            path=None,
            line=None,
            char=None,
            code=LINTER_CODE,
            severity=LintSeverity.ERROR,
            name="Verification failed",
            original=None,
            replacement=None,
            description="Registry verification failed after auto-sync"
        ))

    return lint_messages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-sync graph break registry with source code",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()

    lint_messages = check_registry_sync()

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
