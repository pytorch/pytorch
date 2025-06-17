#!/usr/bin/env python3
import sys
from pathlib import Path

from tools.dynamo.gb_id_mapping import find_unimplemented_v2_calls, load_registry


def test_verify_gb_id_mapping():
    """
    Verifies that all unimplemented_v2 calls in torch/_dynamo match entries in the registry.
    """
    script_dir = Path(__file__).resolve().parent
    dynamo_dir = script_dir.parent.parent / "torch" / "_dynamo"
    registry_path = script_dir.parent.parent / "torch" / "_dynamo" / "graph_break_registry.json"

    python_files = list(dynamo_dir.glob("**/*.py"))

    reg = load_registry(registry_path)
    gb_type_to_entry = {entries[0]["Gb_type"]: entries[0] for _, entries in reg.items()}

    mismatches = []
    for file_path in python_files:
        calls = find_unimplemented_v2_calls(file_path)
        for call in calls:
            gb_type = call["gb_type"]
            if gb_type not in gb_type_to_entry:
                mismatches.append((gb_type, file_path, "Not found in registry"))
                continue

            entry = gb_type_to_entry[gb_type]
            if call["context"] != entry["Context"]:
                mismatches.append((gb_type, file_path, "Context mismatch"))
            elif call["explanation"] != entry["Explanation"]:
                mismatches.append((gb_type, file_path, "Explanation mismatch"))
            elif sorted(call["hints"]) != sorted(entry["Hints"]):
                mismatches.append((gb_type, file_path, "Hints mismatch"))

    if mismatches:
        print("Found unimplemented_v2 calls that don't match the registry.")
        for gb_type, file_path, reason in mismatches:
            print(f"  - {gb_type} in {file_path}: {reason}")
        return False

    print("All unimplemented_v2 calls match the registry.")
    return True



if __name__ == "__main__":
    success = test_verify_gb_id_mapping()
    sys.exit(0 if success else 1)
