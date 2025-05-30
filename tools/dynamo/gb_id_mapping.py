# mypy: ignore-errors

import argparse
import ast
import json
import re
from pathlib import Path


def get_source_segment(source, node):
    if hasattr(ast, "get_source_segment"):
        return ast.get_source_segment(source, node)


"""
Normalizes string literals by removing formatting artifacts and escape sequences.
Handles f-strings, quotes, newlines, and other syntax elements for cleaner output.
"""


def clean_string(s):
    if s is None:
        return None
    if isinstance(s, str):
        s = re.sub(r'^f["\']', r'"', s)
        s = re.sub(r'["\'] f["\']', " ", s)
        s = re.sub(r'^["\'](.*)["\']$', r"\1", s)
        s = re.sub(r"\s*\n\s*", " ", s)
        s = s.replace('\\"', '"').replace("\\'", "'")
        s = s.replace("\\", "")
        s = s.replace("\\", "")
        s = re.sub(r'" "', " ", s)
        s = re.sub(r"\{[^}]*\}", "", s)
        s = re.sub(r"``", "", s)
    return s


# Expands hint references to their actual values from graph_break_hints.
def expand_hints(hints):
    import inspect

    from torch._dynamo import graph_break_hints

    hint_constants = {
        name: value
        for name, value in inspect.getmembers(graph_break_hints)
        if isinstance(value, list) and name.isupper()
    }

    expanded_hints = []
    for hint in hints:
        for name, value in hint_constants.items():
            if f"*graph_break_hints.{name}" in hint:
                expanded_hints.extend(value)
                break
    return expanded_hints


def find_unimplemented_v2_calls(dynamo_dir):
    results = []
    dynamo_dir = Path(dynamo_dir)

    for file_path in dynamo_dir.glob("**/*.py"):
        with open(file_path) as f:
            source = f.read()
            try:
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        is_unimplemented = False
                        if (
                            isinstance(node.func, ast.Name)
                            and node.func.id == "unimplemented_v2"
                        ):
                            is_unimplemented = True

                        if is_unimplemented:
                            info = {
                                "gb_type": None,
                                "context": None,
                                "explanation": None,
                                "hints": [],
                            }

                            for kw in node.keywords:
                                if kw.arg in info:
                                    param_source = get_source_segment(source, kw.value)
                                    if isinstance(kw.value, ast.Constant):
                                        info[kw.arg] = kw.value.value
                                    else:
                                        info[kw.arg] = clean_string(param_source)

                            if info["gb_type"] is None:
                                continue

                            if info["hints"]:
                                hints = info["hints"]
                                expanded_hints = []
                                items = re.findall(r'"([^"]*)"', hints)
                                if items:
                                    expanded_hints.extend(items)

                                if "*graph_break_hints." in hints:
                                    expanded_hints.extend(expand_hints([hints]))

                                info["hints"] = expanded_hints

                            results.append(info)
            except SyntaxError:
                print(f"Syntax error in {file_path}")

    return results


def create_registry(dynamo_dir, registry_path):
    calls = find_unimplemented_v2_calls(dynamo_dir)
    registry = {}

    gb_types = {}
    for info in calls:
        gb_types[info["gb_type"]] = info

    GB_ID_INDEX = 0000
    for i, (gb_type, info) in enumerate(sorted(gb_types.items()), GB_ID_INDEX):
        gb_id = f"GB{i}"
        hints = info["hints"]

        registry[gb_id] = {
            "Gb_type": gb_type,
            "Context": info["context"],
            "Explanation": info["explanation"],
            "Hints": hints if hints else [],
        }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Manage graph break registry.")
    parser.add_argument(
        "--dynamo-dir",
        type=str,
        default=str(Path(__file__).parent.parent.parent / "torch" / "_dynamo"),
        help="Directory to search for unimplemented_v2 calls.",
    )
    parser.add_argument(
        "--registry-path",
        type=str,
        default=str(Path(__file__).parent / "graph_break_registry.json"),
        help="Path to save the registry JSON file.",
    )
    args = parser.parse_args()

    create_registry(args.dynamo_dir, args.registry_path)


if __name__ == "__main__":
    main()
