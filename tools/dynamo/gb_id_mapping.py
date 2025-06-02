# mypy: ignore-errors

import argparse
import ast
import json
import re
from pathlib import Path


def get_source_segment(source, node):
    return ast.get_source_segment(source, node)


def clean_string(s):
    """
    Normalizes string literals by removing formatting artifacts and escape sequences.
    Handles f-strings, quotes, newlines, and other syntax elements for cleaner output.
    """
    if isinstance(s, str):
        # Convert f-string prefix to regular string prefix (e.g., f"hello" -> "hello")
        s = re.sub(r'^f["\']', r'"', s)
        # Replace quoted strings with f-prefix in the middle with a space (e.g., " f"" -> " ")
        s = re.sub(r'["\'] f["\']', " ", s)
        # Remove surrounding quotes, keeping only the content (e.g., "hello" -> hello)
        s = re.sub(r'^["\'](.*)["\']$', r"\1", s)
        # Replace any whitespace around newlines with a single space for consistent formatting
        s = re.sub(r"\s*\n\s*", " ", s)
        # Replace escaped quotes with their unescaped versions (e.g., \" -> ", \' -> ')
        s = s.replace('\\"', '"').replace("\\'", "'")
        # Remove any remaining backslashes used for escaping
        s = s.replace("\\", "")
        # Replace adjacent quoted strings with a space (e.g., " "" -> " ")
        s = re.sub(r'" "', " ", s)
        # Remove any curly brace expressions used in f-strings (e.g., {variable})
        s = re.sub(r"\{[^}]*\}", "", s)
        # Remove backticks used in docstrings or code examples
        s = re.sub(r"``", "", s)
    return s


def expand_hints(hints):
    # Expands hint references to their actual values from graph_break_hints.
    from torch._dynamo import graph_break_hints

    hint_constants = {
        name: value
        for name, value in graph_break_hints.__dict__.items()
        if isinstance(value, list) and name.isupper()
    }

    expanded_hints = []
    for hint in hints:
        for name, value in hint_constants.items():
            if f"*graph_break_hints.{name}" in hint:
                expanded_hints.extend(value)
                break
    return expanded_hints


def extract_info_from_keyword(source, kw):
    """
    Extracts and returns the value of a keyword argument from an AST node.

    This function handles different types of AST nodes:
    - If the node is a constant, it returns the constant value.
    - If the node is an f-string, it reconstructs the string by
      evaluating formatted values and concatenating them with string literals.
    - For other types, it cleans the source segment to remove formatting artifacts.

    """
    param_source = get_source_segment(source, kw.value)
    if isinstance(kw.value, ast.Constant):
        return kw.value.value
    elif isinstance(kw.value, ast.JoinedStr):
        evaluated_context = []
        for value in kw.value.values:
            if isinstance(value, ast.FormattedValue):
                evaluated_context.append(f"{{{ast.unparse(value.value)}}}")
            elif isinstance(value, ast.Constant):
                evaluated_context.append(value.value)
        return "".join(evaluated_context)
    else:
        return clean_string(param_source)


def find_unimplemented_v2_calls(dynamo_dir):
    results = []
    dynamo_dir = Path(dynamo_dir)

    for file_path in dynamo_dir.glob("**/*.py"):
        with open(file_path) as f:
            source = f.read()
            try:
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name == "unimplemented_v2":
                            continue
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "unimplemented_v2"
                    ):
                        info = {
                            "gb_type": None,
                            "context": None,
                            "explanation": None,
                            "hints": [],
                        }

                        for kw in node.keywords:
                            if kw.arg in info:
                                info[kw.arg] = extract_info_from_keyword(source, kw)

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
        gb_id = f"GB{i:04d}"
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
        "--dynamo_dir",
        type=str,
        default=None,
        help="Directory to search for unimplemented_v2 calls.",
    )
    parser.add_argument(
        "--registry-path",
        type=str,
        default=str(Path(__file__).parent / "graph_break_registry.json"),
        help="Path to save the registry JSON file.",
    )
    args = parser.parse_args()

    if args.dynamo_dir is None:
        try:
            import torch._dynamo

            args.dynamo_dir = str(Path(torch._dynamo.__file__).parent)
        except ImportError:
            args.dynamo_dir = str(
                Path(__file__).parent.parent.parent / "torch" / "_dynamo"
            )

    create_registry(args.dynamo_dir, args.registry_path)


if __name__ == "__main__":
    main()
