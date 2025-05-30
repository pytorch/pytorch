import ast
import json
import os
import re
from pathlib import Path


def get_source_segment(source, node):
    if hasattr(ast, "get_source_segment"):
        return ast.get_source_segment(source, node)
    else:
        # Fallback for older Python versions
        return source[node.lineno - 1][node.col_offset : node.end_col_offset]


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
        s = re.sub(r'" "', " ", s)
        return s
    return s


def expand_hints(hints):
    from torch._dynamo import graph_break_hints

    expanded_hints = []
    for hint in hints:
        if isinstance(hint, str):
            if "*graph_break_hints.USER_ERROR" in hint:
                expanded_hints.extend(graph_break_hints.USER_ERROR)
            elif "*graph_break_hints.DYNAMO_BUG" in hint:
                expanded_hints.extend(graph_break_hints.DYNAMO_BUG)
            elif "*graph_break_hints.DIFFICULT" in hint:
                expanded_hints.extend(graph_break_hints.DIFFICULT)
            elif "*graph_break_hints.FUNDAMENTAL" in hint:
                expanded_hints.extend(graph_break_hints.FUNDAMENTAL)
            elif "*graph_break_hints.SUPPORTABLE" in hint:
                expanded_hints.extend(graph_break_hints.SUPPORTABLE)
            elif "*graph_break_hints.CAUSED_BY_EARLIER_GRAPH_BREAK" in hint:
                expanded_hints.extend(graph_break_hints.CAUSED_BY_EARLIER_GRAPH_BREAK)
            elif "*graph_break_hints.INFERENCE_MODE" in hint:
                expanded_hints.extend(graph_break_hints.INFERENCE_MODE)
            else:
                expanded_hints.append(hint)
        else:
            expanded_hints.append(hint)

    return expanded_hints


def find_unimplemented_v2_calls():
    dynamo_dir = Path(__file__).parent.parent.parent / "torch" / "_dynamo"
    results = []

    for file_path in dynamo_dir.glob("**/*.py"):
        with open(file_path, "r") as f:
            source = f.read()
            source_lines = source.splitlines()

            try:
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and hasattr(node, "func"):
                        is_unimplemented = False
                        if (
                            isinstance(node.func, ast.Name)
                            and node.func.id == "unimplemented_v2"
                        ):
                            is_unimplemented = True
                        elif (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr == "unimplemented_v2"
                        ):
                            is_unimplemented = True

                        if is_unimplemented:
                            info = {
                                "gb_type": None,
                                "context": None,
                                "explanation": None,
                                "hints": [],
                                "from_exc": None,
                            }

                            for kw in node.keywords:
                                if kw.arg in info:
                                    param_source = get_source_segment(source, kw.value)
                                    if param_source:
                                        info[kw.arg] = clean_string(param_source)
                                    elif isinstance(kw.value, ast.Constant):
                                        info[kw.arg] = kw.value.value
                                    else:
                                        info[kw.arg] = "DYNAMIC_CONTEXT"

                            results.append((file_path, node.lineno, info))
            except SyntaxError:
                print(f"Syntax error in {file_path}")

    return results


def create_registry():
    calls = find_unimplemented_v2_calls()
    registry = {}

    gb_types = {}
    for file_path, line, info in calls:
        if info["gb_type"] and info["gb_type"] not in gb_types:
            gb_types[info["gb_type"]] = info
            gb_types[info["gb_type"]]["file_path"] = str(file_path)
            gb_types[info["gb_type"]]["line"] = line

    for i, (gb_type, info) in enumerate(sorted(gb_types.items()), 1001):
        gb_id = f"GB{i}"

        hints = info["hints"]
        if isinstance(hints, str):
            if hints.startswith("[") and hints.endswith("]"):
                try:
                    items = re.findall(r'"([^"]*)"', hints)
                    if items:
                        hints = items
                    elif "*graph_break_hints." in hints:
                        hints = expand_hints([hints])
                except:
                    pass

        has_dynamic = bool(
            re.search(
                r"{[^}]+}",
                str(gb_type),
            )
        )

        registry[gb_id] = {
            "Version": "v1.0",
            "Gb_type": clean_string(gb_type),
            "Context": clean_string(info["context"]),
            "Explanation": clean_string(info["explanation"]),
            "Hints": hints if hints else [],
            "From_exc": info["from_exc"],
            "File_path": info["file_path"],
            "Line": info["line"],
            "Has_dynamic": has_dynamic,
        }

    registry_path = Path(__file__).parent / "graph_break_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Created registry with {len(registry)} entries at {registry_path}")


if __name__ == "__main__":
    create_registry()
