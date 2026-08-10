import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import Any


def get_source_segment(source: str, node: ast.AST) -> str | None:
    return ast.get_source_segment(source, node)


def load_registry(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open() as f:
            return json.load(f)  # type: ignore[no-any-return]
    return {}


def save_registry(reg: dict[str, Any], path: Path) -> None:
    with path.open("w") as f:
        json.dump(reg, f, indent=2)


def next_gb_id(reg: dict[str, Any]) -> str:
    """Generate a random unused GB ID from GB0000-GB9999 range."""
    used_ids = set(reg.keys())
    max_attempts = 100

    # Try random selection first
    for _ in range(max_attempts):
        candidate = f"GB{random.randint(0, 9999):04d}"
        if candidate not in used_ids:
            return candidate

    # Fallback: find first available ID if random selection keeps colliding
    for i in range(10000):
        candidate = f"GB{i:04d}"
        if candidate not in used_ids:
            return candidate

    raise RuntimeError("No available GB IDs in range GB0000-GB9999")


def clean_string(s: Any) -> Any:
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
        # Replace any whitespace
        s = " ".join(s.splitlines())
        # Replace escaped quotes with their unescaped versions
        s = s.encode().decode("unicode_escape")
        # Replace adjacent quoted strings with a space (e.g., " "" -> " ")
        s = re.sub(r'" "', " ", s)
    return s


def expand_hints(hints: list[str], dynamo_dir: str | None = None) -> list[str]:
    """
    Expands hint references to their actual values from graph_break_hints.
    Uses exec() to avoid import dependencies.
    """
    if dynamo_dir is None:
        script_dir = Path(__file__).resolve().parent
        dynamo_dir_path = script_dir.parent.parent / "torch" / "_dynamo"
    else:
        dynamo_dir_path = Path(dynamo_dir)

    graph_break_hints_path = dynamo_dir_path / "graph_break_hints.py"

    with open(graph_break_hints_path) as f:
        hints_source = f.read()

    hints_namespace: dict[str, Any] = {}
    exec(hints_source, hints_namespace)

    hint_constants = {
        name: value
        for name, value in hints_namespace.items()
        if isinstance(value, list) and name.isupper() and not name.startswith("_")
    }

    expanded_hints = []
    for hint in hints:
        expanded = False
        for name, value in hint_constants.items():
            if f"*graph_break_hints.{name}" in hint:
                expanded_hints.extend(value)
                expanded = True
                break
        if not expanded:
            expanded_hints.append(hint)

    return expanded_hints


def extract_info_from_value(
    source: str, value_node: ast.AST, substitutions: dict[str, str] | None = None
) -> Any:
    substitutions = substitutions or {}

    if isinstance(value_node, ast.Constant):
        return value_node.value
    elif isinstance(value_node, ast.JoinedStr):
        evaluated_context = []
        for value in value_node.values:
            if isinstance(value, ast.FormattedValue):
                if (
                    isinstance(value.value, ast.Name)
                    and value.value.id in substitutions
                ):
                    evaluated_context.append(substitutions[value.value.id])
                else:
                    evaluated_context.append(f"{{{ast.unparse(value.value)}}}")
            elif isinstance(value, ast.Constant):
                # pyrefly: ignore [bad-argument-type]
                evaluated_context.append(value.value)
        return "".join(evaluated_context)
    else:
        # Only call get_source_segment when actually needed (avoids expensive
        # _splitlines_no_ff call for every keyword argument)
        param_source = get_source_segment(source, value_node)
        return clean_string(param_source)


def extract_info_from_keyword(
    source: str, kw: ast.keyword, substitutions: dict[str, str] | None = None
) -> Any:
    """
    Extracts and returns the value of a keyword argument from an AST node.

    This function handles different types of AST nodes:
    - If the node is a constant, it returns the constant value.
    - If the node is an f-string, it reconstructs the string by
      evaluating formatted values and concatenating them with string literals.
    - For other types, it cleans the source segment to remove formatting artifacts.

    """
    return extract_info_from_value(source, kw.value, substitutions)


def extract_hint_list(
    source: str,
    value_node: ast.AST,
    substitutions: dict[str, str],
    dynamo_dir: str | None,
) -> list[str]:
    if not isinstance(value_node, ast.List):
        hints = extract_info_from_value(source, value_node, substitutions)
        if not isinstance(hints, str):
            return []

        expanded_hints = []
        items = re.findall(r'"([^"]*)"', hints)
        if items:
            expanded_hints.extend(items)
        if "*graph_break_hints." in hints:
            expanded_hints.extend(expand_hints([hints], dynamo_dir))
        return expanded_hints

    expanded_hints = []
    for elt in value_node.elts:
        if isinstance(elt, ast.Starred):
            hint_source = get_source_segment(source, elt) or ""
            if "*graph_break_hints." in hint_source:
                expanded_hints.extend(expand_hints([hint_source], dynamo_dir))
            continue

        hint = extract_info_from_value(source, elt, substitutions)
        if isinstance(hint, str):
            expanded_hints.append(hint)

    return expanded_hints


def extract_constant_str_arg(node: ast.Call, name: str, index: int) -> str | None:
    if len(node.args) > index:
        arg = node.args[index]
        if isinstance(arg, ast.Constant):
            value = arg.value
            if isinstance(value, str):
                return value

    for kw in node.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant):
            value = kw.value.value
            if isinstance(value, str):
                return value

    return None


def find_helper_unimplemented_call(helper: ast.FunctionDef) -> ast.Call | None:
    for node in ast.walk(helper):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("unimplemented", "_unimplemented")
        ):
            return node

    return None


def extract_call_info(
    source: str,
    node: ast.Call,
    substitutions: dict[str, str],
    dynamo_dir: str | None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "gb_type": None,
        "context": None,
        "explanation": None,
        "hints": [],
    }

    for kw in node.keywords:
        if kw.arg == "hints":
            info["hints"] = extract_hint_list(
                source, kw.value, substitutions, dynamo_dir
            )
        elif kw.arg in info:
            info[kw.arg] = extract_info_from_keyword(source, kw, substitutions)

    return info


def find_unimplemented_calls(
    path: str, dynamo_dir: str | None = None
) -> list[dict[str, Any]]:
    results = []
    path_obj = Path(path)

    if path_obj.is_dir():
        file_paths = path_obj.glob("**/*.py")
    else:
        file_paths = [path_obj]  # type: ignore[assignment]

    for file_path in file_paths:
        with open(file_path) as f:
            source = f.read()
            try:
                tree = ast.parse(source)
                helper_calls = {
                    node.name: helper_call
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "unimplemented_direct_disable_call"
                    and (helper_call := find_helper_unimplemented_call(node))
                    is not None
                }

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name in (
                            "unimplemented",
                            "unimplemented_with_warning",
                        ):
                            continue
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id
                        in ("unimplemented", "unimplemented_with_warning")
                    ):
                        info: dict[str, Any] = {
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
                                expanded_hints.extend(expand_hints([hints], dynamo_dir))

                            info["hints"] = expanded_hints

                        results.append(info)
                    elif (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in helper_calls
                    ):
                        api_name = extract_constant_str_arg(node, "api_name", 0)
                        if api_name is not None:
                            info = extract_call_info(
                                source,
                                helper_calls[node.func.id],
                                {"api_name": api_name},
                                dynamo_dir,
                            )
                            if info["gb_type"] is not None:
                                results.append(info)
            except SyntaxError:
                print(f"Syntax error in {file_path}")

    return results


def create_registry(dynamo_dir: str, registry_path: str) -> None:
    calls = find_unimplemented_calls(dynamo_dir)
    registry = {}

    gb_types = {}
    for info in calls:
        gb_types[info["gb_type"]] = info

    # Use sequential IDs for initial registry creation
    GB_ID_INDEX = 0
    for i, (gb_type, info) in enumerate(sorted(gb_types.items()), GB_ID_INDEX):
        gb_id = f"GB{i:04d}"
        hints = info["hints"]

        registry[gb_id] = [
            {
                "Gb_type": gb_type,
                "Context": info["context"],
                "Explanation": info["explanation"],
                "Hints": hints if hints else [],
            }
        ]

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    registry_path = repo_root / "torch" / "_dynamo" / "graph_break_registry.json"

    try:
        import torch._dynamo

        default_dynamo_dir = str(Path(torch._dynamo.__file__).parent)
    except ImportError:
        default_dynamo_dir = str(repo_root / "torch" / "_dynamo")

    parser = argparse.ArgumentParser(description="Manage graph break registry.")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    create_parser = subparsers.add_parser("create", help="Create registry from scratch")
    create_parser.add_argument(
        "--dynamo_dir",
        type=str,
        default=default_dynamo_dir,
        help="Directory to search for unimplemented calls.",
    )

    parser.add_argument(
        "--registry-path",
        type=str,
        default=str(registry_path),
        help="Path to save the registry JSON file",
    )

    args = parser.parse_args()

    if args.command == "create":
        create_registry(args.dynamo_dir, args.registry_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
