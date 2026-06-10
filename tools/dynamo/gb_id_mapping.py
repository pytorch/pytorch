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


def _find_enclosing_function(
    tree: ast.Module, target_lineno: int
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Find the function/method whose body contains the given line number."""
    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if hasattr(node, "end_lineno") and node.end_lineno is not None:
                if node.lineno <= target_lineno <= node.end_lineno:
                    if best is None or node.lineno > best.lineno:
                        best = node
    return best


def _extract_string_from_node(node: ast.expr) -> str | None:
    """Extract a string literal from an ast.Constant or ast.JoinedStr node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append(f"{{{ast.unparse(value.value)}}}")
        return "".join(parts)
    return None


def _extract_dynamic_hints(
    tree: ast.Module,
    call_node: ast.Call,
    var_name: str,
    dynamo_dir: str | None = None,
) -> tuple[list[str], bool]:
    """Extract hints from dynamic variable assignments/appends in the enclosing function.

    Returns (extracted_hints, is_dynamic).
    """
    func = _find_enclosing_function(tree, call_node.lineno)
    if func is None:
        return [], True

    hints: list[str] = []
    found_any = False

    for stmt in ast.walk(func):
        # var = [str1, str2, ...]
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == var_name
            and isinstance(stmt.value, ast.List)
        ):
            found_any = True
            for elt in stmt.value.elts:
                s = _extract_string_from_node(elt)
                if s is not None:
                    hints.append(s)

        # var.append(str)
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == var_name
            and stmt.value.func.attr == "append"
            and len(stmt.value.args) == 1
        ):
            found_any = True
            s = _extract_string_from_node(stmt.value.args[0])
            if s is not None:
                hints.append(s)

        # var += [str, ...]
        if (
            isinstance(stmt, ast.AugAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == var_name
            and isinstance(stmt.value, ast.List)
        ):
            found_any = True
            for elt in stmt.value.elts:
                s = _extract_string_from_node(elt)
                if s is not None:
                    hints.append(s)

        # var.extend(graph_break_hints.X) or var.extend([...])
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == var_name
            and stmt.value.func.attr == "extend"
            and len(stmt.value.args) == 1
        ):
            found_any = True
            arg = stmt.value.args[0]
            if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
                if arg.value.id == "graph_break_hints":
                    ref = f"*graph_break_hints.{arg.attr}"
                    hints.extend(expand_hints([ref], dynamo_dir))
            elif isinstance(arg, ast.List):
                for elt in arg.elts:
                    s = _extract_string_from_node(elt)
                    if s is not None:
                        hints.append(s)

    return hints, found_any


def extract_info_from_keyword(source: str, kw: ast.keyword) -> Any:
    """
    Extracts and returns the value of a keyword argument from an AST node.

    This function handles different types of AST nodes:
    - If the node is a constant, it returns the constant value.
    - If the node is an f-string, it reconstructs the string by
      evaluating formatted values and concatenating them with string literals.
    - For other types, it cleans the source segment to remove formatting artifacts.

    """
    if isinstance(kw.value, ast.Constant):
        return kw.value.value
    elif isinstance(kw.value, ast.JoinedStr):
        evaluated_context = []
        for value in kw.value.values:
            if isinstance(value, ast.FormattedValue):
                evaluated_context.append(f"{{{ast.unparse(value.value)}}}")
            elif isinstance(value, ast.Constant):
                # pyrefly: ignore [bad-argument-type]
                evaluated_context.append(value.value)
        return "".join(evaluated_context)
    else:
        # Only call get_source_segment when actually needed (avoids expensive
        # _splitlines_no_ff call for every keyword argument)
        param_source = get_source_segment(source, kw.value)
        return clean_string(param_source)


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

                        # Check if hints was passed as a variable name (dynamic hints)
                        hints_kw = None
                        for kw in node.keywords:
                            if kw.arg == "hints":
                                hints_kw = kw
                                break

                        if (
                            hints_kw is not None
                            and isinstance(hints_kw.value, ast.Name)
                        ):
                            extracted, is_dynamic = _extract_dynamic_hints(
                                tree,
                                node,
                                hints_kw.value.id,
                                dynamo_dir,
                            )
                            info["hints"] = extracted
                            info["hints_are_dynamic"] = is_dynamic
                        elif info["hints"]:
                            hints = info["hints"]
                            expanded_hints = []
                            items = re.findall(r'"([^"]*)"', hints)
                            if items:
                                expanded_hints.extend(items)

                            if "*graph_break_hints." in hints:
                                expanded_hints.extend(expand_hints([hints], dynamo_dir))

                            info["hints"] = expanded_hints

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
                **({"Hints_dynamic": True} if info.get("hints_are_dynamic") else {}),
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
