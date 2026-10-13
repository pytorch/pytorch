#!/usr/bin/env python3
"""Ban named VariableTracker construction outside variables/builder.py.

Discover VTs from their inheritance in torch/_dynamo/variables, without importing
torch. Resolve ordinary imports and qualified names in their scopes, but do not
infer types or follow assignment aliases. Global/nonlocal name redirects are not
modeled, and repeated imports in one scope are not flow-tracked. Calls to factory
methods and dynamic callees are outside this rule's scope. An exact VT may also
construct itself in its own static or class method named ``create``.

An exact class-definition ``# noqa: RAW_VT_CONSTRUCTION`` permits that class,
without permitting subclasses. Individual calls may use the same directive on
the callee line, closing line, or immediately preceding standalone comment.
On a function definition line or immediately preceding standalone comment, the
directive exempts its body, excluding nested scopes, signatures, and decorators.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "RAW_VT_CONSTRUCTION"
_VARIABLES_MODULE = "torch._dynamo.variables"
_VARIABLES_DIR = Path(__file__).resolve().parents[3] / "torch/_dynamo/variables"
_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSION_SCOPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SCOPES = (ast.Module, ast.ClassDef, *_FUNCTION_SCOPES, *_COMPREHENSION_SCOPES)


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: str
    name: str
    original: str | None
    replacement: str | None
    description: str | None


@dataclass(frozen=True)
class VariableTrackerInventory:
    variables_dir: Path
    classes: frozenset[str]
    allowed: frozenset[str]
    aliases: dict[str, str]


def _module_name(filename: Path, variables_dir: Path) -> str:
    filename = filename.resolve()
    if filename.is_relative_to(variables_dir):
        parts = [
            _VARIABLES_MODULE,
            *filename.relative_to(variables_dir).with_suffix("").parts,
        ]
    else:
        parts = list(filename.with_suffix("").parts)
        parts = parts[parts.index("torch") :] if "torch" in parts else [filename.stem]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _imports(tree: ast.AST, module: str, is_package: bool) -> dict[str, str | None]:
    bindings: dict[str, str | None] = {}
    package = module if is_package else module.rpartition(".")[0]
    nodes = [tree]
    while nodes:
        node = nodes.pop()
        if node is not tree and isinstance(node, _SCOPES):
            if isinstance(node, ast.ClassDef):
                bindings[node.name] = f"{module}.{node.name}"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bindings[node.name] = None
            continue
        nodes.extend(reversed(list(ast.iter_child_nodes(node))))
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.partition(".")[0]
                bindings[name] = alias.name if alias.asname else name
        elif isinstance(node, ast.ImportFrom):
            target = node.module or ""
            if node.level:
                if not package:
                    continue
                try:
                    target = importlib.util.resolve_name(
                        "." * node.level + target, package
                    )
                except ImportError:
                    continue
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = f"{target}.{alias.name}"
        elif isinstance(node, ast.arg):
            bindings[node.arg] = None
        elif isinstance(node, ast.comprehension):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                    bindings[target.id] = None
    return bindings


def _resolve(
    node: ast.expr,
    parents: dict[ast.AST, ast.AST],
    scopes: dict[ast.AST, dict[str, str | None]],
) -> str | None:
    name = _dotted_name(node)
    if name is None:
        return None
    root, separator, tail = name.partition(".")
    child: ast.AST = node
    ancestors = set()
    skip_classes = False
    while child in parents:
        ancestors.add(child)
        parent = parents[child]
        if (
            isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and child not in parent.body
        ) or (isinstance(parent, ast.Lambda) and child is not parent.body):
            child = parent
            continue
        if isinstance(parent, _COMPREHENSION_SCOPES):
            if parent.generators[0].iter in ancestors:
                child = parent
                continue
        if parent in scopes:
            if not isinstance(parent, ast.ClassDef) or not skip_classes:
                if root in scopes[parent]:
                    target = scopes[parent][root]
                    return target + separator + tail if target is not None else None
            # Class namespaces are not closures for nested scopes.
            skip_classes = True
        child = parent
    return name


def _canonical(name: str | None, aliases: dict[str, str]) -> str | None:
    seen = set()
    while name is not None and name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    return name


def _noqa_comments(source: str) -> dict[int, tokenize.TokenInfo]:
    comments = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        match = re.match(r"#\s*noqa:\s*([A-Za-z0-9_, \t]+)", token.string)
        if match and LINTER_CODE in re.split(r"[, \t]+", match[1]):
            comments[token.start[0]] = token
    return comments


def _definition_noqa_line(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    comments: dict[int, tokenize.TokenInfo],
    lines: list[str],
) -> int | None:
    if node.lineno in comments:
        return node.lineno
    previous = comments.get(node.lineno - 1)
    if (
        previous is not None
        and not lines[previous.start[0] - 1][: previous.start[1]].strip()
    ):
        return previous.start[0]
    return None


def _discover_variable_tracker_classes(
    variables_dir: Path = _VARIABLES_DIR,
) -> VariableTrackerInventory:
    variables_dir = variables_dir.resolve()
    definitions = {}
    aliases = {}
    for filename in sorted(variables_dir.rglob("*.py")):
        source = filename.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filename))
        module = _module_name(filename, variables_dir)
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        scopes: dict[ast.AST, dict[str, str | None]] = {
            node: _imports(node, module, filename.stem == "__init__")
            for node in ast.walk(tree)
            if isinstance(node, _SCOPES)
        }
        for name, target in scopes[tree].items():
            qualified = f"{module}.{name}"
            if target is not None and qualified != target:
                aliases[qualified] = target
        comments = _noqa_comments(source)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [
                    base.value if isinstance(base, ast.Subscript) else base
                    for base in node.bases
                ]
                definitions[f"{module}.{node.name}"] = (
                    tuple(_resolve(base, parents, scopes) for base in bases),
                    _definition_noqa_line(node, comments, lines) is not None,
                )
    root = f"{_VARIABLES_MODULE}.base.VariableTracker"
    classes = {root} if root in definitions else set()
    while True:
        previous_count = len(classes)
        for name, (bases, _) in definitions.items():
            if any(_canonical(base, aliases) in classes for base in bases):
                classes.add(name)
        if len(classes) == previous_count:
            break
    allowed = {name for name in classes if definitions[name][1]}
    return VariableTrackerInventory(
        variables_dir, frozenset(classes), frozenset(allowed), aliases
    )


def _call_noqa(
    node: ast.Call, comments: dict[int, tokenize.TokenInfo], lines: list[str]
) -> bool:
    if node.func.lineno in comments or node.end_lineno in comments:
        return True
    previous = comments.get(node.func.lineno - 1)
    return (
        previous is not None
        and not lines[previous.start[0] - 1][: previous.start[1]].strip()
    )


def _syntax_error(error: SyntaxError) -> LintMessage:
    return LintMessage(
        error.filename,
        error.lineno,
        error.offset,
        LINTER_CODE,
        "error",
        "syntax-error",
        None,
        None,
        f"Failed to parse file: {error}",
    )


def _function_exempt(
    node: ast.Call,
    target: str | None,
    module: str,
    parents: dict[ast.AST, ast.AST],
    comments: dict[int, tokenize.TokenInfo],
    lines: list[str],
) -> bool:
    child: ast.AST = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, (ast.ClassDef, ast.Lambda)):
            return False
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if child not in parent.body:
                return False
            if _definition_noqa_line(parent, comments, lines) is not None:
                return True
            cls = parents.get(parent)
            return (
                isinstance(cls, ast.ClassDef)
                and target == f"{module}.{cls.name}"
                and parent.name == "create"
                and any(
                    isinstance(decorator, ast.Name)
                    and decorator.id in ("classmethod", "staticmethod")
                    for decorator in parent.decorator_list
                )
            )
        child = parent
    return False


def check_file(
    filename: str, inventory: VariableTrackerInventory | None = None
) -> list[LintMessage]:
    inventory = inventory or _discover_variable_tracker_classes()
    path = Path(filename).resolve()
    if path == inventory.variables_dir / "builder.py":
        return []
    source = path.read_text(encoding="utf-8")
    if not path.is_relative_to(inventory.variables_dir.parent) and (
        "_dynamo" not in source or "variables" not in source
    ):
        return []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        return [_syntax_error(error)]
    module = _module_name(path, inventory.variables_dir)
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    scopes: dict[ast.AST, dict[str, str | None]] = {
        node: _imports(node, module, path.stem == "__init__")
        for node in ast.walk(tree)
        if isinstance(node, _SCOPES)
    }
    comments = _noqa_comments(source)
    lines = source.splitlines()
    function_noqas = {
        line
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if (line := _definition_noqa_line(node, comments, lines)) is not None
    }
    call_comments = {
        line: token for line, token in comments.items() if line not in function_noqas
    }
    messages = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _canonical(_resolve(node.func, parents, scopes), inventory.aliases)
        if target not in inventory.classes or target in inventory.allowed:
            continue
        if _call_noqa(node, call_comments, lines) or _function_exempt(
            node, target, module, parents, comments, lines
        ):
            continue
        messages.append(
            LintMessage(
                filename,
                node.func.lineno,
                node.func.col_offset + 1,
                LINTER_CODE,
                "error",
                "raw-variable-tracker-construction",
                None,
                None,
                "Use VariableTracker.build or a VT factory, or exempt this call, "
                "function, or exact VT class with '# noqa: RAW_VT_CONSTRUCTION'.",
            )
        )
    return sorted(messages, key=lambda message: (message.line or 0, message.char or 0))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    parser.add_argument("filenames", nargs="+", help="Python files to check")
    args = parser.parse_args(argv)
    try:
        inventory = _discover_variable_tracker_classes()
    except SyntaxError as error:
        print(json.dumps(_syntax_error(error)._asdict()), flush=True)
        return
    for filename in args.filenames:
        for message in check_file(filename, inventory):
            print(json.dumps(message._asdict()), flush=True)


if __name__ == "__main__":
    main()
