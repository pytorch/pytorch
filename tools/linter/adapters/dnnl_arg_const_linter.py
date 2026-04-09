#!/usr/bin/env python3
"""
DNNL_ARG_CONST: Enforce explicit Tensor pointer intent for oneDNN execution args
constructed via make_onednn_memory in mkldnn code.

Rules:
  - Input-only oneDNN args must use const_data_ptr()
  - Output/InputOutput args must use mutable_data_ptr()
  - raw data_ptr() is rejected in these contexts

Classification is based on oneDNN argument mutability documented in
uxlfoundation/oneDNN#4843, kept as a local snapshot for deterministic linting.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple


LINTER_CODE = "DNNL_ARG_CONST"


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


class PointerKind(str, Enum):
    RAW = "data_ptr"
    CONST = "const_data_ptr"
    MUTABLE = "mutable_data_ptr"


class MemoryCreation(NamedTuple):
    lineno: int
    ptr_kind: PointerKind
    tensor_expr: str
    original_line: str


INPUT_ARGS: frozenset[str] = frozenset(
    {
        "DNNL_ARG_SRC",
        "DNNL_ARG_SRC_0",
        "DNNL_ARG_SRC_1",
        "DNNL_ARG_SRC_2",
        "DNNL_ARG_WEIGHTS",
        "DNNL_ARG_BIAS",
        "DNNL_ARG_SCALE",
        "DNNL_ARG_SHIFT",
        "DNNL_ARG_DIFF_DST",
        "DNNL_ARG_DIFF_DST_LAYER",
        "DNNL_ARG_DIFF_DST_ITER",
        "DNNL_ARG_DIFF_DST_ITER_C",
        "DNNL_ARG_FROM",
        "DNNL_ARG_ATTR_DROPOUT_PROBABILITY",
        "DNNL_ARG_ATTR_DROPOUT_SEED",
        "DNNL_ARG_ATTR_POST_OP_DW",
        "DNNL_ARG_SRC_LAYER",
        "DNNL_ARG_SRC_LAYER_ATTENTION",
        "DNNL_ARG_SRC_ITER",
        "DNNL_ARG_SRC_ITER_C",
        "DNNL_ARG_WEIGHTS_LAYER",
        "DNNL_ARG_WEIGHTS_ITER",
        "DNNL_ARG_WEIGHTS_PEEPHOLE",
        "DNNL_ARG_WEIGHTS_PROJECTION",
    }
)

INPUT_COMPOUND_PREFIXES: tuple[str, ...] = (
    "DNNL_ARG_ATTR_SCALES",
    "DNNL_ARG_ATTR_ZERO_POINTS",
    "DNNL_ARG_ATTR_MULTIPLE_POST_OP",
)


_POINTER_EXPR_RE = re.compile(
    r"""
    (?P<tensor>[A-Za-z_][A-Za-z0-9_\.\->\)\(]*)
    \.\s*(?P<kind>const_data_ptr|mutable_data_ptr|data_ptr)
    \s*\(\s*\)
    """,
    re.VERBOSE,
)

_ARG_INSERT_BRACE_RE = re.compile(
    r"""
    ^\s*\{\s*
    (?P<key>.+?)
    \s*,\s*
    (?P<value>.+?)
    \s*\}\s*$
    """,
    re.VERBOSE | re.DOTALL,
)

_ARG_SUBSCRIPT_RE = re.compile(
    r"""
    args\s*\[\s*(?P<key>[^\]]+?)\s*\]\s*=\s*(?P<value>.+)
    """,
    re.VERBOSE,
)


def _line_of(source: str, pos: int) -> int:
    return source.count("\n", 0, pos) + 1


def _find_balanced_body(source: str, open_pos: int, opener: str, closer: str) -> str | None:
    depth = 1
    i = open_pos
    limit = min(len(source), open_pos + 4096)
    while i < limit and depth > 0:
        ch = source[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
        i += 1
    if depth == 0:
        return source[open_pos : i - 1]
    return None


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    out: list[str] = []
    start = 0
    paren = 0
    brace = 0
    bracket = 0
    angle = 0

    for i, ch in enumerate(text):
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(paren - 1, 0)
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(brace - 1, 0)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(bracket - 1, 0)
        elif ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(angle - 1, 0)
        elif (
            ch == delimiter
            and paren == 0
            and brace == 0
            and bracket == 0
            and angle == 0
        ):
            out.append(text[start:i].strip())
            start = i + 1

    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _is_input_arg(key_expr: str) -> bool:
    key_tokens = re.findall(r"DNNL_ARG_\w+", key_expr)
    for token in key_tokens:
        for prefix in INPUT_COMPOUND_PREFIXES:
            if token.startswith(prefix):
                return True
    return any(token in INPUT_ARGS for token in key_tokens)


def _required_kind_for_key(key_expr: str) -> PointerKind:
    return PointerKind.CONST if _is_input_arg(key_expr) else PointerKind.MUTABLE


def _extract_pointer_kind(ptr_expr: str) -> tuple[PointerKind, str] | None:
    match = _POINTER_EXPR_RE.search(ptr_expr)
    if match is None:
        return None
    kind = PointerKind(match.group("kind"))
    tensor_expr = match.group("tensor")
    return (kind, tensor_expr)


def _pointer_replacement(line: str, required_kind: PointerKind) -> str:
    method = f"{required_kind.value}()"
    line = line.replace(".data_ptr()", f".{method}")
    line = line.replace(".const_data_ptr()", f".{method}")
    line = line.replace(".mutable_data_ptr()", f".{method}")
    return line


def _extract_varname_before_call(source: str, call_pos: int) -> str | None:
    prefix = source[max(0, call_pos - 300) : call_pos]
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*$", prefix.rstrip())
    if match is None:
        return None
    varname = match.group(1)
    if varname in {"auto", "const", "return", "memory", "dnnl"}:
        return None
    return varname


def _extract_memory_creations(source: str, lines: list[str]) -> dict[str, list[MemoryCreation]]:
    creations: dict[str, list[MemoryCreation]] = {}
    marker = "make_onednn_memory("
    scan_pos = 0

    while True:
        call_pos = source.find(marker, scan_pos)
        if call_pos < 0:
            break
        body_start = call_pos + len(marker)
        body = _find_balanced_body(source, body_start, "(", ")")
        scan_pos = body_start
        if body is None:
            continue

        args = _split_top_level(body)
        if len(args) < 3:
            continue
        pointer = _extract_pointer_kind(args[2])
        if pointer is None:
            continue
        ptr_kind, tensor_expr = pointer

        varname = _extract_varname_before_call(source, call_pos)
        if varname is None:
            continue

        lineno = _line_of(source, call_pos)
        original_line = lines[lineno - 1].rstrip()
        info = MemoryCreation(lineno, ptr_kind, tensor_expr, original_line)
        creations.setdefault(varname, []).append(info)

    return creations


def _parse_args_insert(body: str) -> tuple[str, str] | None:
    collapsed = " ".join(body.split())
    match = _ARG_INSERT_BRACE_RE.match(collapsed)
    if match is None:
        return None
    return (match.group("key").strip(), match.group("value").strip())


def _iter_arg_bindings(source: str, lines: list[str]) -> list[tuple[int, str, str]]:
    bindings: list[tuple[int, str, str]] = []

    marker = "args.insert("
    scan_pos = 0
    while True:
        call_pos = source.find(marker, scan_pos)
        if call_pos < 0:
            break
        body_start = call_pos + len(marker)
        body = _find_balanced_body(source, body_start, "(", ")")
        scan_pos = body_start
        if body is None:
            continue
        parsed = _parse_args_insert(body)
        if parsed is None:
            continue
        key_expr, value_expr = parsed
        lineno = _line_of(source, call_pos)
        bindings.append((lineno, key_expr, value_expr))

    for lineno, raw in enumerate(lines, 1):
        match = _ARG_SUBSCRIPT_RE.search(raw)
        if match is None:
            continue
        key_expr = match.group("key").strip()
        value_expr = match.group("value").strip().rstrip(";")
        bindings.append((lineno, key_expr, value_expr))

    return bindings


def _resolve_ptr_from_value(
    value_expr: str,
    binding_lineno: int,
    creations: dict[str, list[MemoryCreation]],
) -> MemoryCreation | None:
    if "make_onednn_memory(" in value_expr:
        call_open = value_expr.find("make_onednn_memory(") + len("make_onednn_memory(")
        body = _find_balanced_body(value_expr, call_open, "(", ")")
        if body is None:
            return None
        args = _split_top_level(body)
        if len(args) < 3:
            return None
        pointer = _extract_pointer_kind(args[2])
        if pointer is None:
            return None
        ptr_kind, tensor_expr = pointer
        return MemoryCreation(
            lineno=binding_lineno,
            ptr_kind=ptr_kind,
            tensor_expr=tensor_expr,
            original_line=value_expr,
        )

    best: MemoryCreation | None = None
    for varname, infos in creations.items():
        if not re.search(r"\b" + re.escape(varname) + r"\b", value_expr):
            continue
        for info in reversed(infos):
            if info.lineno <= binding_lineno:
                if best is None or info.lineno > best.lineno:
                    best = info
                break
    return best


def check_file(filename: str) -> list[LintMessage]:
    path = Path(filename)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.WARNING,
                name="io-error",
                original=None,
                replacement=None,
                description=f"Could not read file: {exc}",
            )
        ]

    lines = source.splitlines()
    creations = _extract_memory_creations(source, lines)
    bindings = _iter_arg_bindings(source, lines)

    seen: set[tuple[str, int]] = set()
    messages: list[LintMessage] = []

    for binding_lineno, key_expr, value_expr in bindings:
        required_kind = _required_kind_for_key(key_expr)
        resolved = _resolve_ptr_from_value(value_expr, binding_lineno, creations)
        if resolved is None:
            continue
        if resolved.ptr_kind == required_kind:
            continue

        dedup = (filename, resolved.lineno)
        if dedup in seen:
            continue
        seen.add(dedup)

        original = resolved.original_line
        replacement = _pointer_replacement(original, required_kind)
        expected_method = f"{required_kind.value}()"
        found_method = f"{resolved.ptr_kind.value}()"

        messages.append(
            LintMessage(
                path=filename,
                line=resolved.lineno,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.WARNING,
                name="incorrect-pointer-kind",
                original=original,
                replacement=replacement,
                description=(
                    f"{key_expr} expects `{expected_method}` by oneDNN argument mutability "
                    f"classification, but found `{found_method}`."
                ),
            )
        )

    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lint oneDNN argument pointer mutability in mkldnn C++ files.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="Paths to C++ files to lint.",
    )
    parser.add_argument(
        "--severity",
        default="warning",
        choices=("error", "warning", "advice", "disabled"),
        help="Lint severity (default: warning).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    severity = LintSeverity(args.severity)
    had_errors = False
    for filename in args.filenames:
        for msg in check_file(filename):
            had_errors = True
            print(json.dumps(msg._replace(severity=severity)._asdict()), flush=True)

    if had_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
