#!/usr/bin/env python3
"""
GENERATED_SHIMS_VERSION: Ensures that newly added entries (or vN variants) in
torchgen/aoti/fallback_ops.py carry a "since" key set to the current
TORCH_VERSION, and that "since" values on already-shipped entries are not
modified.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools.linter.adapters._stable_shim_utils import (
    get_current_version,
    LintMessage,
    LintSeverity,
)


LINTER_CODE = "GENERATED_SHIMS_VERSION"

_TORCH_VERSION_PATTERN = re.compile(r"^TORCH_VERSION_\d+_\d+_\d+$")

# {op_name: {op_version: (since_value_or_None, lineno)}}
# op_version == 1 represents the base op (top-level "since" in the per-op dict);
# op_version >= 2 represents a "vN" variant.
ParsedOps = dict[str, dict[int, tuple[str | None, int]]]


def _const_str(node: ast.expr | None) -> str | None:
    """Return `node`'s string-literal value, or None if it isn't one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _str_value_for_key(dict_node: ast.Dict, key: str) -> str | None:
    """Return the str literal at `dict_node[key]`, or None if absent/non-literal."""
    for k_node, v_node in zip(dict_node.keys, dict_node.values):
        if _const_str(k_node) == key:
            return _const_str(v_node)
    return None


def _ingest_op_dict(dict_node: ast.Dict) -> ParsedOps:
    """Walk one of the top-level fallback dicts and return a `ParsedOps` map."""
    out: ParsedOps = {}
    for op_key_node, op_val_node in zip(dict_node.keys, dict_node.values):
        if op_key_node is None or not isinstance(op_val_node, ast.Dict):
            continue
        op_name = _const_str(op_key_node)
        if op_name is None:
            continue
        per_op: dict[int, tuple[str | None, int]] = {
            1: (_str_value_for_key(op_val_node, "since"), op_key_node.lineno),
        }
        for var_key_node, var_val_node in zip(op_val_node.keys, op_val_node.values):
            if var_key_node is None:
                continue
            var_key = _const_str(var_key_node)
            if var_key is None or not (
                var_key.startswith("v") and var_key[1:].isdigit()
            ):
                continue
            ver_id = int(var_key[1:])
            if ver_id <= 1:
                continue
            since = (
                _str_value_for_key(var_val_node, "since")
                if isinstance(var_val_node, ast.Dict)
                else None
            )
            per_op[ver_id] = (since, var_key_node.lineno)
        out[op_name] = per_op
    return out


def parse_fallback_ops(src: str) -> dict[str, ParsedOps]:
    """
    Parse a `fallback_ops.py` source string. Returns a map keyed by the
    module-level variable name (e.g. `inductor_fallback_ops`,
    `aten_shimified_ops`) so current-vs-base comparison can pair dicts by name
    rather than by position.
    """
    out: dict[str, ParsedOps] = {}
    tree = ast.parse(src)
    for node in tree.body:
        if not (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Dict)
        ):
            continue
        target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
        if isinstance(target, ast.Name):
            out[target.id] = _ingest_op_dict(node.value)
    return out


def _read_at_merge_base(filename: str) -> str | None:
    """
    Return `filename` contents at the merge-base of HEAD with origin/main, or
    None if the file did not exist at that point. Raises if git operations fail.
    """
    result = subprocess.run(
        ["git", "fetch", "origin", "main"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch origin. Error: {result.stderr.strip()}")

    result = subprocess.run(
        ["git", "merge-base", "HEAD", "origin/main"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to find merge-base with origin/main. "
            f"Error: {result.stderr.strip()}"
        )
    merge_base = result.stdout.strip()

    # `git show <ref>:<path>` requires <path> relative to the repo root;
    # lintrunner may pass `filename` as an absolute path.
    rel_path = Path(filename).resolve().relative_to(REPO_ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{merge_base}:{rel_path}"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        # File didn't exist at merge-base; treat all current entries as new.
        return None
    return result.stdout


def _msg(filename: str, line: int, name: str, description: str) -> LintMessage:
    return LintMessage(
        path=filename,
        line=line,
        char=None,
        code=LINTER_CODE,
        severity=LintSeverity.ERROR,
        name=name,
        original=None,
        replacement=None,
        description=description,
    )


def _check_one_dict(
    current: ParsedOps,
    base: ParsedOps,
    expected_since: str,
    filename: str,
) -> list[LintMessage]:
    messages: list[LintMessage] = []
    for op_name, variants in current.items():
        prev_variants = base.get(op_name, {})
        for ver_id, (since, lineno) in variants.items():
            label = f"'{op_name}'" if ver_id == 1 else f"'{op_name}' v{ver_id}"
            prev = prev_variants.get(ver_id)

            if prev is None:
                # Newly added op (ver_id == 1) or newly added vN variant.
                if since is None:
                    messages.append(
                        _msg(
                            filename,
                            lineno,
                            "missing-version-on-new-op",
                            f"{label} is newly added; it must include a "
                            f"'since' key set to {expected_since}.",
                        )
                    )
                elif not _TORCH_VERSION_PATTERN.match(since):
                    messages.append(
                        _msg(
                            filename,
                            lineno,
                            "malformed-version",
                            f"{label} 'since' value {since!r} does not match "
                            f"the format TORCH_VERSION_X_Y_Z.",
                        )
                    )
                elif since != expected_since:
                    messages.append(
                        _msg(
                            filename,
                            lineno,
                            "wrong-version-on-new-op",
                            f"{label} 'since' should be {expected_since} "
                            f"(the current TORCH_VERSION); got {since}.",
                        )
                    )
            else:
                prev_since, _ = prev
                if since != prev_since:
                    messages.append(
                        _msg(
                            filename,
                            lineno,
                            "op-version-modified",
                            f"{label} 'since' changed from {prev_since!r} "
                            f"to {since!r}; once an op has shipped its "
                            f"'since' version value must not be modified.",
                        )
                    )

    return messages


def check_file(filename: str) -> list[LintMessage]:
    current_dicts = parse_fallback_ops(Path(filename).read_text())
    base_src = _read_at_merge_base(filename)
    base_dicts = parse_fallback_ops(base_src) if base_src is not None else {}

    major, minor, patch = get_current_version()
    expected_since = f"TORCH_VERSION_{major}_{minor}_{patch}"

    messages: list[LintMessage] = []
    for name, current in current_dicts.items():
        base = base_dicts.get(name, {})
        messages.extend(_check_one_dict(current, base, expected_since, filename))
    return messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generated shims version linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("filenames", nargs="+", help="paths to lint")
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
    )

    lint_messages: list[LintMessage] = []
    for fn in args.filenames:
        lint_messages.extend(check_file(fn))
    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)
