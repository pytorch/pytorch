"""Emit a minimal selected_operators.yaml from an inline operator list.

Replaces a shell-based genrule that produced YAML which some interpreters
parsed as a bare scalar (e.g. ``false`` on Windows cmd.exe when ``echo off``
expanded) rather than a mapping. Writing the file from Python with a fixed
``\\n`` newline guarantees a proper YAML dict regardless of shell.
"""

from __future__ import annotations

import argparse


_HEADER = (
    "include_all_non_op_selectives: false\n"
    "include_all_operators: false\n"
    "operators:\n"
)


def _render(oplist: list[str]) -> str:
    lines = [_HEADER]
    for op in oplist:
        lines.append(f"  {op}:\n")
        lines.append("    is_used_for_training: false\n")
        lines.append("    include_all_overloads: true\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("ops", nargs="*")
    args = parser.parse_args()

    with open(args.output, "w", newline="\n") as f:
        f.write(_render(args.ops))


if __name__ == "__main__":
    main()  # pragma: no cover
