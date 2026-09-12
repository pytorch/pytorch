#!/usr/bin/env python3
"""Build a libdevice control with selected functions restored from an older copy."""

import argparse
import hashlib
import re
from pathlib import Path

from llvmlite import binding as llvm


VARIANT_FUNCTIONS = {
    "old_erff": ("__nv_erff",),
    "old_powf": ("__nv_powf",),
    "old_erff_powf": ("__nv_erff", "__nv_powf"),
}


def module_text(path: Path) -> str:
    # Separate contexts keep identified struct names stable across the two modules.
    context = llvm.create_context()
    return str(llvm.parse_bitcode(path.read_bytes(), context=context))


def find_definition(module: str, function: str) -> re.Match[str]:
    pattern = re.compile(
        rf"^define\b[^\n]*@{re.escape(function)}\([^\n]*\)[^\n]*\{{\n.*?^\}}",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(module)
    if match is None:
        raise RuntimeError(f"Could not find a definition for {function}")
    if pattern.search(module, match.end()) is not None:
        raise RuntimeError(f"Found multiple definitions for {function}")
    return match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANT_FUNCTIONS, required=True)
    args = parser.parse_args()

    current = module_text(args.current)
    old = module_text(args.old)
    for function in VARIANT_FUNCTIONS[args.variant]:
        old_definition = find_definition(old, function).group(0)
        current_match = find_definition(current, function)
        current = (
            current[: current_match.start()]
            + old_definition
            + current[current_match.end() :]
        )

    # Emit text rather than LLVM 22 bitcode so Triton's own LLVM parser owns the
    # final serialization format. parse_assembly verifies the constructed module.
    context = llvm.create_context()
    llvm.parse_assembly(current, context=context).verify()
    args.output.write_text(current)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    functions = ", ".join(VARIANT_FUNCTIONS[args.variant])
    print(f"Wrote {args.output} ({digest}); restored {functions}")


if __name__ == "__main__":
    main()
