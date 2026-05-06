"""Wrap installed headers with TORCH_STABLE_ONLY / TORCH_TARGET_VERSION guards.

Headers under the include directory are wrapped so they emit a compile error
when included with TORCH_STABLE_ONLY or TORCH_TARGET_VERSION defined.
Certain directories (stable API, headeronly, AOTI shims) are excluded.

Called at install time by cmake/PostBuildSteps.cmake.
"""

import argparse
import pathlib


HEADER_EXTENSIONS = (".h", ".hpp", ".cuh")

EXCLUDE_PATTERNS = (
    "torch/headeronly/",
    "torch/csrc/stable/",
    "torch/csrc/inductor/aoti_torch/c/",
    "torch/csrc/inductor/aoti_torch/generated/",
)

WRAP_MARKER = "#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "include_dir", type=pathlib.Path, help="Installed include directory"
    )
    args = parser.parse_args()

    include_dir = args.include_dir
    if not include_dir.exists():
        return

    for header in sorted(include_dir.rglob("*")):
        if header.suffix not in HEADER_EXTENSIONS:
            continue

        rel = header.relative_to(include_dir).as_posix()
        if any(rel.startswith(pat) for pat in EXCLUDE_PATTERNS):
            continue

        content = header.read_text(encoding="utf-8")
        if content.startswith(WRAP_MARKER):
            continue

        wrapped = (
            f"{WRAP_MARKER}\n"
            f"{content}\n"
            "#else\n"
            '#error "This file should not be included when either '
            'TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined."\n'
            "#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)\n"
        )
        header.write_text(wrapped, encoding="utf-8")


if __name__ == "__main__":
    main()
