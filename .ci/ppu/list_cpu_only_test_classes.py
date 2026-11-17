#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path


EXTRA_CPU_ONLY_CLASSES = (
    "SDPAPatternRewriterCpuDynamicTests",
    "TestSDPACpuOnly",
    "TestAutocastCPU",
)

COPY_TESTS_RE = re.compile(
    r"""copy_tests\(\s*[\w.]+\s*,\s*(\w+)\s*,\s*['"]cpu['"]""", re.S
)
CONVENTION_RE = re.compile(
    r"^\s*class\s+(\w*(?:(?:Cpu|CPU)Test|Test(?:Cpu|CPU))\w*)\s*[(:]", re.M
)
ANY_CLASS_RE = re.compile(r"^\s*class\s+(\w+)\s*[(:]", re.M)


def collect(test_dir: Path) -> tuple[set[str], set[str]]:
    """Return (CPU-only class names, all class names under test/)."""
    cpu_only: set[str] = set()
    declared: set[str] = set()
    for path in sorted(test_dir.rglob("*.py")):
        # errors="ignore": test/ has a few intentionally non-UTF-8 / binary fixtures
        src = path.read_text(encoding="utf-8", errors="ignore")
        cpu_only.update(COPY_TESTS_RE.findall(src))
        cpu_only.update(CONVENTION_RE.findall(src))
        declared.update(ANY_CLASS_RE.findall(src))
    return cpu_only, declared


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <test dir>", file=sys.stderr)
        return 2

    test_dir = Path(argv[1])
    if not test_dir.is_dir():
        print(f"[cuda-only] Directory not found: {test_dir}", file=sys.stderr)
        return 1

    cpu_only, declared = collect(test_dir)

    stale = sorted(name for name in EXTRA_CPU_ONLY_CLASSES if name not in declared)
    if stale:
        print(
            "[cuda-only] EXTRA_CPU_ONLY_CLASSES is stale, please update "
            ".ci/ppu/list_cpu_only_test_classes.py:\n"
            "  these classes no longer exist under test/ (renamed or removed upstream): " + " ".join(stale),
            file=sys.stderr,
        )
        return 1
    cpu_only.update(EXTRA_CPU_ONLY_CLASSES)

    if not cpu_only:
        print(
            '[cuda-only] No CPU-only test classes found under test/: copy_tests(..., "cpu") '
            "and the *CpuTest* naming convention may both have changed upstream;\n"
            "  please update .ci/ppu/list_cpu_only_test_classes.py",
            file=sys.stderr,
        )
        return 1

    minimal = sorted(
        name
        for name in cpu_only
        if not any(other != name and other in name for other in cpu_only)
    )
    print(" and ".join(f"not {name}" for name in minimal))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
