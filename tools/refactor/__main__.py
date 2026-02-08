"""
Entry point for python -m tools.refactor (reserved for future use)
"""

import sys

if __name__ == "__main__":
    print("tools.refactor package - no default entry point", file=sys.stderr)
    print("Available modules:", file=sys.stderr)
    print("  - import_smoke_static: Static import graph checker", file=sys.stderr)
    sys.exit(1)


