#!/usr/bin/env python3
"""
Check for moved/deleted doc files and optionally auto-update redirects.py.

This script detects when documentation files in docs/source/ are moved or deleted
and verifies that corresponding redirects exist in docs/source/redirects.py.

Usage:
    # Check only (CI mode) - reports missing redirects
    python check_doc_redirects.py --base-ref origin/main

    # Auto-update redirects.py with missing entries
    python check_doc_redirects.py --base-ref origin/main --auto-fix
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path


def run_git(args: list[str]) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(["git"] + args, capture_output=True, text=True)
    return result.stdout.strip()


def get_doc_changes(base_ref: str) -> list[tuple[str, str, str | None]]:
    """
    Get moved/deleted doc files between base_ref and HEAD.

    Returns:
        List of (status, old_path, new_path) tuples.
        For deletions, new_path is None.
    """
    diff = run_git(
        [
            "diff",
            "--name-status",
            "-M",  # Enable rename detection
            f"{base_ref}...HEAD",
            "--",
            "docs/source/*.rst",
            "docs/source/*.md",
        ]
    )

    changes = []
    for line in diff.split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]

        # Renames: R100 (100% similar), R095 (95% similar), etc.
        if status.startswith("R") and len(parts) >= 3:
            changes.append((status, parts[1], parts[2]))
        # Deletions
        elif status == "D" and len(parts) >= 2:
            changes.append((status, parts[1], None))

    return changes


def path_to_key(path: str) -> str:
    """
    Convert a file path to a redirect key.

    Example: docs/source/torch.compiler.rst -> torch.compiler
    Example: docs/source/user_guide/foo.rst -> user_guide/foo
    """
    return re.sub(r"\.(rst|md)$", "", path.replace("docs/source/", ""))


def path_to_url(path: str) -> str:
    """
    Convert a file path to an HTML URL for redirects.

    Example: docs/source/user_guide/foo.rst -> user_guide/foo.html
    """
    return re.sub(r"\.(rst|md)$", ".html", path.replace("docs/source/", ""))


def parse_existing_redirects(redirects_file: Path) -> dict[str, str]:
    """
    Parse redirects.py and return the existing redirects dictionary.

    Uses AST parsing for robustness.
    """
    content = redirects_file.read_text()
    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "redirects":
                    if isinstance(node.value, ast.Dict):
                        return {
                            k.value: v.value
                            for k, v in zip(node.value.keys, node.value.values)
                            if isinstance(k, ast.Constant)
                            and isinstance(v, ast.Constant)
                        }
    return {}


def find_missing_redirects(
    changes: list[tuple[str, str, str | None]],
    existing: dict[str, str],
) -> list[tuple[str, str | None]]:
    """
    Find file changes that don't have corresponding redirects.

    Returns:
        List of (old_key, new_url) tuples. new_url is None for deletions.
    """
    missing = []
    for status, old_path, new_path in changes:
        old_key = path_to_key(old_path)
        if old_key not in existing:
            new_url = path_to_url(new_path) if new_path else None
            missing.append((old_key, new_url))
    return missing


def update_redirects_file(
    redirects_file: Path, new_entries: list[tuple[str, str]]
) -> None:
    """
    Add new redirect entries to redirects.py.

    Inserts entries just before the closing brace of the redirects dict.
    """
    content = redirects_file.read_text()
    lines = content.split("\n")

    # Find the line with the closing brace
    insert_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "}":
            insert_idx = i
            break

    if insert_idx is None:
        print("Error: Could not find closing brace in redirects.py", file=sys.stderr)
        sys.exit(1)

    # Generate new entries
    new_lines = ["    # Auto-generated redirects for moved files"]
    for old_key, new_url in new_entries:
        new_lines.append(f'    "{old_key}": "{new_url}",')

    # Insert the new entries before the closing brace
    lines = lines[:insert_idx] + new_lines + lines[insert_idx:]

    redirects_file.write_text("\n".join(lines))
    print(f"✅ Added {len(new_entries)} redirect(s) to {redirects_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check for missing doc redirects and optionally auto-fix"
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Base git ref to compare against (default: origin/main)",
    )
    parser.add_argument(
        "--redirects-file",
        default="docs/source/redirects.py",
        help="Path to redirects.py (default: docs/source/redirects.py)",
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically add missing redirects for moved files",
    )
    args = parser.parse_args()

    redirects_file = Path(args.redirects_file)
    if not redirects_file.exists():
        print(f"Error: {redirects_file} not found", file=sys.stderr)
        sys.exit(1)

    # Get file changes
    changes = get_doc_changes(args.base_ref)
    if not changes:
        print("✅ No doc files were moved or deleted")
        return

    print(f"Found {len(changes)} moved/deleted doc file(s)")

    # Parse existing redirects
    existing = parse_existing_redirects(redirects_file)
    print(f"Found {len(existing)} existing redirect(s)")

    # Find missing redirects
    missing = find_missing_redirects(changes, existing)

    if not missing:
        print("✅ All moved/deleted doc files have corresponding redirects")
        return

    # Separate auto-fixable (moves with known destination) from manual (deletes)
    auto_fixable = [(k, v) for k, v in missing if v is not None]
    manual_needed = [(k, v) for k, v in missing if v is None]

    # Auto-fix mode
    if args.auto_fix and auto_fixable:
        update_redirects_file(redirects_file, auto_fixable)

        if manual_needed:
            print(f"\n⚠️  {len(manual_needed)} deleted file(s) need manual redirects:")
            for old_key, _ in manual_needed:
                print(f"  • {old_key}")
            print("\nPlease add redirects for deleted files manually.")
            sys.exit(1)
        return

    # Report mode - show what's missing
    print("\n❌ Missing redirects detected!\n")

    for old_key, new_url in missing:
        if new_url:
            print(f"  • MOVED: {old_key} → {new_url}")
        else:
            print(f"  • DELETED: {old_key} (needs manual redirect target)")

    if auto_fixable:
        print("\n📝 Suggested additions to docs/source/redirects.py:\n")
        for old_key, new_url in auto_fixable:
            print(f'    "{old_key}": "{new_url}",')
        print("\n💡 To auto-fix, run:")
        print(
            f"    python3 .github/scripts/check_doc_redirects.py "
            f"--base-ref {args.base_ref} --auto-fix"
        )

    if manual_needed:
        print(f"\n⚠️  {len(manual_needed)} deleted file(s) need manual redirects.")
        print("Please determine appropriate redirect targets for deleted files.")

    sys.exit(1)


if __name__ == "__main__":
    main()
