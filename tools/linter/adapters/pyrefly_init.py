#!/usr/bin/env python3
"""
Initializer script for pyrefly linter.

This script:
1. Installs required pip packages for pyrefly
2. Checks if .pyi stub files exist
3. Checks if the commit hash has changed since stubs were last generated
4. Regenerates stub files if they're missing or if the commit has changed
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Find repository root using git or hg, with fallback to searching for torch/."""
    # Try git first
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try hg (Mercurial)
    try:
        result = subprocess.run(
            ["hg", "root"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: search for torch/ directory
    cwd = Path.cwd()
    if (cwd / "torch").is_dir():
        return cwd

    for parent in cwd.parents:
        if (parent / "torch").is_dir():
            return parent

    # Last resort: use current directory
    print(
        "Warning: Could not find repository root, using current directory",
        file=sys.stderr,
    )
    return cwd


def install_packages(dry_run: str) -> bool:
    """Install required pip packages using pip_init.py."""
    repo_root = find_repo_root()
    pip_init_script = repo_root / "tools" / "linter" / "adapters" / "pip_init.py"

    packages = [
        'numpy==2.1.0 ; python_version >= "3.12"',
        "expecttest==0.3.0",
        "pyrefly==0.36.2",
        "sympy==1.13.3",
        "types-requests==2.27.25",
        "types-pyyaml==6.0.2",
        "types-tabulate==0.8.8",
        "types-protobuf==5.29.1.20250403",
        "types-setuptools==79.0.0.20250422",
        "types-jinja2==2.11.9",
        "types-colorama==0.4.6",
        "filelock==3.18.0",
        "junitparser==2.1.1",
        "rich==14.1.0",
        "optree==0.17.0",
        "types-openpyxl==3.1.5.20250919",
        "types-python-dateutil==2.9.0.20251008",
    ]

    cmd = [sys.executable, str(pip_init_script), f"--dry-run={dry_run}"] + packages

    print("Installing pyrefly dependencies...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Failed to install packages", file=sys.stderr)
        return False

    print("Packages installed successfully")
    return True


def check_stub_files() -> tuple[bool, list[str]]:
    """Check if all required .pyi stub files exist."""
    repo_root = find_repo_root()

    expected_stub_files = [
        "torch/_C/__init__.pyi",
        "torch/_C/_VariableFunctions.pyi",
        "torch/_C/_nn.pyi",
        "torch/_VF.pyi",
        "torch/return_types.pyi",
        "torch/nn/functional.pyi",
        "torch/utils/data/datapipes/datapipe.pyi",
    ]

    missing_files = []
    for stub_file in expected_stub_files:
        file_path = repo_root / stub_file
        if not file_path.exists():
            missing_files.append(stub_file)

    all_exist = len(missing_files) == 0
    return all_exist, missing_files


def get_current_commit_hash() -> str | None:
    """Get current commit hash from git or hg.

    Returns commit hash string, or None if not in a VCS repository.
    """
    # Try git first
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try hg (Mercurial)
    try:
        result = subprocess.run(
            ["hg", "id", "-i"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def check_commit_changed() -> bool:
    """Check if the current commit differs from when stubs were last generated.

    Returns True if stubs need regeneration (commit changed or no record exists).
    """
    repo_root = find_repo_root()
    stub_commit_file = repo_root / ".pyrefly_stub_commit"

    # If the commit file doesn't exist, we need to generate
    if not stub_commit_file.exists():
        return True

    try:
        # Read the saved commit hash
        saved_commit = stub_commit_file.read_text().strip()

        # Get the current commit hash
        current_commit = get_current_commit_hash()

        # If we can't get a commit hash, check if saved was "unknown"
        # If so, assume stubs are still valid; otherwise regenerate
        if current_commit is None:
            if saved_commit == "unknown":
                return False  # Both unknown, assume valid
            return True  # Was tracked before, now can't track - regenerate

        # Return True if commits differ (need regeneration)
        return saved_commit != current_commit

    except OSError as e:
        print(f"Warning: Could not check commit hash: {e}", file=sys.stderr)
        # If we can't check, assume we need to regenerate to be safe
        return True


def generate_stub_files() -> bool:
    """Generate .pyi stub files by calling the generate_stubs.sh script."""
    repo_root = find_repo_root()
    generate_script = repo_root / "tools" / "linter" / "adapters" / "generate_stubs.sh"

    if not generate_script.exists():
        print(
            f"Error: generate_stubs.sh not found at {generate_script}", file=sys.stderr
        )
        return False

    result = subprocess.run([str(generate_script)])

    if result.returncode != 0:
        print("Failed to generate stub files", file=sys.stderr)
        return False

    return True


def main() -> None:
    """Main entry point for pyrefly initialization."""
    parser = argparse.ArgumentParser(
        description="Initialize pyrefly linter with dependencies and stub files"
    )
    parser.add_argument(
        "--dry-run",
        default="0",
        help="Pass '1' for dry run mode (don't actually install/generate)",
    )

    args = parser.parse_args()

    print("Initializing pyrefly linter...")

    # Step 1: Install packages
    if not install_packages(args.dry_run):
        sys.exit(1)

    # Step 2: Check stub files and commit hash
    print("\n Checking for .pyi stub files...")
    all_exist, missing_files = check_stub_files()
    commit_changed = check_commit_changed()

    need_regeneration = False
    regeneration_reason = []

    if not all_exist:
        need_regeneration = True
        regeneration_reason.append(f"Missing {len(missing_files)} stub file(s)")
        for missing in missing_files:
            print(f"     - {missing}")

    if commit_changed and all_exist:
        need_regeneration = True
        regeneration_reason.append("Commit hash changed since last generation")
        print("Commit has changed since stubs were last generated")

    if not need_regeneration:
        print("All .pyi stub files exist and are up to date")
    else:
        print(f"Stub regeneration needed: {', '.join(regeneration_reason)}")

        if args.dry_run == "1":
            print("\n[DRY RUN] Would generate stub files, but skipping in dry run mode")
        else:
            print("\n Generating stub files...")
            if not generate_stub_files():
                sys.exit(1)

            # Verify they were created
            all_exist, still_missing = check_stub_files()
            if not all_exist:
                print(
                    f"Failed to generate {len(still_missing)} stub file(s)",
                    file=sys.stderr,
                )
                for missing in still_missing:
                    print(f"     - {missing}", file=sys.stderr)
                sys.exit(1)

            print("All stub files verified")

    print("\n Pyrefly initialization complete!")


if __name__ == "__main__":
    main()
