#!/usr/bin/env python3
"""
Initializer script for pyrefly linter.

This script:
1. Installs required pip packages for pyrefly
2. Checks if .pyi stub files exist
3. If stub files are missing, generates them
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """Find repository root using git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        sys.exit("Not in a git repository")


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


def generate_stub_files() -> bool:
    """Generate .pyi stub files by running generation scripts."""
    repo_root = find_repo_root()

    print("Generating .pyi stub files...")

    # Step 1: Generate torch version
    print("Generating torch version...")
    result = subprocess.run(
        [sys.executable, "-m", "tools.generate_torch_version", "--is_debug=false"],
        cwd=repo_root,
    )
    if result.returncode != 0:
        print("Failed to generate torch version", file=sys.stderr)
        return False

    # Step 2: Generate main stub files
    print("Generating main stub files...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.pyi.gen_pyi",
            "--native-functions-path",
            "aten/src/ATen/native/native_functions.yaml",
            "--tags-path",
            "aten/src/ATen/native/tags.yaml",
            "--deprecated-functions-path",
            "tools/autograd/deprecated.yaml",
        ],
        cwd=repo_root,
    )
    if result.returncode != 0:
        print("Failed to generate main stub files", file=sys.stderr)
        return False

    # Step 3: Generate DataPipe stub files
    print("Generating DataPipe stub files...")
    result = subprocess.run(
        [sys.executable, "torch/utils/data/datapipes/gen_pyi.py"],
        cwd=repo_root,
    )
    if result.returncode != 0:
        print("Failed to generate DataPipe stub files", file=sys.stderr)
        return False

    print("All stub files generated successfully")
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

    # Step 2: Check stub files
    print("\n Checking for .pyi stub files...")
    all_exist, missing_files = check_stub_files()

    if all_exist:
        print("All .pyi stub files already exist")
    else:
        print(f"Missing {len(missing_files)} stub file(s):")
        for missing in missing_files:
            print(f"     - {missing}")

        if args.dry_run == "1":
            print("\n[DRY RUN] Would generate stub files, but skipping in dry run mode")
        else:
            print("\n Generating missing stub files...")
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
