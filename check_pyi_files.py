#!/usr/bin/env python3
"""Check if generated .pyi stub files exist in the PyTorch repository."""

from pathlib import Path

# List of expected generated .pyi files (relative to repo root)
EXPECTED_PYI_FILES = [
    "torch/_C/__init__.pyi",
    "torch/_C/_VariableFunctions.pyi",
    "torch/_C/_nn.pyi",
    "torch/_VF.pyi",
    "torch/return_types.pyi",
    "torch/nn/functional.pyi",
    "torch/utils/data/datapipes/datapipe.pyi",
]


def check_pyi_files():
    """Check if all expected .pyi files exist."""
    repo_root = Path(__file__).parent

    all_exist = True
    missing_files = []
    existing_files = []

    print("Checking for generated .pyi files...\n")

    for pyi_file in EXPECTED_PYI_FILES:
        file_path = repo_root / pyi_file
        exists = file_path.exists()

        if exists:
            existing_files.append(pyi_file)
            print(f"✓ {pyi_file}")
        else:
            missing_files.append(pyi_file)
            all_exist = False
            print(f"✗ {pyi_file} (MISSING)")

    print(f"\n{'='*60}")
    print(f"Summary: {len(existing_files)}/{len(EXPECTED_PYI_FILES)} files found")

    if missing_files:
        print(f"\nMissing files ({len(missing_files)}):")
        for missing in missing_files:
            print(f"  - {missing}")
        print("\nTo generate missing files, run:")
        print("  python3 -m tools.generate_torch_version --is_debug=false")
        print("  python3 -m tools.pyi.gen_pyi \\")
        print("      --native-functions-path aten/src/ATen/native/native_functions.yaml \\")
        print("      --tags-path aten/src/ATen/native/tags.yaml \\")
        print("      --deprecated-functions-path tools/autograd/deprecated.yaml")
        print("  python3 torch/utils/data/datapipes/gen_pyi.py")
    else:
        print("\n✓ All .pyi files exist!")

    return all_exist


if __name__ == "__main__":
    import sys
    all_exist = check_pyi_files()
    sys.exit(0 if all_exist else 1)
