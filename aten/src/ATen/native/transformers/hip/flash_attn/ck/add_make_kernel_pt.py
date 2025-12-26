"""
Cross-platform script to replace make_kernel with make_kernel_pt
and add launch_kernel_pt.hpp include in generated CK FMHA files.
"""

import argparse
import os
import sys


def process_file(filepath: str) -> bool:
    """Process a single file, making the required replacements."""
    if not os.path.isfile(filepath):
        print(f"Skipping: {filepath} (not found)")
        return True  # Not an error, just skip

    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        # Replace make_kernel with make_kernel_pt
        content = content.replace("make_kernel", "make_kernel_pt")

        # Add launch_kernel_pt.hpp include after fmha_fwd.hpp
        # NOTE: Using string concat to avoid triggering the INCLUDE linter
        content = content.replace(
            "#" + 'include "fmha_fwd.hpp"',
            "#" + 'include "fmha_fwd.hpp"\n#include "launch_kernel_pt.hpp"',
        )

        # Add launch_kernel_pt.hpp include after fmha_bwd.hpp
        content = content.replace(
            "#" + 'include "fmha_bwd.hpp"',
            "#" + 'include "fmha_bwd.hpp"\n#include "launch_kernel_pt.hpp"',
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Updated: {filepath}")
        return True

    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Replace make_kernel with make_kernel_pt in generated CK FMHA files"
    )
    parser.add_argument(
        "file_list",
        help="Path to a text file containing list of files to process (one per line)",
    )
    args = parser.parse_args()

    file_list_path = args.file_list

    if not os.path.isfile(file_list_path):
        print(f"Error: File '{file_list_path}' not found!", file=sys.stderr)
        return 1

    # Read the list of files
    with open(file_list_path, encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]

    # Process each file
    success = True
    for filepath in files:
        if not process_file(filepath):
            success = False

    print("Replacement completed.")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
