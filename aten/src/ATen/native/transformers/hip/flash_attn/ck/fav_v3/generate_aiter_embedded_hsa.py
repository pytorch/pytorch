#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Generate aiter_embedded_hsa.h with embedded binary .co files for AITER HSA kernels.

import argparse
import re
import sys
from pathlib import Path


def sanitize_identifier(name: str) -> str:
    """Convert a file path to a valid C++ identifier."""
    return re.sub(r"[^a-zA-Z0-9]", "_", name)


def bytes_to_hex_array(data: bytes, bytes_per_line: int = 16) -> str:
    """Convert bytes to a formatted C hex array string."""
    hex_bytes = []
    for i, byte in enumerate(data):
        if i > 0 and i % bytes_per_line == 0:
            hex_bytes.append("\n    ")
        hex_bytes.append(f"0x{byte:02x}")
        if i < len(data) - 1:
            hex_bytes.append(",")
    return "".join(hex_bytes)


def generate_embedded_hsa_header(
    hsa_dir: Path, output_file: Path, subdirs: list[str]
) -> int:
    """
    Generate a C++ header file embedding all .co files from specified subdirectories.

    Args:
        hsa_dir: Base directory containing hsa files (e.g., third_party/aiter/hsa)
        output_file: Path to the output header file
        subdirs: List of subdirectories to scan for .co files (e.g., ["gfx942/fmha_v3_bwd", "gfx950/fmha_v3_bwd"])

    Returns:
        Number of .co files embedded
    """
    # Collect all .co files
    co_files: list[tuple[str, Path]] = []
    for subdir in subdirs:
        pattern_dir = hsa_dir / subdir
        if pattern_dir.exists():
            for co_file in sorted(pattern_dir.glob("*.co")):
                # Key format: hsa/gfx942/fmha_v3_bwd/xxx.co
                # Use as_posix() to ensure forward slashes on all platforms
                rel_path = co_file.relative_to(hsa_dir).as_posix()
                map_key = f"hsa/{rel_path}"
                co_files.append((map_key, co_file))

    if not co_files:
        print(f"Warning: No .co files found in {hsa_dir} under {subdirs}")
        return 0

    # Generate header content
    # Using std::string_view instead of std::span<const unsigned char> for C++17 compatibility
    # std::string_view provides .data() method which is what hipModuleLoadData needs
    lines = [
        "// Auto-generated file. Do not edit.",
        "// Embedded AITER HSA binary files for fmha_v3_bwd",
        "#pragma once",
        "",
        "#include <cstdint>",
        "#include <string>",
        "#include <string_view>",
        "#include <unordered_map>",
        "",
        "// Define AITER_EMBEDDED_HSA_MAP macro so that aiter_hip_common.h",
        "// can detect the embedded map is available via #if defined(AITER_EMBEDDED_HSA_MAP)",
        "#define AITER_EMBEDDED_HSA_MAP ::aiter_hsa::embedded_hsa_map",
        "",
        "namespace aiter_hsa {",
        "",
    ]

    # Generate array declarations and map entries
    array_entries = []
    for map_key, co_file in co_files:
        with open(co_file, "rb") as f:
            data = f.read()

        # Only generate array and map entry if file has content
        if len(data) > 0:
            safe_name = sanitize_identifier(co_file.relative_to(hsa_dir).as_posix())
            array_name = f"data_{safe_name}"
            file_size = len(data)
            array_entries.append((map_key, array_name, file_size))

            hex_array = bytes_to_hex_array(data)
            lines.append(
                f"alignas(4096) inline const unsigned char {array_name}[] = {{\n    {hex_array}\n}};"
            )
            lines.append("")

    # Generate the map
    lines.append(
        "inline const std::unordered_map<std::string, std::string_view> embedded_hsa_map = {"
    )
    for map_key, array_name, file_size in array_entries:
        lines.append(
            f'    {{"{map_key}", std::string_view(reinterpret_cast<const char*>({array_name}), {file_size})}},'
        )
    lines.append("};")
    lines.append("")
    lines.append("} // namespace aiter_hsa")
    lines.append("")

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    return len(array_entries)


def main():
    parser = argparse.ArgumentParser(
        description="Generate aiter_embedded_hsa.h with embedded binary .co files"
    )
    parser.add_argument(
        "--hsa-dir", required=True, type=Path, help="Path to the aiter hsa directory"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Path to the output header file"
    )
    parser.add_argument(
        "--subdirs",
        nargs="+",
        default=["gfx942/fmha_v3_bwd", "gfx950/fmha_v3_bwd"],
        help="Subdirectories to scan for .co files",
    )

    args = parser.parse_args()

    if not args.hsa_dir.exists():
        print(f"Error: HSA directory does not exist: {args.hsa_dir}", file=sys.stderr)
        return 1

    count = generate_embedded_hsa_header(args.hsa_dir, args.output, args.subdirs)
    print(f"Generated {args.output} with {count} embedded .co files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
