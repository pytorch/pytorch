#!/usr/bin/env python3
"""Generate a stub shaders.h when slangc is not available.

This allows compilation to succeed, but all shader dispatch will fail at
runtime with "Failed to create shader module" since the SPIR-V is invalid.
"""

from pathlib import Path

SHADER_DIR = Path(__file__).parent.parent / "shaders"
OUTPUT_DIR = Path(__file__).parent.parent / "csrc" / "generated"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shaders = sorted(SHADER_DIR.rglob("*.slang"))
    shaders = [s for s in shaders if "common/" not in str(s)]

    lines = [
        "#pragma once",
        "#include <cstdint>",
        "#include <cstddef>",
        "",
        "// STUB shaders.h — generated without slangc.",
        "// All SPIR-V arrays contain invalid data.",
        "// Run `python tools/compile_shaders.py` with slangc to generate real shaders.",
        "namespace torch_vulkan { namespace shaders {",
        "",
    ]

    for shader in shaders:
        rel_path = shader.relative_to(SHADER_DIR)
        var_base = str(rel_path.with_suffix("")).replace("/", "_").replace("\\", "_")

        lines.append(f"// {rel_path} (STUB)")
        lines.append(f"static const uint32_t {var_base}_fwd[] = {{0}};")
        lines.append(f"static const size_t {var_base}_fwd_size = sizeof({var_base}_fwd);")
        lines.append("")

    lines.append("}} // namespace torch_vulkan::shaders")

    header_path = OUTPUT_DIR / "shaders.h"
    with open(header_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated stub: {header_path}")


if __name__ == "__main__":
    main()
