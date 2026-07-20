"""Used by CMake to transform *.metal into *_metallib.h."""

import subprocess
import sys
from pathlib import Path


def write_metallib_headers(metal_filename: str, output_filename: str):
    # This is run during CMake configuration, when we can't import _cpp_embed_headers
    # directly.  Run it as a script instead.
    embedded_headers = subprocess.run(
        [
            sys.executable,
            Path(__file__).resolve().parent.parent
            / "torch"
            / "utils"
            / "_cpp_embed_headers.py",
            metal_filename,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    with open(output_filename, "w") as out:
        out.writelines(
            [
                "#include <ATen/native/mps/OperationUtils.h>\n",
                'static ::at::native::mps::MetalShaderLibrary lib(R"SHDR(\n',
                embedded_headers.stdout,
                ')SHDR");\n',
            ]
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage:\n {sys.argv[0]} metal_filename output_filename")
        sys.exit(1)

    write_metallib_headers(sys.argv[1], sys.argv[2])
