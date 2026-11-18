"""Used by CMake to transform *.metal into *_metallib.h."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "torch" / "utils"))
from _cpp_embed_headers import embed_headers


def write_metallib_headers(metal_filename: str, output_filename: str):
    embedded_headers = embed_headers(metal_filename)

    with open(output_filename, "w") as out:
        out.writelines(
            [
                "#include <ATen/native/mps/OperationUtils.h>\n",
                'static ::at::native::mps::MetalShaderLibrary lib(R"SHDR(\n',
                embedded_headers,
                ')SHDR");\n',
            ]
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage:\n {sys.argv[0]} metal_filename output_filename")
        sys.exit(1)

    write_metallib_headers(sys.argv[1], sys.argv[2])
