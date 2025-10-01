import argparse
import os
import subprocess
from pathlib import Path


def gen_linker_script(
    filein: str = "cmake/prioritized_text.txt", fout: str = "cmake/linker_script.ld"
) -> None:
    with open(filein) as f:
        prioritized_text = f.readlines()
        prioritized_text = [
            line.replace("\n", "") for line in prioritized_text if line != "\n"
        ]
    ld = os.environ.get("LD", "ld")
    linker_script_lines = subprocess.check_output([ld, "-verbose"], text=True).split(
        "\n"
    )

    indices = [
        i
        for i, x in enumerate(linker_script_lines)
        if x == "=================================================="
    ]
    linker_script_lines = linker_script_lines[indices[0] + 1 : indices[1]]

    text_line_start = [
        i for i, line in enumerate(linker_script_lines) if ".text           :" in line
    ]
    assert len(text_line_start) == 1, "The linker script has multiple text sections!"
    text_line_start = text_line_start[0]

    # ensure that parent directory exists before writing
    fout = Path(fout)
    fout.parent.mkdir(parents=True, exist_ok=True)

    with open(fout, "w") as f:
        for lineid, line in enumerate(linker_script_lines):
            if lineid == text_line_start + 2:
                f.write("    *(\n")
                for plines in prioritized_text:
                    f.write(f"      .text.{plines}\n")
                f.write("    )\n")
            f.write(f"{line}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate linker file based on prioritized symbols. Used for link-time optimization.",
    )
    parser.add_argument(
        "--filein",
        help="Path to prioritized_text.txt input file",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fout", help="Output path for linker ld file", default=argparse.SUPPRESS
    )
    # convert args to a dict to pass to gen_linker_script
    kwargs = vars(parser.parse_args())
    gen_linker_script(**kwargs)
