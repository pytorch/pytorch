import os
import subprocess


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

    with open(fout, "w") as f:
        for lineid, line in enumerate(linker_script_lines):
            if lineid == text_line_start + 2:
                f.write("    *(\n")
                for plines in prioritized_text:
                    f.write(f"      .text.{plines}\n")
                f.write("    )\n")
            f.write(f"{line}\n")
