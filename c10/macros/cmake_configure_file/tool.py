from __future__ import annotations

import pathlib
import subprocess
import sys

def main(argv: list[str]) -> None:
    template_path = pathlib.Path(sys.argv[1])
    output_path = pathlib.Path(sys.argv[2])
    definitions = list(sys.argv[3:])
    cmake_configure_file(template_path, definitions, output_path)

def cmake_configure_file(template_path: pathlib.Path, definitions: list[str], output_path: pathlib.Path):
    command = ["sed", "--regexp-extended"]
    for definition in definitions:
        command.append(
            "--expr=s@#cmakedefine {}@#define {}@".format(
                definition,
                definition,
            ),
        )

    # Replace any that remain with /* #undef FOO */.
    command.append("--expr=s@#cmakedefine ([A-Z0-9_]+)@/* #undef \\1 */@")

    # The input is the final argument to sed.
    command.append(template_path)

    subprocess.run(command, check=True, stdout=output_path.open("x"))


if __name__ == "__main__":
    main(sys.argv)
