from __future__ import annotations

import subprocess
import sys

def main(template_path: str, output_path: str, definitions: list[str]) -> None:
    subprocess.run(" ".join(cmake_configure_file_cmd(template_path, definitions, output_path)),
                   check=True, shell=True)

def cmake_configure_file_cmd(template_path, definitions, output_path):
    command = ["sed", "--regexp-extended"]
    for definition in definitions:
        command.append(
            # Note that because of the shell redirection at the end of
            # the command, we must return arguments that are parseable
            # by a shell, hence the quoting here.
            "--expr='s@#cmakedefine {}@#define {}@'".format(
                definition,
                definition,
            ),
        )

    # Replace any that remain with /* #undef FOO */.
    command.append("--expr='s@#cmakedefine ([A-Z0-9_]+)@/* #undef \\1 */@'")

    # The input is the final argument to sed.
    command.append(template_path)

    # You can not specify the output to sed, so file redirection is
    # our only option. This means that even though we're returning a
    # list, we have to join it into a string and have a shell evaluate
    # it.
    command.append("> {}".format(output_path))

    return command


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], list(sys.argv[3:]))
