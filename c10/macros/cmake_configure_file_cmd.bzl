def cmake_configure_file_cmd(template_path, definitions, output_path):
    command = ["cat {}".format(template_path)]
    for definition in definitions:
        command.append(
            "| sed 's@#cmakedefine {}@#define {}@'".format(
                definition,
                definition,
            ),
        )

    # Replace any that remain with /* #undef FOO */.
    command.append("| sed -r 's@#cmakedefine ([A-Z0-9_]+)@/* #undef \\1 */@'")
    command.append("> {}".format(output_path))

    return command
