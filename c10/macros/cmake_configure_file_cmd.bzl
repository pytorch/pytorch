def cmake_configure_file_cmd(definitions, path):
    command = ["sed",
               "--in-place",
               "--regexp-extended",
               path,  # this is our input and output argument
               ]
    for definition in definitions:
        command.append(
            # We must return arguments that are parseable by a shell
            # because of our internal implementation, hence the
            # quoting here.
            "--expr='s@#cmakedefine {}@#define {}@'".format(
                definition,
                definition,
            ),
        )

    # Replace any that remain with /* #undef FOO */.
    command.append("--expr='s@#cmakedefine ([A-Z0-9_]+)@/* #undef \\1 */@'")

    return command
