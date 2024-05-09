# Forked from header_template_rule. header_template_rule is not
# compatible with our usage of select because its substitutions
# attribute is a dict, and dicts may not be appended with select. We
# get around this limitation by using a list as our substitutions.
def _cmake_configure_file_impl(ctx):
    command = ["cat $1"]
    for definition in ctx.attr.definitions:
        command.append(
            "| sed 's@#cmakedefine {}@#define {}@'".format(
                definition,
                definition,
            ),
        )

    # Replace any that remain with /* #undef FOO */.
    command.append("| sed -r 's@#cmakedefine ([A-Z0-9_]+)@/* #undef \\1 */@'")
    command.append("> $2")

    ctx.actions.run_shell(
        inputs = [ctx.file.src],
        outputs = [ctx.outputs.out],
        command = " ".join(command),
        arguments = [
            ctx.file.src.path,
            ctx.outputs.out.path,
        ],
    )
    return [
        # create a provider which says that this
        # out file should be made available as a header
        CcInfo(compilation_context = cc_common.create_compilation_context(

            # pass out the include path for finding this header
            includes = depset([ctx.outputs.out.dirname, ctx.bin_dir.path]),

            # and the actual header here.
            headers = depset([ctx.outputs.out]),
        )),
    ]

cmake_configure_file = rule(
    implementation = _cmake_configure_file_impl,
    doc = """
Mimics CMake's configure_file in Bazel.

Args:
  name: A unique name for this rule.
  src: The input file template.
  out: The generated output.
  definitions: A mapping of identifier in template to its value.
""",
    attrs = {
        # We use attr.string_list for compatibility with select and
        # config_setting. See the comment above _cmake_configure_file_impl
        # for more information.
        "definitions": attr.string_list(mandatory = True),
        "out": attr.output(mandatory = True),
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
)
