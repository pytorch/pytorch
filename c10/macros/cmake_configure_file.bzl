def cmake_configure_file(name, src, out, definitions):
    """Mimics CMake's configure_file in Bazel.

    Supports using config_setting's to customize the generated
    file. To use a boolean config_setting identified with the label
    "//c10:using_glog", you would pass
    "config_setting://c10:using_glog" as the value in definitions.

    Args:
      name: A unique name for this rule.
      src: The input file template.
      out: The generated output.
      definitions: A mapping of identifier in template to its value,
                   possibly configurable.
    """
    substitutions = []
    for identifier, value in definitions.items():
        if type(value) == "bool":
            template = "#define {}" if value else "#undef {}"
            substitutions += [template.format(identifier)]
        elif type(value) == "string":
            scheme, target = value.split(":", 1)
            if scheme != "config_setting":
                fail("String config value types must be of form " +
                     "config_setting://label/of/config_setting, but found " +
                     value)
            substitutions += select({
                target: ["#define " + identifier],
                "//conditions:default": ["#undef " + identifier],
            })
        else:
            fail("Unknown config value type. Only boolean literals or " +
                 "config_setting://label/of/config_setting are supported, " +
                 "but found: " + value)

    _header_template(
        name = name,
        src = src,
        out = out,
        substitutions = substitutions,
    )

# Forked from header_template_rule. header_template_rule is not
# compatible with our usage of select because its substitutions
# attribute is a dict, and dicts may not be appended with select. We
# get around this limitation by using a list as our substitutions. As
# such, we hardcode substitution semantics such that #define will be
# defined and #undef will be commented out.
def _header_template_impl(ctx):
    subs = {}
    for sub in ctx.attr.substitutions:
        cmd, token = sub.split(" ")
        replacement = "{} {}".format(cmd, token) if cmd == "#define" else "/* #undef {} */".format(token)
        subs["#cmakedefine {}".format(token)] = replacement
    ctx.actions.expand_template(
        template = ctx.file.src,
        output = ctx.outputs.out,
        substitutions = subs,
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

_header_template = rule(
    attrs = {
        "out": attr.output(mandatory = True),
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        # We use attr.string_list for compatibility with select and
        # config_setting. See the comment above _header_template_impl
        # for more information.
        "substitutions": attr.string_list(mandatory = True),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = _header_template_impl,
)
