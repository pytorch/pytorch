# This Bazel rules file is derived from https://github.com/tensorflow/tensorflow/blob/master/third_party/common.bzl

# Rule for simple expansion of template files. This performs a simple
# search over the template file for the keys in substitutions,
# and replaces them with the corresponding values.
#
# Typical usage:
#   load("/tools/build_rules/template_rule", "template_rule")
#   template_rule(
#       name = "ExpandMyTemplate",
#       src = "my.template",
#       out = "my.txt",
#       substitutions = {
#         "$VAR1": "foo",
#         "$VAR2": "bar",
#       }
#   )
#
# Args:
#   name: The name of the rule.
#   template: The template file to expand
#   out: The destination of the expanded file
#   substitutions: A dictionary mapping strings to their substitutions

def template_rule_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file.src,
        output = ctx.outputs.out,
        substitutions = ctx.attr.substitutions,
    )

template_rule = rule(
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "out": attr.output(mandatory = True),
        "substitutions": attr.string_dict(mandatory = True),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = template_rule_impl,
)

# Header template rule is an extension of template substitution rule
# That also makes this header a valid dependency for cc_library
# From https://stackoverflow.com/a/55407399
def header_template_rule_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file.src,
        output = ctx.outputs.out,
        substitutions = ctx.attr.substitutions,
    )
    return [
            # create a provider which says that this
            # out file should be made available as a header
            CcInfo(compilation_context=cc_common.create_compilation_context(

                # pass out the include path for finding this header
                includes=depset([ctx.outputs.out.dirname, ctx.bin_dir.path]),

                # and the actual header here.
                headers=depset([ctx.outputs.out])
            ))
        ]

header_template_rule = rule(
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "out": attr.output(mandatory = True),
        "substitutions": attr.string_dict(mandatory = True),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = header_template_rule_impl,
)
