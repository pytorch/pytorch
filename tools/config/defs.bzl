"""
 Macros for selecting with / without various GPU libraries.  Most of these are meant to be used
 directly by tensorflow in place of their build's own configure.py + bazel-gen system.
"""

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@rules_cc//cc:defs.bzl", "cc_library")

def if_cuda(if_true, if_false = []):
    """Helper for selecting based on the whether CUDA is configured. """
    return selects.with_or({
        "@//tools/config:cuda_enabled_and_capable": if_true,
        "//conditions:default": if_false,
    })

def if_tensorrt(if_true, if_false = []):
    """Helper for selecting based on the whether TensorRT is configured. """
    return select({
        "//conditions:default": if_false,
    })

def if_rocm(if_true, if_false = []):
    """Helper for selecting based on the whether ROCM is configured. """
    return select({
        "//conditions:default": if_false,
    })

def if_sycl(if_true, if_false = []):
    """Helper for selecting based on the whether SYCL/ComputeCPP is configured."""

    # NOTE: Tensorflow expects some stange behavior (see their if_sycl) if we
    # actually plan on supporting this at some point.
    return select({
        "//conditions:default": if_false,
    })

def if_ccpp(if_true, if_false = []):
    """Helper for selecting based on the whether ComputeCPP is configured. """
    return select({
        "//conditions:default": if_false,
    })

def cuda_default_copts():
    return if_cuda(["-DGOOGLE_CUDA=1"])

def cuda_default_features():
    return if_cuda(["-per_object_debug_info", "-use_header_modules", "cuda_clang"])

def rocm_default_copts():
    return if_rocm(["-x", "rocm"])

def rocm_copts(opts = []):
    return rocm_default_copts() + if_rocm(opts)

def cuda_is_configured():
    # FIXME(dcollins): currently only used by tensorflow's xla stuff, which we aren't building.  However bazel
    # query hits it so this needs to be defined.  Because bazel doesn't actually resolve config at macro expansion
    # time, `select` can't be used here (since xla expects lists of strings and not lists of select objects).
    # Instead, the xla build rules must be rewritten to use `if_cuda_is_configured`
    return False

def if_cuda_is_configured(x):
    return if_cuda(x, [])

def if_rocm_is_configured(x):
    return if_rocm(x, [])

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
    command.append("| sed --regexp-extended 's@#cmakedefine (\\w+)@/* #undef \\1 */@'")
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

rules = struct(
    cc_library = cc_library,
    cmake_configure_file = cmake_configure_file,
    select = select,
)
