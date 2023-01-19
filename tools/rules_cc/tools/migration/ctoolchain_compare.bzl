"""A test rule that compares two CToolchains in proto format."""

def _impl(ctx):
    toolchain_config_proto = ctx.actions.declare_file(ctx.label.name + "_toolchain_config.proto")
    ctx.actions.write(
        toolchain_config_proto,
        ctx.attr.toolchain_config[CcToolchainConfigInfo].proto,
    )

    script = ("%s --before='%s' --after='%s' --toolchain_identifier='%s'" % (
        ctx.executable._comparator.short_path,
        ctx.file.crosstool.short_path,
        toolchain_config_proto.short_path,
        ctx.attr.toolchain_identifier,
    ))
    test_executable = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(test_executable, script, is_executable = True)

    runfiles = ctx.runfiles(files = [toolchain_config_proto, ctx.file.crosstool])
    runfiles = runfiles.merge(ctx.attr._comparator[DefaultInfo].default_runfiles)

    return DefaultInfo(runfiles = runfiles, executable = test_executable)

cc_toolchains_compare_test = rule(
    implementation = _impl,
    attrs = {
        "crosstool": attr.label(
            mandatory = True,
            allow_single_file = True,
            doc = "Location of the CROSSTOOL file",
        ),
        "toolchain_config": attr.label(
            mandatory = True,
            providers = [CcToolchainConfigInfo],
            doc = ("Starlark rule that replaces the CROSSTOOL file functionality " +
                   "for the CToolchain with the given identifier"),
        ),
        "toolchain_identifier": attr.string(
            mandatory = True,
            doc = "identifier of the CToolchain that is being compared",
        ),
        "_comparator": attr.label(
            default = ":ctoolchain_comparator",
            executable = True,
            cfg = "host",
        ),
    },
    test = True,
)
