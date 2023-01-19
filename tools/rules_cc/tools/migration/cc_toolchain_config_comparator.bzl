"""A test rule that compares two C++ toolchain configuration rules in proto format."""

def _impl(ctx):
    first_toolchain_config_proto = ctx.actions.declare_file(
        ctx.label.name + "_first_toolchain_config.proto",
    )
    ctx.actions.write(
        first_toolchain_config_proto,
        ctx.attr.first[CcToolchainConfigInfo].proto,
    )

    second_toolchain_config_proto = ctx.actions.declare_file(
        ctx.label.name + "_second_toolchain_config.proto",
    )
    ctx.actions.write(
        second_toolchain_config_proto,
        ctx.attr.second[CcToolchainConfigInfo].proto,
    )

    script = ("%s --before='%s' --after='%s'" % (
        ctx.executable._comparator.short_path,
        first_toolchain_config_proto.short_path,
        second_toolchain_config_proto.short_path,
    ))
    test_executable = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(test_executable, script, is_executable = True)

    runfiles = ctx.runfiles(files = [first_toolchain_config_proto, second_toolchain_config_proto])
    runfiles = runfiles.merge(ctx.attr._comparator[DefaultInfo].default_runfiles)

    return DefaultInfo(runfiles = runfiles, executable = test_executable)

cc_toolchain_config_compare_test = rule(
    implementation = _impl,
    attrs = {
        "first": attr.label(
            mandatory = True,
            providers = [CcToolchainConfigInfo],
            doc = "A C++ toolchain config rule",
        ),
        "second": attr.label(
            mandatory = True,
            providers = [CcToolchainConfigInfo],
            doc = "A C++ toolchain config rule",
        ),
        "_comparator": attr.label(
            default = ":ctoolchain_comparator",
            executable = True,
            cfg = "host",
        ),
    },
    test = True,
)
