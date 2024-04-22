load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_cc//cc:defs.bzl", "cc_library")

CPU_CAPABILITY_NAMES = ["DEFAULT", "AVX2"]
CAPABILITY_COMPILER_FLAGS = {
    "AVX2": ["-mavx2", "-mfma", "-mf16c"],
    "DEFAULT": [],
}

PREFIX = "aten/src/ATen/native/"
EXTRA_PREFIX = "aten/src/ATen/"

def intern_build_aten_ops(copts, deps, extra_impls):
    for cpu_capability in CPU_CAPABILITY_NAMES:
        srcs = []
        for impl in native.glob(
            [
                PREFIX + "cpu/*.cpp",
                PREFIX + "quantized/cpu/kernels/*.cpp",
            ],
        ):
            name = impl.replace(PREFIX, "")
            out = PREFIX + name + "." + cpu_capability + ".cpp"
            native.genrule(
                name = name + "_" + cpu_capability + "_cp",
                srcs = [impl],
                outs = [out],
                cmd = "cp $< $@",
            )
            srcs.append(out)

        for impl in extra_impls:
            name = impl.replace(EXTRA_PREFIX, "")
            out = EXTRA_PREFIX + name + "." + cpu_capability + ".cpp"
            native.genrule(
                name = name + "_" + cpu_capability + "_cp",
                srcs = [impl],
                outs = [out],
                cmd = "cp $< $@",
            )
            srcs.append(out)

        cc_library(
            name = "ATen_CPU_" + cpu_capability,
            srcs = srcs,
            copts = copts + [
                "-DCPU_CAPABILITY=" + cpu_capability,
                "-DCPU_CAPABILITY_" + cpu_capability,
            ] + CAPABILITY_COMPILER_FLAGS[cpu_capability],
            deps = deps,
            linkstatic = 1,
        )
    cc_library(
        name = "ATen_CPU",
        deps = [":ATen_CPU_" + cpu_capability for cpu_capability in CPU_CAPABILITY_NAMES],
        linkstatic = 1,
    )

def generate_aten_impl(ctx):
    # Declare the entire ATen/ops/ directory as an output
    ops_dir = ctx.actions.declare_directory("aten/src/ATen/ops")
    outputs = [ops_dir] + ctx.outputs.outs

    install_dir = paths.dirname(ops_dir.path)
    ctx.actions.run(
        outputs = outputs,
        inputs = ctx.files.srcs,
        executable = ctx.executable.generator,
        arguments = [
            "--source-path",
            "aten/src/ATen",
            "--per-operator-headers",
            "--install_dir",
            install_dir,
        ],
        use_default_shell_env = True,
        mnemonic = "GenerateAten",
    )
    return [DefaultInfo(files = depset(outputs))]

generate_aten = rule(
    implementation = generate_aten_impl,
    attrs = {
        "generator": attr.label(
            executable = True,
            allow_files = True,
            mandatory = True,
            cfg = "exec",
        ),
        "outs": attr.output_list(),
        "srcs": attr.label_list(allow_files = True),
    },
)
