"""Rule that produces PyTorch's native build outputs with the regular CMake build."""

def _whole_machine(_os, _input_count):
    # The CMake build schedules its own compile jobs on every core, so ask for
    # more CPUs than exist: Bazel then runs this action alone.
    return {"cpu": 1000000}

def _torch_native_impl(ctx):
    outputs = {path: ctx.actions.declare_file(path) for path in ctx.attr.files}
    outputs |= {path: ctx.actions.declare_directory(path) for path in ctx.attr.dirs}

    srcs = ctx.actions.args()
    srcs.set_param_file_format("multiline")
    srcs.use_param_file("--srcs=%s", use_always = True)
    srcs.add_all(ctx.files.srcs)

    args = ctx.actions.args()
    args.add("--status", ctx.info_file)
    for path, output in outputs.items():
        args.add("--output", "%s=%s" % (path, output.path))

    ctx.actions.run(
        executable = ctx.executable._builder,
        arguments = [srcs, args],
        # The status file changes when anything under third_party/ does.
        inputs = ctx.files.srcs + [ctx.info_file],
        outputs = outputs.values(),
        mnemonic = "TorchNativeBuild",
        progress_message = "Building PyTorch native libraries with CMake",
        use_default_shell_env = True,
        # Python packages come from the builder's dependencies, never ~/.local.
        env = {"PYTHONNOUSERSITE": "1"},
        # Unsandboxed and uncached remotely: the builder keeps its CMake build
        # directory in the output base between runs.
        execution_requirements = {"local": "1"},
        resource_set = _whole_machine,
    )
    files = depset(outputs.values())
    return [DefaultInfo(files = files, runfiles = ctx.runfiles(transitive_files = files))]

torch_native = rule(
    implementation = _torch_native_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            doc = "Every file outside third_party/ that the CMake build reads.",
        ),
        "files": attr.string_list(
            doc = "Installed files to expose, as paths in the install tree.",
        ),
        "dirs": attr.string_list(
            doc = "Installed directories to expose, as paths in the install tree.",
        ),
        "_builder": attr.label(
            default = "//tools/bazel:build_torch",
            executable = True,
            cfg = "exec",
        ),
    },
)
