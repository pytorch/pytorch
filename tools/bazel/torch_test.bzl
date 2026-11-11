"""Tests of one PyTorch test file: a target for each of its cases.

The cases of a file are known only once it is imported, with torch built, while
targets have to be declared when BUILD files load, before anything is built. So
the cases come from a build of their own: //test:cases lists them, in a second
Bazel that the loading of //test starts (see extensions.bzl).

A case runs in a build action, and its test only reports what that found. Unlike
a test, a build action can tell Bazel which of its inputs it did not use: here,
the torch Python files that no process of the case opened. Bazel then keeps the
result until one of the others changes.
"""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("//tools/bazel:torch_tree.bzl", "TorchTreeInfo")

# What a process that imports torch may use.
CPUS = 4

def _cpus(_os, _input_count):
    return {"cpu": CPUS}

def _run(ctx, mode, outputs, case = [], env = {}, **kwargs):
    """Runs the file's program (run_test.py) with torch as inputs it may leave unused."""
    tree = ctx.attr._tree[TorchTreeInfo]
    unused = ctx.actions.declare_file(ctx.label.name + ".unused")
    args = ctx.actions.args()
    args.add_all([mode, ctx.attr.file] + case)
    args.add_all([tree.python] + outputs + [unused])
    ctx.actions.run(
        executable = ctx.attr.runner[DefaultInfo].files_to_run,
        arguments = [args],
        inputs = depset([tree.python], transitive = [tree.files]),
        outputs = outputs + [unused],
        unused_inputs_list = unused,
        env = env | {"MKL_NUM_THREADS": str(CPUS), "OMP_NUM_THREADS": str(CPUS)},
        use_default_shell_env = True,
        resource_set = _cpus,
        **kwargs
    )

def _torch_list_impl(ctx):
    listed = ctx.actions.declare_file(ctx.label.name)
    _run(
        ctx,
        "--list",
        [listed],
        mnemonic = "TorchTestList",
        progress_message = "Listing the cases of test/" + ctx.attr.file,
    )
    return [DefaultInfo(files = depset([listed]))]

_REPORT = """#!/bin/bash
# Reports what the action that ran the case found.
cp "{xml}" "$XML_OUTPUT_FILE"
status=$(< "{status}")
[ "$status" = 0 ] || cat "{log}"
exit "$status"
"""

def _torch_case_impl(ctx):
    results = {kind: ctx.actions.declare_file("%s.%s" % (ctx.label.name, kind)) for kind in ("xml", "log", "status")}

    # A case uses the GPUs if its name says so, which is how the test framework
    # tells as well (wrap_with_cuda_policy in common_utils.py). The other cases
    # see no GPU, so they do not wait for one either.
    name = ctx.attr.case.lower()
    gpu = "cuda" in name or "gpu" in name
    _run(
        ctx,
        "--run",
        results.values(),
        case = [ctx.attr.case],
        env = {
            # pytest takes more options from this variable.
            "PYTEST_ADDOPTS": ctx.attr._pytest_args[BuildSettingInfo].value,
            # A result is a build output, so Bazel keeps it until an input changes,
            # even if the case failed. Changing this variable runs the case again.
            "RERUN": ctx.attr._rerun[BuildSettingInfo].value,
        } | ({} if gpu else {"CUDA_VISIBLE_DEVICES": ""}),
        # Bounded by --local_resources=gpu in .bazelrc.
        execution_requirements = {"resources:gpu:1": ""} if gpu else {},
        mnemonic = "TorchTestRun",
        progress_message = "Testing %{label}",
    )
    script = ctx.actions.declare_file(ctx.label.name)
    paths = {kind: result.short_path for kind, result in results.items()}
    ctx.actions.write(script, _REPORT.format(**paths), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(results.values()))]

_ATTRS = {
    "file": attr.string(doc = "The test file, as a path in test/."),
    "runner": attr.label(
        executable = True,
        # Not "exec": the cases run against the torch this build is for.
        cfg = "target",
        doc = "Program that runs the test file: run_test.py with what the file needs, torch aside.",
    ),
    "_tree": attr.label(default = "//:torch_tree"),
}

_torch_list = rule(
    implementation = _torch_list_impl,
    attrs = _ATTRS,
    doc = "The cases of a test file, as its entry of CASES in tests.bzl.",
)

_torch_case_test = rule(
    implementation = _torch_case_impl,
    test = True,
    attrs = _ATTRS | {
        "case": attr.string(doc = "The pytest id of the case."),
        "_pytest_args": attr.label(default = "//tools/bazel:pytest_args"),
        "_rerun": attr.label(default = "//tools/bazel:rerun"),
    },
)

def torch_test(name, srcs, cases = {}, data = [], deps = [], tags = [], target_compatible_with = [], **kwargs):
    """Declares the tests of test/<name>.py: <name>/<case> for each case.

    <name> itself runs them all, and <name>.cases lists them.

    Args:
      name: the test, like "test_nn" or "inductor/test_torchinductor".
      srcs: the Python files under test/ that it imports, itself included.
      cases: its cases, each one's pytest id by the name of its target.
      data: the other files under test/ that it reads.
      deps: the Python packages that it imports, torch aside.
      tags: tags of the tests.
      target_compatible_with: where the tests can run.
      **kwargs: what else the file needs, as arguments of py_binary.
    """
    py_binary(
        name = name + ".bin",
        srcs = srcs + ["//tools/bazel:run_test.py"],
        # The whole file in one process: bazel run //test:<name>.bin -- -k <case>
        args = [name + ".py", "--use-pytest"],
        # torch is not among the program's files: each action gets it as inputs
        # of its own. This only has the tree built, and sitecustomize.py puts it
        # on the path.
        data = data + ["//:torch_tree"],
        main = "//tools/bazel:run_test.py",
        # Every process would otherwise compile the modules it imports.
        pyc_collection = "include_pyc",
        tags = ["manual"],
        target_compatible_with = target_compatible_with,
        deps = deps + ["//tools/bazel:site"],
        **kwargs
    )

    common = dict(file = name + ".py", runner = name + ".bin", target_compatible_with = target_compatible_with)

    # Only //test:cases asks for the listing.
    _torch_list(name = name + ".cases", tags = tags + ["manual"], **common)
    for case, id in cases.items():
        # The test action only reports.
        _torch_case_test(name = "%s/%s" % (name, case), case = id, size = "small", tags = tags, **common)

    # A suite of no tests would stand for every test of the package.
    if cases:
        native.test_suite(name = name, tests = ["%s/%s" % (name, case) for case in cases], tags = tags)

def _torch_cases_impl(ctx):
    files = ctx.actions.declare_file(ctx.label.name + ".files")
    ctx.actions.write(files, "TESTS = %r\n" % ctx.attr.tests)
    tests = ctx.actions.declare_file("tests.bzl")
    args = ctx.actions.args()
    args.add_all([tests, files] + ctx.files.listings)
    ctx.actions.run_shell(
        command = 'tests=$1 files=$2; shift 2; { cat "$files"; echo "CASES = {"; cat /dev/null "$@"; echo "}"; } > "$tests"',
        arguments = [args],
        inputs = [files] + ctx.files.listings,
        outputs = [tests],
        mnemonic = "TorchTestCases",
        progress_message = "Writing the list of every test case",
    )
    return [DefaultInfo(files = depset([tests]))]

torch_cases = rule(
    implementation = _torch_cases_impl,
    attrs = {
        "listings": attr.label_list(allow_files = True, doc = "The <name>.cases of each test file."),
        "tests": attr.string_list_dict(doc = "TESTS: the test files, and the files under test/ each one needs."),
    },
    doc = "tests.bzl as @torch_tests provides it: the test files and the cases of each.",
)
