"""Repositories derived from the source tree."""

# Set, as a --repo_env, in the build that lists the test cases.
LISTING = "TORCH_TESTS_LISTING"

# What the build of //test:cases reads, as labels.
_SOURCES = 'kind("source file", deps(//test:cases)) + buildfiles(deps(//test:cases))'

# The files outside that build's packages that configure it.
_CONFIGURATION = [".bazelrc", ".bazelversion", "MODULE.bazel", "user.bazelrc"]

# Copies a file to a directory under the name of its checksum, in place of any
# other copy there, and prints the new path.
_KEEP = """
sum=$(sha256sum "$1" | cut -d" " -f1) && mkdir -p "$2" || exit 1
find "$2" -maxdepth 1 -name "*.bzl" ! -name "$sum.bzl" -delete
[ -e "$2/$sum.bzl" ] || cp "$1" "$2/$sum.bzl" || exit 1
echo "$2/$sum.bzl"
"""

def _files(rctx):
    """tests.bzl with the test files that test/run_test.py runs, and no cases."""
    root = rctx.workspace_root
    script = rctx.path(Label("//tools/bazel:list_tests.py"))
    rctx.watch_tree(root.get_child("test"))
    rctx.watch(root.get_child("tools/testing/discover_tests.py"))
    rctx.watch(script)

    # -B keeps the interpreter from writing bytecode into the source tree.
    result = rctx.execute([rctx.path(rctx.attr._python), "-B", script, root])
    if result.return_code != 0:
        fail("could not list tests: %s" % result.stderr)
    rctx.file("tests.bzl", "TESTS = %r\nCASES = {}\n" % json.decode(result.stdout))

def _cases(rctx):
    """tests.bzl as a second Bazel builds it: the test files and the cases of each.

    The cases of a test file are known only once it is imported, with torch
    built, and Bazel needs them while BUILD files load, before it builds
    anything. So a Bazel of its own builds //test:cases here, in an output base
    inside this one.
    """
    root = rctx.workspace_root
    state = rctx.path(".").dirname.dirname.get_child("torch_tests")
    if not rctx.which("bazel"):
        fail("bazel is not on PATH, which is where Bazelisk puts the Bazel it started")
    bazel = [rctx.which("bazel"), "--output_base=%s" % state.get_child("output_base")]
    listing = "--repo_env=%s=1" % LISTING

    # Its bazel-bin link goes here instead of over the workspace's.
    quiet = ["--symlink_prefix=%s/bazel-" % rctx.path("."), "--color=no", "--curses=no", "--noshow_progress"]
    rctx.report_progress("building PyTorch and listing the test cases with a second Bazel")
    build = rctx.execute(bazel + ["build", listing] + quiet + ["//test:cases"], working_directory = str(root), timeout = 24 * 60 * 60)
    if build.return_code != 0:
        fail("could not list the test cases:\n" + build.stderr[-20000:])

    # Bazel loads the result through a link to a copy named after its content,
    # so it loads //test again when the tests change, not whenever this runs.
    kept = rctx.execute(["sh", "-c", _KEEP, "sh", rctx.path("bazel-bin/test/tests.bzl"), state])
    if kept.return_code != 0:
        fail("could not keep the list of test cases: " + kept.stderr)
    rctx.symlink(kept.stdout.strip(), "tests.bzl")

    # This runs again when anything changes in the directories that build reads.
    sources = rctx.execute(bazel + ["query", listing, _SOURCES], working_directory = str(root))
    if sources.return_code != 0:
        fail("could not list the sources of the test cases:\n" + sources.stderr[-20000:])
    labels = [label[2:].replace(":", "/") for label in sources.stdout.splitlines() if label.startswith("//")]
    for top in {label.lstrip("/").split("/")[0]: None for label in labels}.keys() + _CONFIGURATION:
        path = root.get_child(top)
        if path.is_dir:
            rctx.watch_tree(path)
        else:
            rctx.watch(path)

def _test_list_impl(rctx):
    rctx.file("BUILD.bazel", "")
    if rctx.getenv(LISTING):
        _files(rctx)
    else:
        _cases(rctx)

_test_list = repository_rule(
    implementation = _test_list_impl,
    attrs = {"_python": attr.label(default = "@python_3_14_host//:python")},
    doc = "Provides tests.bzl with the test files that test/run_test.py runs and the cases of each.",
)

def _torch_impl(_mctx):
    _test_list(name = "torch_tests")

torch = module_extension(implementation = _torch_impl)
