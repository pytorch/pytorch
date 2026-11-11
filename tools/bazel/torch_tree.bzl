"""Lays the torch packages out the way pip installs them, each file its own input.

A test's action can tell Bazel afterwards which of its inputs it never opened,
and is then left alone when only those change (see torch_test.bzl). That works
for files an action lists as inputs one by one, not for files in the runfiles of
the program it runs, so torch reaches a test through this tree instead.
"""

load("@rules_python//python:py_info.bzl", "PyInfo")

TorchTreeInfo = provider(
    doc = "The torch packages as a directory of links.",
    fields = {
        "files": "depset[File]: every file of the tree.",
        "python": "File: the paths of its .py and .pyc files, one per line.",
    },
)

def _torch_tree_impl(ctx):
    library = ctx.attr.library[PyInfo]
    sources = library.direct_original_sources.to_list() + library.direct_pyc_files.to_list()
    links = []
    for target in sources + ctx.files.data + ctx.files.native:
        path = "%s/%s" % (ctx.label.name, target.short_path)
        link = ctx.actions.declare_directory(path) if target.is_directory else ctx.actions.declare_file(path)
        ctx.actions.symlink(output = link, target_file = target)
        links.append(link)

    python = ctx.actions.declare_file(ctx.label.name + ".python")
    paths = ctx.actions.args()
    paths.add_all([link for link in links if link.extension in ("py", "pyc")])
    ctx.actions.write(python, paths)

    # A program cannot have the tree in its runfiles, but it can have this file,
    # which exists once the tree does and never changes.
    built = ctx.actions.declare_file(ctx.label.name + ".built")
    ctx.actions.run_shell(
        command = "touch $1",
        arguments = [built.path],
        inputs = links,
        outputs = [built],
        execution_requirements = {"no-sandbox": "1"},
        mnemonic = "TorchTree",
        progress_message = "Laying out the torch packages",
    )
    return [
        DefaultInfo(files = depset([built])),
        TorchTreeInfo(files = depset(links), python = python),
    ]

torch_tree = rule(
    implementation = _torch_tree_impl,
    attrs = {
        "library": attr.label(
            providers = [PyInfo],
            doc = "The Python sources, compiled (precompile = \"enabled\").",
        ),
        "data": attr.label_list(allow_files = True, doc = "Other files of the packages."),
        "native": attr.label(doc = "The native build outputs, as paths in the install tree."),
    },
)
