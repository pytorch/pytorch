def _impl(repository_ctx):
    archive = repository_ctx.attr.name + ".tar"
    reference = Label("@%s_unpatched//:README" % repository_ctx.attr.name)
    dirname = repository_ctx.path(reference).dirname
    repository_ctx.execute(["tar", "hcf", archive, "-C", dirname, "."])
    repository_ctx.extract(archive)
    for patch in repository_ctx.attr.patches:
        repository_ctx.patch(repository_ctx.path(patch), repository_ctx.attr.patch_strip)
    build_file = repository_ctx.path(repository_ctx.attr.build_file)
    repository_ctx.execute(["cp", build_file, "BUILD.bazel"])

_patched_rule = repository_rule(
    implementation = _impl,
    attrs = {
        "build_file": attr.label(),
        "patch_strip": attr.int(),
        "patches": attr.label_list(),
    },
)

def new_patched_local_repository(name, path, **kwargs):
    native.new_local_repository(
        name = name + "_unpatched",
        build_file_content = """
pkg_tar(name = "content", srcs = glob(["**"]))
""",
        path = path,
    )
    _patched_rule(name = name, **kwargs)

def _new_empty_repository_impl(repo_ctx):
    build_file = repo_ctx.attr.build_file
    build_file_content = repo_ctx.attr.build_file_content
    if not (bool(build_file) != bool(build_file_content)):
        fail("Exactly one of 'build_file' or 'build_file_content' is required")

    if build_file_content:
        repo_ctx.file("BUILD", build_file_content)
    elif build_file:
        repo_ctx.template("BUILD", repo_ctx.attr.build_file, {})

new_empty_repository = repository_rule(
    attrs = {
        "build_file": attr.label(allow_files = True),
        "build_file_content": attr.string(),
    },
    implementation = _new_empty_repository_impl,
)

"""Create an empty repository with the supplied BUILD file.

This is mostly useful to create wrappers for specific target that we want
to be used with the '@' syntax.
"""
