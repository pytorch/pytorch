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
        "patches": attr.label_list(),
        "patch_strip": attr.int(),
        "build_file": attr.label(),
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
