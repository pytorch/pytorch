"""Dependencies for CUDA rules."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def _local_cuda_impl(repository_ctx):
    # Path to CUDA Toolkit is
    # - taken from CUDA_PATH environment variable or
    # - determined through 'which ptxas' or
    # - defaults to '/usr/local/cuda'
    cuda_path = "/usr/local/cuda"
    ptxas_path = repository_ctx.which("ptxas")
    if ptxas_path:
        cuda_path = ptxas_path.dirname.dirname
    cuda_path = repository_ctx.os.environ.get("CUDA_PATH", cuda_path)

    defs_template = "def if_local_cuda(true, false = []):\n    return %s"
    if repository_ctx.path(cuda_path).exists:
        repository_ctx.symlink(cuda_path, "cuda")
        repository_ctx.symlink(Label("//private:BUILD.local_cuda"), "BUILD")
        repository_ctx.file("defs.bzl", defs_template % "true")
    else:
        repository_ctx.file("BUILD")  # Empty file
        repository_ctx.file("defs.bzl", defs_template % "false")

_local_cuda = repository_rule(
    implementation = _local_cuda_impl,
    environ = ["CUDA_PATH", "PATH"],
    # remotable = True,
)

def _rules_cc():
    if native.existing_rule("rules_cc"):
        fail("@rules_cc repository already exists. Unable to patch feature 'cuda'.")
    http_archive(
        name = "rules_cc",
        sha256 = "cb8ce8a25464b2a8536450971ad1b45ee309491c1f5e052a611b9e249cfdd35d",
        strip_prefix = "rules_cc-40548a2974f1aea06215272d9c2b47a14a24e556",
        patches = [Label("//private:rules_cc.patch")],
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/40548a2974f1aea06215272d9c2b47a14a24e556.tar.gz",
            "https://github.com/bazelbuild/rules_cc/archive/40548a2974f1aea06215272d9c2b47a14a24e556.tar.gz",
        ],
    )

def rules_cuda_dependencies(with_rules_cc = True):
    """Loads rules_cuda dependencies. To be called from WORKSPACE file.

    Args:
      with_rules_cc: whether to load and patch rules_cc repository.
    """
    maybe(
        name = "bazel_skylib",
        repo_rule = http_archive,
        sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        ],
    )
    maybe(
        name = "platforms",
        repo_rule = http_archive,
        sha256 = "48a2d8d343863989c232843e01afc8a986eb8738766bfd8611420a7db8f6f0c3",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.2/platforms-0.0.2.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.2/platforms-0.0.2.tar.gz",
        ],
    )
    _local_cuda(name = "local_cuda")
    if with_rules_cc:
        _rules_cc()
