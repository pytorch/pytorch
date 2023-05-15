workspace(name = "pytorch")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//tools/rules:workspace.bzl", "new_patched_local_repository")

http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-40548a2974f1aea06215272d9c2b47a14a24e556",
    patches = [
        "//:tools/rules_cc/cuda_support.patch",
    ],
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/40548a2974f1aea06215272d9c2b47a14a24e556.tar.gz",
        "https://github.com/bazelbuild/rules_cc/archive/40548a2974f1aea06215272d9c2b47a14a24e556.tar.gz",
    ],
)

http_archive(
    name = "rules_cuda",
    strip_prefix = "runtime-b1c7cce21ba4661c17ac72421c6a0e2015e7bef3/third_party/rules_cuda",
    urls = ["https://github.com/tensorflow/runtime/archive/b1c7cce21ba4661c17ac72421c6a0e2015e7bef3.tar.gz"],
)

load("@rules_cuda//cuda:dependencies.bzl", "rules_cuda_dependencies")

rules_cuda_dependencies(with_rules_cc = False)

load("@rules_cc//cc:repositories.bzl", "rules_cc_toolchains")

rules_cc_toolchains()

http_archive(
    name = "bazel_skylib",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-992381ced716ae12122360b0fbadbc3dda436dbf",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/992381ced716ae12122360b0fbadbc3dda436dbf.zip"],
)

new_local_repository(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    path = "third_party/pybind11",
)

http_archive(
    name = "com_github_glog",
    strip_prefix = "glog-0.4.0",
    urls = [
        "https://github.com/google/glog/archive/v0.4.0.tar.gz",
    ],
    build_file_content = """
licenses(['notice'])

load(':bazel/glog.bzl', 'glog_library')
# TODO: figure out why enabling gflags leads to SIGSEV on the logging init
glog_library(with_gflags=0)
    """
)

http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-2.2.2",
    urls = [
        "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz",
    ],
)

new_local_repository(
    name = "gloo",
    build_file = "//third_party:gloo.BUILD",
    path = "third_party/gloo",
)

new_local_repository(
    name = "onnx",
    build_file = "//third_party:onnx.BUILD",
    path = "third_party/onnx",
)

new_local_repository(
    name = "foxi",
    build_file = "//third_party:foxi.BUILD",
    path = "third_party/foxi",
)

local_repository(
    name = "com_google_protobuf",
    path = "third_party/protobuf",
)

new_local_repository(
    name = "eigen",
    build_file = "//third_party:eigen.BUILD",
    path = "third_party/eigen",
)

new_local_repository(
    name = "cutlass",
    build_file = "//third_party:cutlass.BUILD",
    path = "third_party/cutlass",
)

new_local_repository(
    name = "fbgemm",
    build_file = "//third_party:fbgemm/BUILD.bazel",
    path = "third_party/fbgemm",
    repo_mapping = {"@cpuinfo" : "@org_pytorch_cpuinfo"}
)

new_local_repository(
    name = "ideep",
    build_file = "//third_party:ideep.BUILD",
    path = "third_party/ideep",
)

new_local_repository(
    name = "mkl_dnn",
    build_file = "//third_party:mkl-dnn.BUILD",
    path = "third_party/ideep/mkl-dnn",
)

new_local_repository(
    name = "org_pytorch_cpuinfo",
    build_file = "//third_party:cpuinfo/BUILD.bazel",
    path = "third_party/cpuinfo",
)

new_local_repository(
    name = "asmjit",
    build_file = "//third_party:fbgemm/third_party/asmjit.BUILD",
    path = "third_party/fbgemm/third_party/asmjit",
)

new_local_repository(
    name = "sleef",
    build_file = "//third_party:sleef.BUILD",
    path = "third_party/sleef",
)

new_local_repository(
    name = "fmt",
    build_file = "//third_party:fmt.BUILD",
    path = "third_party/fmt",
)

new_local_repository(
    name = "kineto",
    build_file = "//third_party:kineto.BUILD",
    path = "third_party/kineto",
)

new_patched_local_repository(
    name = "tbb",
    patches = [
        "@//third_party:tbb.patch",
    ],
    patch_strip = 1,
    build_file = "//third_party:tbb.BUILD",
    path = "third_party/tbb",
)

new_local_repository(
    name = "tensorpipe",
    build_file = "//third_party:tensorpipe.BUILD",
    path = "third_party/tensorpipe",
)

http_archive(
    name = "mkl",
    build_file = "//third_party:mkl.BUILD",
    strip_prefix = "lib",
    sha256 = "59154b30dd74561e90d547f9a3af26c75b6f4546210888f09c9d4db8f4bf9d4c",
    urls = [
        "https://anaconda.org/anaconda/mkl/2020.0/download/linux-64/mkl-2020.0-166.tar.bz2",
    ],
)

http_archive(
    name = "mkl_headers",
    build_file = "//third_party:mkl_headers.BUILD",
    sha256 = "2af3494a4bebe5ddccfdc43bacc80fcd78d14c1954b81d2c8e3d73b55527af90",
    urls = [
        "https://anaconda.org/anaconda/mkl-include/2020.0/download/linux-64/mkl-include-2020.0-166.tar.bz2",
    ],
)

http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
    sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python", python_version="3")

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

new_local_repository(
    name = "cuda",
    build_file = "@//third_party:cuda.BUILD",
    path = "/usr/local/cuda",
)

new_local_repository(
    name = "cudnn",
    build_file = "@//third_party:cudnn.BUILD",
    path = "/usr/",
)

local_repository(
    name = "com_github_google_flatbuffers",
    path = "third_party/flatbuffers",
)

local_repository(
    name = "google_benchmark",
    path = "third_party/benchmark",
)

local_repository(
    name = "com_google_googletest",
    path = "third_party/googletest",
)

local_repository(
    name = "pthreadpool",
    path = "third_party/pthreadpool",
    repo_mapping = {"@com_google_benchmark" : "@google_benchmark"}
)

local_repository(
    name = "FXdiv",
    path = "third_party/FXdiv",
    repo_mapping = {"@com_google_benchmark" : "@google_benchmark"}
)

local_repository(
    name = "XNNPACK",
    path = "third_party/XNNPACK",
    repo_mapping = {"@com_google_benchmark" : "@google_benchmark"}
)

local_repository(
    name = "gemmlowp",
    path = "third_party/gemmlowp/gemmlowp",
)

### Unused repos start

# `unused` repos are defined to hide bazel files from submodules of submodules.
# This allows us to run `bazel build //...` and not worry about the submodules madness.
# Otherwise everything traverses recursively and a lot of submodules of submodules have
# they own bazel build files.

local_repository(
    name = "unused_tensorpipe_googletest",
    path = "third_party/tensorpipe/third_party/googletest",
)

local_repository(
    name = "unused_fbgemm",
    path = "third_party/fbgemm",
)

local_repository(
    name = "unused_ftm_bazel",
    path = "third_party/fmt/support/bazel",
)

local_repository(
    name = "unused_kineto_fmt_bazel",
    path = "third_party/kineto/libkineto/third_party/fmt/support/bazel",
)

local_repository(
    name = "unused_kineto_dynolog_googletest",
    path = "third_party/kineto/libkineto/third_party/dynolog/third_party/googletest",
)

local_repository(
    name = "unused_kineto_dynolog_gflags",
    path = "third_party/kineto/libkineto/third_party/dynolog/third_party/gflags",
)

local_repository(
    name = "unused_kineto_dynolog_glog",
    path = "third_party/kineto/libkineto/third_party/dynolog/third_party/glog",
)

local_repository(
    name = "unused_kineto_googletest",
    path = "third_party/kineto/libkineto/third_party/googletest",
)

local_repository(
    name = "unused_onnx_benchmark",
    path = "third_party/onnx/third_party/benchmark",
)

local_repository(
    name = "unused_onnx_tensorrt_benchmark",
    path = "third_party/onnx-tensorrt/third_party/onnx/third_party/benchmark",
)

### Unused repos end
