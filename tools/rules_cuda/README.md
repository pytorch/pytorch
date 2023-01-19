# [CUDA](http://nvidia.com/cuda) rules for [Bazel](http://bazel.build)

The `@rules_cuda` repository primarily provides a `cuda_library()` macro which
allows compiling a C++ bazel target containing CUDA device code using nvcc or
clang.

The CUDA example program in this repository can be run with:

```
bazel run --cuda //examples:hello_cuda
```

For this, [rules_cc](https://github.com/bazelbuild/rules_cc)'s auto-configured
toolchain is patched to support a `cuda` feature which is then enabled for
`cuda_library()` targets.

A secondary `@local_cuda` repository contains bazel targets for the CUDA toolkit
installed on the execution machine.

## Requirements

Requires bazel 4.0 or later.

## Setup

Add the following snippet to your `WORKSPACE` file:

```python
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_cuda",
    urls = ["https://github.com/bazelbuild/rules_cuda/archive/????.zip"],
    sha256 = "????",
)
load("//cuda:dependencies.bzl", "rules_cuda_dependencies")
rules_cuda_dependencies()
load("@rules_cc//cc:repositories.bzl", "rules_cc_toolchains")
rules_cc_toolchains()
```

## Using `cuda_library`

Then, in the `BUILD` and/or `*.bzl` files in your own workspace, you can create
C++ targets containing CUDA device code with the `cuda_library` macro:

```python
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "kernel",
    srcs = ["kernel.cu"],
)
```

## Configuration

The normal `bazel build` command allows configuring a few properties of the
`@rules_cuda` repository.

*   `--@rules_cuda//cuda:enable_cuda`: triggers the `is_cuda_enabled` config
    setting. This config setting is not used in the package itself, but is
    rather intended as a central switch to let users know whether building with
    CUDA support has been requested.
*   `--@rules_cuda//cuda:cuda_targets=sm_xy,...`: configures the list of CUDA
    compute architectures to compile for as a comma-separated list. The default
    is `"sm_52"`. For details, please consult the
    [--cuda-gpu-arch](https://llvm.org/docs/CompileCudaWithLLVM.html#invoking-clang)
    clang flag.
*   `--@rules_cuda//cuda:copts=...`: comma-separated list of arguments to add to
    cuda_library() compile commands.
*   `--@rules_cuda//cuda:cuda_runtime=<label>`: configures the CUDA runtime
    target. The default is `"@local_cuda//:cuda_runtime_static"`. This target is
    implicitly added as a dependency to cuda_library() targets.
    `@rules_cuda//cuda:cuda_runtime` should be used as the CUDA runtime target
    everywhere so that the actual runtime can be configured in a central place.
*   `--@rules_cuda//cuda:compiler=<compiler>`: configures the compiler to use
    for `cuda_library()` sources. Supported values are `nvcc` (default) and
    `clang`. The latter uses the auto-configured host compiler with
    clang-specific CUDA flags, and should therefore be combined with
    `--repo_env=CC=clang`.
*   `--repo_env=CUDA_PATH=<path>`: Specifies the path to the locally installed
    CUDA toolkit. The default is `/usr/local/cuda`.

## Toolchains

This repository patches [rules_cc](https://github.com/bazelbuild/rules_cc)'s
auto-config toolchain to add support for a 'cuda' feature which triggers
compiling CUDA code for the chosen target architectures. The functionality for
this is provided by cuda/toolchain.bzl and can also be applied to custom
toolchains.
