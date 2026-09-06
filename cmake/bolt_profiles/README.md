# LLVM BOLT profiles

This directory holds the BOLT profiles consumed when building with
`USE_LLVM_BOLT=ON`. The profiles are kept compressed in `torch.tar.zst` to keep
the repo small; the build extracts them into the build tree at configure time.

## Archive contents

The archive contains YAML profiles named `lib<target>.yaml` for each of
the following targets: `c10`, `c10_cuda`, `torch_cpu`, `torch_cuda`, and
`torch_python`. It may also contain python version specific profiles for
`torch_python` e.g. `libtorch_python-cpython-313-aarch64-linux-gnu.yaml`.

```
torch.tar.zst
|_ libc10.yaml
|_ libc10_cuda.yaml
|_ libtorch_cpu.yaml
|_ libtorch_cuda.yaml
|_ libtorch_python.yaml
|_ libtorch_python-cpython-313-aarch64-linux-gnu.yaml  (optional)
|_ libtorch_python-cpython-314t-aarch64-linux-gnu.yaml (optional)
```

## How profiles are consumed

Optimization happens at build time, in `torch_optimize_layout_if_enabled`
(`cmake/public/utils.cmake`), as a `POST_BUILD` step on each optimized target.

For libraries other than `libtorch_python`, we use `lib<target>.yaml` as
the BOLT profile. For `libtorch_python`, we first try to find a version-
specific profile and fall back to `libtorch_python.yaml` if needed.

Right after a library is linked, its freshly-linked `lib<name>.so` is moved
into a `prebolt/` subdirectory and `llvm-bolt` writes the optimized library
back in its place. The build tree thus carries the optimized lib at the
canonical path, while the unoptimized original is retained under `prebolt/`

## Profile collection

Profiles must be collected on binaries built the BOLT-compatible
compile flags (`-fno-plt -fno-reorder-blocks-and-partition`,
`-Wl,--emit-relocs`). For best results, the same binaries should be
used for profile collection and optimization.
