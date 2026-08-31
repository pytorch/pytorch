# LLVM BOLT profiles

This directory holds the BOLT profiles consumed when building with
`USE_LLVM_BOLT=ON`. The profiles are kept compressed in `torch.tar.zst` to keep
the repo small; the build extracts them into the build tree at configure time.

## Archive contents

Each library has a YAML profile named `lib<target>.yaml`. Each call to
`torch_optimize_layout_if_enabled` passes profile names in priority order and
the first profile found is used. The optimized libraries are: `libtorch_cuda`,
`libtorch_cpu`, `libtorch_python`, `libc10`, and `libc10_cuda`.

`libtorch_python` may also have profiles for individual Python versions,
named `libtorch_python-${Python_SOABI}.yaml` (for example,
`libtorch_python-cpython-313-aarch64-linux-gnu.yaml`). The matching
version-specific profile is preferred when present; otherwise the build
uses `libtorch_python.yaml`.

## How profiles are consumed

Optimization happens at build time, in `torch_optimize_layout_if_enabled`
(`cmake/public/utils.cmake`), as a `POST_BUILD` step on each optimized target.
Right after a library is linked, its freshly-linked `lib<name>.so` is moved
into a `prebolt/` subdirectory and `llvm-bolt` writes the optimized library
back in its place. The build tree thus carries the optimized lib at the
canonical path, so `install(TARGETS)` mirrors it (and applies the usual
`$ORIGIN` rpath fixup), while the unoptimized original is retained under
`prebolt/` (not installed).

## Profile collection

Profiles must be collected on binaries built the BOLT-compatible
compile flags (`-fno-plt -fno-reorder-blocks-and-partition`,
`-Wl,--emit-relocs`). For best results, the same binaries should be
used for profile collection and optimization.
