# LLVM BOLT profiles

This directory holds the BOLT profiles consumed when building with
`USE_LLVM_BOLT=ON`. The profiles are kept compressed in `torch.tar.zst` to keep
the repo small; the build extracts them into the build tree at configure time.

## Archive contents

One YAML profile per library, named `lib<target>.yaml`. Each call to
`target_optimize_if_llvm_bolt_enabled(<target>)` looks up `lib<target>.yaml`
(e.g. target `torch_cuda` -> `libtorch_cuda.yaml`). The optimized libraries
are: `libtorch_cuda`, `libtorch_cpu`, `libtorch`, `libtorch_python`, `libc10`,
`libc10_cuda`.

## How profiles are consumed

Optimization happens at build time, in `target_optimize_if_llvm_bolt_enabled`
(`cmake/public/utils.cmake`), as a `POST_BUILD` step on each optimized target.
Right after a library is linked, its freshly-linked `lib<name>.so` is moved
into a `prebolt/` subdirectory and `llvm-bolt` writes the optimized library
back in its place. The build tree thus carries the optimized lib at the
canonical path, so `install(TARGETS)` mirrors it (and applies the usual
`$ORIGIN` rpath fixup), while the unoptimized original is retained under
`prebolt/` (not installed).

## Profile collection

Profiles must be collected on binaries built with the prioritized-text linker
script enabled (`USE_PRIORITIZED_TEXT_FOR_LD=ON`) and the BOLT-compatible
compile flags (`-fno-plt -fno-reorder-blocks-and-partition`,
`-Wl,--emit-relocs`), matching the layout BOLT optimizes here.
