Tests in this directory are meant to guard certain ATen/c10 util functions and data structures are implemented in a header-only fashion, to make sure AOTInductor generated CPU model code is ABI backward-compatible.

Tests that test functionality offered by the shims and require linking against torch should go into the `shim` test directory.

Tests that test device-only headeronly functionality should go into the `cuda/` subdirectory. These are built into the same binary only when `USE_CUDA` is on.
