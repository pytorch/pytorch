"""Single source of truth for the ROCm shared-library basenames used by the
ROCm wheel tooling.

This module is intentionally dependency-free (no ``torch`` import, no third-party
imports) so it can be consumed two very different ways:

- ``torch/__init__.py`` imports ``ROCM_SO_FILES`` (as ``_rocm_core_libs``) and
  preloads those libs, leaf-first, with ``RTLD_GLOBAL`` before ``import torch._C``
  so a TheRock ROCm wheel is self-resolving.
- ``.ci/manywheel/repair_wheel.py`` loads this file *by path* (it runs before the
  built wheel is importable) and bundles ``ROCM_SO_FILES_ALL`` into the wheel.

Two lists, one union:

- ``ROCM_SO_FILES``            -- the runtime core libs that must be preloaded.
                                  Ordered leaf-first so that ``RTLD_GLOBAL``
                                  preloading satisfies inter-lib NEEDED entries as
                                  we go.
- ``ROCM_SO_FILES_BUNDLE_ONLY`` -- libs that ship in the wheel but are NOT
                                  preloaded, because libtorch does not DT_NEEDED
                                  them (they are optional / ``dlopen``'d at
                                  runtime): MAGMA, the rocprofiler / aqlprofile
                                  profiler stack, and the rocRoller hipBLASLt
                                  backend. Preload order is irrelevant for these,
                                  so this list is kept in the (order-insensitive)
                                  form the bundler used historically.
- ``ROCM_SO_FILES_ALL``        -- the union that the wheel bundler ships. Set-equal
                                  to the historical ``repair_wheel.py`` list.
"""

ROCM_SO_FILES: list[str] = [
    "libamd_comgr.so",
    "libhsa-runtime64.so",
    "libamdhip64.so",
    "libhiprtc.so",
    "librocm-core.so",
    "librocm_smi64.so",
    "libroctx64.so",
    "libroctracer64.so",
    "librocblas.so",
    "libhipblas.so",
    "libhipblaslt.so",
    "librocfft.so",
    "libhipfft.so",
    "librocrand.so",
    "libhiprand.so",
    "librocsolver.so",
    "libhipsolver.so",
    "librocsparse.so",
    "libhipsparse.so",
    "libhipsparselt.so",
    "libMIOpen.so",
    "librccl.so",
]

# Bundled into the wheel but intentionally NOT preloaded: libtorch has no
# DT_NEEDED entry for these, so nothing needs their symbols resolved at import
# time. They are optional / loaded on demand (MAGMA linear-algebra, the
# rocprofiler + aqlprofile profiling stack, and the rocRoller hipBLASLt backend).
ROCM_SO_FILES_BUNDLE_ONLY: list[str] = [
    "libmagma.so",
    "librocprofiler-sdk.so",
    "librocprofiler-register.so",
    "libhsa-amd-aqlprofile64.so",
    "librocroller.so",
]

# Every ROCm shared lib that the wheel bundler ships.
ROCM_SO_FILES_ALL: list[str] = ROCM_SO_FILES + ROCM_SO_FILES_BUNDLE_ONLY
