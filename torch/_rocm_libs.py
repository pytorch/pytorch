"""Canonical list of ROCm shared libraries bundled into / preloaded by torch.

Single source of truth shared by two very different consumers, so the two never
drift apart:

* ``torch/__init__.py`` imports :data:`ROCM_SO_FILES` and RTLD_GLOBAL-preloads
  each one (via ``_preload_rocm_deps``) so a TheRock-built wheel is importable
  without ``LD_LIBRARY_PATH``. Only libs that a TheRock-built libtorch actually
  ``DT_NEEDED``\\s belong here, and the **order matters**: it is leaf-first so
  each preload satisfies the inter-lib ``NEEDED`` entries of the ones loaded
  after it.
* ``.ci/manywheel/repair_wheel.py`` copies every lib in
  :data:`ROCM_SO_FILES` **plus** :data:`ROCM_SO_FILES_BUNDLE_ONLY` into
  ``torch/lib/`` and patchelf-rewrites ``NEEDED`` entries. Order is irrelevant
  there (it just copies + rewrites each file).

The module is intentionally import-light: it holds only plain string lists and
imports nothing (in particular not ``torch``), so it can be loaded both as the
``torch._rocm_libs`` submodule and standalone-by-path from an unpacked wheel by
``repair_wheel.py`` (which must not ``import torch`` in the build container).
"""

# Core ROCm runtime libs a TheRock-built libtorch DT_NEEDEDs. Ordered LEAF-FIRST
# so RTLD_GLOBAL preloading satisfies inter-lib NEEDED entries as we go. This is
# the set torch/__init__.py::_preload_rocm_deps preloads.
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

# Extra libs that are BUNDLED into the wheel by repair_wheel.py but are NOT part
# of the preload set, because libtorch does not DT_NEEDED them (they are optional
# / dlopen'd at runtime): MAGMA, the rocprofiler/aqlprofile profiler stack, and
# the rocRoller hipBLASLt backend. They ship in the wheel so they are available
# when something loads them, but they need not be RTLD_GLOBAL-preloaded to make
# ``import torch`` succeed.
ROCM_SO_FILES_BUNDLE_ONLY: list[str] = [
    "libmagma.so",
    "librocprofiler-sdk.so",
    "librocprofiler-register.so",
    "libhsa-amd-aqlprofile64.so",
    "librocroller.so",
]

# Full set of ROCm libs bundled into torch/lib/ by repair_wheel.py.
ROCM_SO_FILES_ALL: list[str] = ROCM_SO_FILES + ROCM_SO_FILES_BUNDLE_ONLY
