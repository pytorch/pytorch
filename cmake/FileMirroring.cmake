# File mirroring previously handled by setup.py.
# Copies source files into torchgen/packaged/ and torch/_inductor/kernel/
# so they are included in the installed packages.
#
# These use SKBUILD_PLATLIB_DIR (set by scikit-build-core) to install into
# the Python package tree directly, since they need to go into packages
# other than torch/ (e.g., torchgen/) or into specific torch/ subdirectories.

# Fallback for standalone cmake invocations
if(NOT DEFINED SKBUILD_PLATLIB_DIR)
  set(SKBUILD_PLATLIB_DIR "${CMAKE_INSTALL_PREFIX}")
endif()

# --- mirror_files_into_torchgen ---
# Copy ATen native function definitions and templates into torchgen/packaged/
# so that torchgen can be used standalone without the full source tree.
install(FILES
  "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/native_functions.yaml"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torchgen/packaged/ATen/native"
)
install(FILES
  "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/tags.yaml"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torchgen/packaged/ATen/native"
)
install(DIRECTORY
  "${PROJECT_SOURCE_DIR}/aten/src/ATen/templates/"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torchgen/packaged/ATen/templates"
)
install(DIRECTORY
  "${PROJECT_SOURCE_DIR}/tools/autograd/"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torchgen/packaged/autograd"
)

# --- mirror_inductor_external_kernels ---
# Copy vendored external kernel sources into torch/_inductor.
set(_cutedsl_src "${PROJECT_SOURCE_DIR}/third_party/cutlass/examples/python/CuTeDSL/blackwell/grouped_gemm.py")
if(EXISTS "${_cutedsl_src}")
  install(FILES "${_cutedsl_src}"
    DESTINATION "${SKBUILD_PLATLIB_DIR}/torch/_inductor/kernel/vendored_templates"
    RENAME "cutedsl_grouped_gemm.py"
  )
  # Ensure __init__.py exists for the package
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/_vendored_templates_init.py" "")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_vendored_templates_init.py"
    DESTINATION "${SKBUILD_PLATLIB_DIR}/torch/_inductor/kernel/vendored_templates"
    RENAME "__init__.py"
  )
endif()

# --- Symlink-replacement copies ---
# Copy files that were previously handled via symlinks in setup.py.
install(FILES
  "${PROJECT_SOURCE_DIR}/torch/_utils_internal.py"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/tools/shared"
  RENAME "_utils_internal.py"
)
install(FILES
  "${PROJECT_SOURCE_DIR}/third_party/valgrind-headers/callgrind.h"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torch/utils/benchmark/utils/valgrind_wrapper"
)
install(FILES
  "${PROJECT_SOURCE_DIR}/third_party/valgrind-headers/valgrind.h"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torch/utils/benchmark/utils/valgrind_wrapper"
)
