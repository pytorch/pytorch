# Install non-Python package data into the torch package.  Populates the wheel
# under scikit-build-core; only included when SKBUILD is set.
#
# Destinations are absolute paths under SKBUILD_PLATLIB_DIR rather than
# prefix-relative ones: the install prefix is the C++ root, which is a
# conventional prefix for a standalone install and only happens to coincide
# with the torch package directory under scikit-build-core.
#
# Must be included after FileMirroring.cmake: the valgrind headers picked up
# by the benchmark utilities rule below are copied into the source tree by
# that module.

if(NOT DEFINED TORCH_SRC_DIR)
  set(TORCH_SRC_DIR "${PROJECT_SOURCE_DIR}/torch")
endif()

set(_torch_pkg "${SKBUILD_PLATLIB_DIR}/torch")

# --- torch package data ---

# Type stubs
install(DIRECTORY "${TORCH_SRC_DIR}/"
  DESTINATION "${_torch_pkg}"
  FILES_MATCHING
  PATTERN "*.pyi"
  PATTERN "py.typed"
)

# Benchmark utilities — matches setup.py package_data patterns:
#   utils/benchmark/utils/*.cpp
#   utils/benchmark/utils/valgrind_wrapper/*.cpp
#   utils/benchmark/utils/valgrind_wrapper/*.h
# (*.h files are copied from third_party/ by FileMirroring.cmake; the pattern
# here picks them up if already present.)
install(DIRECTORY "${TORCH_SRC_DIR}/utils/benchmark/utils/"
  DESTINATION "${_torch_pkg}/utils/benchmark/utils"
  FILES_MATCHING PATTERN "*.cpp" PATTERN "*.h"
)

# Model dump utilities
install(FILES
  "${TORCH_SRC_DIR}/utils/model_dump/skeleton.html"
  "${TORCH_SRC_DIR}/utils/model_dump/code.js"
  DESTINATION "${_torch_pkg}/utils/model_dump"
  OPTIONAL
)
install(DIRECTORY "${TORCH_SRC_DIR}/utils/model_dump/"
  DESTINATION "${_torch_pkg}/utils/model_dump"
  FILES_MATCHING PATTERN "*.mjs"
)

# Inductor data files
install(FILES "${TORCH_SRC_DIR}/_inductor/script.ld"
  DESTINATION "${_torch_pkg}/_inductor"
  OPTIONAL
)
install(DIRECTORY "${TORCH_SRC_DIR}/_inductor/codegen/"
  DESTINATION "${_torch_pkg}/_inductor/codegen"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.cpp"
)
install(DIRECTORY "${TORCH_SRC_DIR}/_inductor/kernel/flex/templates/"
  DESTINATION "${_torch_pkg}/_inductor/kernel/flex/templates"
  FILES_MATCHING PATTERN "*.jinja"
)
install(DIRECTORY "${TORCH_SRC_DIR}/_inductor/kernel/templates/"
  DESTINATION "${_torch_pkg}/_inductor/kernel/templates"
  FILES_MATCHING PATTERN "*.jinja"
)

# Export serde data
install(DIRECTORY "${TORCH_SRC_DIR}/_export/serde/"
  DESTINATION "${_torch_pkg}/_export/serde"
  FILES_MATCHING
  PATTERN "*.yaml"
  PATTERN "*.thrift"
)

# AOTI runtime header
install(FILES "${TORCH_SRC_DIR}/csrc/inductor/aoti_runtime/model.h"
  DESTINATION "${_torch_pkg}/csrc/inductor/aoti_runtime"
  OPTIONAL
)

# Generated testing Python module (gitignored so not picked up by scikit-build-core
# package scanning; install explicitly so it ends up in the wheel).
install(FILES "${TORCH_SRC_DIR}/testing/_internal/generated/annotated_fn_args.py"
  DESTINATION "${_torch_pkg}/testing/_internal/generated"
)

# Dynamo data
install(FILES "${TORCH_SRC_DIR}/_dynamo/graph_break_registry.json"
  DESTINATION "${_torch_pkg}/_dynamo"
  OPTIONAL
)

unset(_torch_pkg)
