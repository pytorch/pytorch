# Install non-Python package data into the wheel.
# This replaces setup.py's package_data configuration.
#
# Files installed to CMAKE_INSTALL_PREFIX end up under torch/ in the wheel
# (via wheel.install-dir = "torch"). Files that go into other packages
# (e.g., torchgen) use SKBUILD_PLATLIB_DIR.

if(NOT DEFINED SKBUILD_PLATLIB_DIR)
  set(SKBUILD_PLATLIB_DIR "${CMAKE_INSTALL_PREFIX}")
endif()

# --- torch package data ---

# Type stubs
install(DIRECTORY "${TORCH_SRC_DIR}/"
  DESTINATION "."
  FILES_MATCHING
  PATTERN "*.pyi"
  PATTERN "py.typed"
)

# Benchmark utilities
install(FILES
  "${TORCH_SRC_DIR}/utils/benchmark/utils/valgrind_wrapper/timer_callgrind_template.cpp"
  DESTINATION "utils/benchmark/utils/valgrind_wrapper"
  OPTIONAL
)

# Model dump utilities
install(FILES
  "${TORCH_SRC_DIR}/utils/model_dump/skeleton.html"
  "${TORCH_SRC_DIR}/utils/model_dump/code.js"
  DESTINATION "utils/model_dump"
  OPTIONAL
)
install(DIRECTORY "${TORCH_SRC_DIR}/utils/model_dump/"
  DESTINATION "utils/model_dump"
  FILES_MATCHING PATTERN "*.mjs"
)

# Inductor data files
install(FILES "${TORCH_SRC_DIR}/_inductor/script.ld"
  DESTINATION "_inductor"
  OPTIONAL
)
install(DIRECTORY "${TORCH_SRC_DIR}/_inductor/codegen/"
  DESTINATION "_inductor/codegen"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.cpp"
)
install(DIRECTORY "${TORCH_SRC_DIR}/_inductor/kernel/flex/templates/"
  DESTINATION "_inductor/kernel/flex/templates"
  FILES_MATCHING PATTERN "*.jinja"
)
install(DIRECTORY "${TORCH_SRC_DIR}/_inductor/kernel/templates/"
  DESTINATION "_inductor/kernel/templates"
  FILES_MATCHING PATTERN "*.jinja"
)

# Export serde data
install(DIRECTORY "${TORCH_SRC_DIR}/_export/serde/"
  DESTINATION "_export/serde"
  FILES_MATCHING
  PATTERN "*.yaml"
  PATTERN "*.thrift"
)

# AOTI runtime header
install(FILES "${TORCH_SRC_DIR}/csrc/inductor/aoti_runtime/model.h"
  DESTINATION "csrc/inductor/aoti_runtime"
  OPTIONAL
)

# Dynamo data
install(FILES "${TORCH_SRC_DIR}/_dynamo/graph_break_registry.json"
  DESTINATION "_dynamo"
  OPTIONAL
)
install(FILES "${TORCH_SRC_DIR}/tools/dynamo/gb_id_mapping.py"
  DESTINATION "tools/dynamo"
  OPTIONAL
)

# --- torchgen package data ---
install(DIRECTORY "${PROJECT_SOURCE_DIR}/torchgen/packaged/"
  DESTINATION "${SKBUILD_PLATLIB_DIR}/torchgen/packaged"
  PATTERN "*.pyc" EXCLUDE
  PATTERN "*.pyo" EXCLUDE
)
