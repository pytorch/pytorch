# Install non-Python package data.
# Under scikit-build-core this populates the wheel; under setuptools the
# cmake install target copies files into CMAKE_INSTALL_PREFIX (= <root>/torch)
# where setup.py's package_data patterns pick them up.
#
# Destinations are relative to CMAKE_INSTALL_PREFIX which maps to torch/ in
# the wheel (via wheel.install-dir = "torch") or <root>/torch for setuptools.

if(NOT DEFINED TORCH_SRC_DIR)
  set(TORCH_SRC_DIR "${PROJECT_SOURCE_DIR}/torch")
endif()

# --- torch package data ---

# Type stubs
install(DIRECTORY "${TORCH_SRC_DIR}/"
  DESTINATION "."
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
  DESTINATION "utils/benchmark/utils"
  FILES_MATCHING PATTERN "*.cpp" PATTERN "*.h"
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

# Generated testing Python module (gitignored so not picked up by scikit-build-core
# package scanning; install explicitly so it ends up in the wheel).
install(FILES "${TORCH_SRC_DIR}/testing/_internal/generated/annotated_fn_args.py"
  DESTINATION "testing/_internal/generated"
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
