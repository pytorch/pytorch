# Post-build steps previously handled by setup.py's build_ext.run().
# These run as CMake install(SCRIPT) or install(CODE) commands.

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()
if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include)
endif()

# Normalize paths to forward slashes so they survive embedding inside
# install(CODE "...") strings on Windows (backslashes are escape chars).
file(TO_CMAKE_PATH "${Python_EXECUTABLE}" _python_exe)
file(TO_CMAKE_PATH "${PROJECT_SOURCE_DIR}" _project_src)
file(TO_CMAKE_PATH "${CMAKE_BINARY_DIR}" _cmake_bindir)

# --- Header wrapping with TORCH_STABLE_ONLY guards ---
# Wrap installed headers so they error when included with TORCH_STABLE_ONLY
# or TORCH_TARGET_VERSION defined. This is done at install time via a script.
install(CODE "
  execute_process(
    COMMAND \"${_python_exe}\"
      \"${_project_src}/tools/wrap_headers.py\"
      \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_INCLUDE_DIR}\"
  )
")

# --- Compile commands merging ---
# Merge compile_commands.json from build subdirectories.
add_custom_target(merge_compile_commands ALL
  COMMAND "${_python_exe}"
    "${_project_src}/tools/merge_compile_commands.py"
    "${_cmake_bindir}" "${_project_src}"
  COMMENT "Merging compile_commands.json..."
  VERBATIM
)

# --- License concatenation ---
# Build the bundled license file for wheel distribution.
add_custom_target(bundle_licenses ALL
  COMMAND "${_python_exe}"
    "${_project_src}/tools/bundle_licenses.py"
    "${_project_src}" "${_cmake_bindir}/LICENSES_BUNDLED.txt"
  COMMENT "Generating bundled license file..."
  VERBATIM
)
install(FILES "${_cmake_bindir}/LICENSES_BUNDLED.txt"
  DESTINATION "."
  RENAME "LICENSE"
  OPTIONAL
)
