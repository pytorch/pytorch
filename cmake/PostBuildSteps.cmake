# Post-build steps previously handled by setup.py's build_ext.run().
# These run as CMake install(SCRIPT) or install(CODE) commands.

# --- Header wrapping with TORCH_STABLE_ONLY guards ---
# Wrap installed headers so they error when included with TORCH_STABLE_ONLY
# or TORCH_TARGET_VERSION defined. This is done at install time via a script.
install(CODE "
  set(_include_dir \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_INCLUDE_DIR}\")
  if(EXISTS \"\${_include_dir}\")
    message(STATUS \"Wrapping headers with TORCH_STABLE_ONLY guards...\")
    set(_header_extensions h hpp cuh)
    set(_exclude_patterns
      \"torch/headeronly/\"
      \"torch/csrc/stable/\"
      \"torch/csrc/inductor/aoti_torch/c/\"
      \"torch/csrc/inductor/aoti_torch/generated/\"
    )
    set(_wrap_marker \"#if !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)\")

    foreach(_ext IN ITEMS h hpp cuh)
      file(GLOB_RECURSE _headers \"\${_include_dir}/*.\${_ext}\")
      foreach(_header IN LISTS _headers)
        file(RELATIVE_PATH _rel \"\${_include_dir}\" \"\${_header}\")

        # Check exclusion patterns
        set(_excluded FALSE)
        foreach(_pat IN LISTS _exclude_patterns)
          string(FIND \"\${_rel}\" \"\${_pat}\" _pos)
          if(NOT _pos EQUAL -1)
            set(_excluded TRUE)
            break()
          endif()
        endforeach()
        if(_excluded)
          continue()
        endif()

        file(READ \"\${_header}\" _content)
        string(FIND \"\${_content}\" \"\${_wrap_marker}\" _already_wrapped)
        if(_already_wrapped EQUAL 0)
          continue()
        endif()

        set(_wrapped \"\${_wrap_marker}\\n\${_content}\\n#else\\n\")
        string(APPEND _wrapped
          \"#error \\\"This file should not be included when either TORCH_STABLE_ONLY or TORCH_TARGET_VERSION is defined.\\\"\\n\")
        string(APPEND _wrapped
          \"#endif  // !defined(TORCH_STABLE_ONLY) && !defined(TORCH_TARGET_VERSION)\\n\")
        file(WRITE \"\${_header}\" \"\${_wrapped}\")
      endforeach()
    endforeach()
  endif()
")

# --- Compile commands merging ---
# Merge compile_commands.json from build subdirectories.
# Write the script to a file to avoid CMake stripping newlines from multiline
# command arguments when passed through Ninja.
file(WRITE "${CMAKE_BINARY_DIR}/merge_compile_commands.py"
"import json, pathlib, itertools\n\
build = pathlib.Path('${CMAKE_BINARY_DIR}')\n\
ninja = list(build.glob('*compile_commands.json'))\n\
cmake_sub = list((build / 'torch' / 'lib' / 'build').glob('*/compile_commands.json')) if (build / 'torch' / 'lib' / 'build').exists() else []\n\
cmds = [e for f in itertools.chain(ninja, cmake_sub) for e in json.loads(f.read_text())]\n\
for c in cmds:\n\
    if c.get('command', '').startswith('gcc '):\n\
        c['command'] = 'g++ ' + c['command'][4:]\n\
out = pathlib.Path('${PROJECT_SOURCE_DIR}/compile_commands.json')\n\
new = json.dumps(cmds, indent=2)\n\
if not out.exists() or out.read_text() != new:\n\
    out.write_text(new)\n\
")
add_custom_target(merge_compile_commands ALL
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_BINARY_DIR}/merge_compile_commands.py"
  COMMENT "Merging compile_commands.json..."
  VERBATIM
)

# --- License concatenation ---
# Build the bundled license file for wheel distribution.
file(WRITE "${CMAKE_BINARY_DIR}/bundle_licenses.py"
"import sys, pathlib\n\
third_party = pathlib.Path('${PROJECT_SOURCE_DIR}/third_party')\n\
sys.path.insert(0, str(third_party))\n\
from build_bundled import create_bundled\n\
license_file = pathlib.Path('${PROJECT_SOURCE_DIR}/LICENSE')\n\
bsd_text = license_file.read_text()\n\
with license_file.open('a') as f:\n\
    f.write('\\n\\n')\n\
    create_bundled(str(third_party.resolve()), f, include_files=True)\n\
bundled = license_file.read_text()\n\
license_file.write_text(bsd_text)\n\
pathlib.Path('${CMAKE_BINARY_DIR}/LICENSES_BUNDLED.txt').write_text(bundled)\n\
")
add_custom_target(bundle_licenses ALL
  COMMAND "${Python_EXECUTABLE}" "${CMAKE_BINARY_DIR}/bundle_licenses.py"
  COMMENT "Generating bundled license file..."
  VERBATIM
)
install(FILES "${CMAKE_BINARY_DIR}/LICENSES_BUNDLED.txt"
  DESTINATION "."
  RENAME "LICENSE"
  OPTIONAL
)

# --- Windows export library ---
if(WIN32 AND BUILD_PYTHON AND NOT BUILD_LIBTORCH_WHL)
  install(CODE "
    set(_export_lib \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_LIB_DIR}/_C.lib\")
    # The export lib is generated alongside the _C module
    if(EXISTS \"${CMAKE_BINARY_DIR}/torch/csrc/_C.lib\")
      file(INSTALL \"${CMAKE_BINARY_DIR}/torch/csrc/_C.lib\"
           DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_LIB_DIR}\")
    endif()
  ")
endif()

# --- macOS OpenMP embedding ---
if(APPLE AND BUILD_PYTHON)
  install(CODE "
    set(_lib_dir \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_LIB_DIR}\")
    set(_libtorch_cpu \"\${_lib_dir}/libtorch_cpu.dylib\")
    if(EXISTS \"\${_libtorch_cpu}\")
      # Check if libtorch_cpu links to libomp
      execute_process(
        COMMAND otool -L \"\${_libtorch_cpu}\"
        OUTPUT_VARIABLE _otool_out
        ERROR_QUIET
      )
      if(_otool_out MATCHES \"libomp\\\\.dylib|libiomp5\\\\.dylib\")
        # The full OpenMP embedding logic is complex and platform-specific.
        # For now, ensure the rpath includes @loader_path for the bundled lib.
        execute_process(
          COMMAND install_name_tool -add_rpath @loader_path \"\${_libtorch_cpu}\"
          ERROR_QUIET
        )
      endif()
    endif()
  ")
endif()
