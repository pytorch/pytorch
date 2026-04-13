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

# --- Windows export library ---
if(WIN32 AND BUILD_PYTHON AND NOT BUILD_LIBTORCH_WHL)
  install(CODE "
    if(EXISTS \"${_cmake_bindir}/torch/csrc/_C.lib\")
      file(INSTALL \"${_cmake_bindir}/torch/csrc/_C.lib\"
           DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_LIB_DIR}\")
    endif()
  ")
endif()

# --- Runtime DLL bundling (Windows) ---
# The old CI scripts (copy.bat / copy_cpu.bat) copied runtime DLLs into the
# source tree before setuptools ran.  With scikit-build-core the wheel is
# built from the cmake install prefix, so we install them via cmake instead.
if(WIN32 AND BUILD_PYTHON)
  # OpenMP runtime (libiomp5md.dll) — required by torch_cpu.dll when MKL
  # threading uses Intel OpenMP.
  if(MKL_OPENMP_LIBRARY AND MKL_OPENMP_LIBRARY MATCHES "libiomp5md\\.lib$")
    get_filename_component(_omp_lib_dir "${MKL_OPENMP_LIBRARY}" DIRECTORY)
    get_filename_component(_omp_prefix "${_omp_lib_dir}" DIRECTORY)
    # The DLL lives in bin/ next to the lib/ that contains the import library.
    set(_omp_dll "${_omp_prefix}/bin/libiomp5md.dll")
    if(EXISTS "${_omp_dll}")
      install(FILES "${_omp_dll}" DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    else()
      # Fallback: DLL in the same directory as the import library.
      file(GLOB _omp_dll_fallback "${_omp_lib_dir}/libiomp5md.dll")
      if(_omp_dll_fallback)
        install(FILES ${_omp_dll_fallback} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
      endif()
    endif()
    # Also install the stubs library if present (libiompstubs5md.dll).
    file(GLOB _omp_stubs "${_omp_prefix}/bin/libiompstubs5md.dll")
    if(NOT _omp_stubs)
      file(GLOB _omp_stubs "${_omp_lib_dir}/libiompstubs5md.dll")
    endif()
    if(_omp_stubs)
      install(FILES ${_omp_stubs} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    endif()
  endif()

  # libuv (uv.dll) — required by torch distributed (gloo transport).
  if(USE_DISTRIBUTED)
    if(libuv_DLL_PATH AND EXISTS "${libuv_DLL_PATH}")
      install(FILES "${libuv_DLL_PATH}" DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    elseif(DEFINED ENV{libuv_ROOT})
      file(GLOB _uv_dll "$ENV{libuv_ROOT}/bin/uv.dll")
      if(_uv_dll)
        install(FILES ${_uv_dll} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
      endif()
    endif()
  endif()

  # CUDA runtime DLLs — only for CUDA builds.
  if(USE_CUDA AND CUDA_TOOLKIT_ROOT_DIR)
    # CUDA 13+ moves DLLs to bin/x64.
    if(IS_DIRECTORY "${CUDA_TOOLKIT_ROOT_DIR}/bin/x64")
      set(_cuda_bin "${CUDA_TOOLKIT_ROOT_DIR}/bin/x64")
    else()
      set(_cuda_bin "${CUDA_TOOLKIT_ROOT_DIR}/bin")
    endif()
    set(_cuda_dll_patterns
      "${_cuda_bin}/cusparse*64_*.dll"
      "${_cuda_bin}/cublas*64_*.dll"
      "${_cuda_bin}/cudart*64_*.dll"
      "${_cuda_bin}/curand*64_*.dll"
      "${_cuda_bin}/cufft*64_*.dll"
      "${_cuda_bin}/cusolver*64_*.dll"
      "${_cuda_bin}/nvrtc*64_*.dll"
      "${_cuda_bin}/nvJitLink_*.dll"
      "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudnn*64_*.dll"
      "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64/cupti64_*.dll"
      "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64/nvperf_host*.dll"
    )
    foreach(_pattern ${_cuda_dll_patterns})
      file(GLOB _dlls "${_pattern}")
      if(_dlls)
        install(FILES ${_dlls} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
      endif()
    endforeach()

    # NvToolsExt (legacy, may not exist on all systems).
    set(_nvtoolsext "C:/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/nvToolsExt64_1.dll")
    if(EXISTS "${_nvtoolsext}")
      install(FILES "${_nvtoolsext}" DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    endif()

    # zlibwapi (needed by some CUDA libraries).
    if(EXISTS "C:/Windows/System32/zlibwapi.dll")
      install(FILES "C:/Windows/System32/zlibwapi.dll"
              DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    endif()
  endif()
endif()
