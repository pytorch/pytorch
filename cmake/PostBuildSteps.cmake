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
    COMMAND_ERROR_IS_FATAL ANY
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

# --- Runtime DLL bundling (Windows) ---
# The old CI scripts (copy.bat / copy_cpu.bat) copied runtime DLLs into the
# source tree before setuptools ran.  With scikit-build-core the wheel is
# built from the cmake install prefix, so we install them via cmake instead.
if(WIN32 AND BUILD_PYTHON)
  # OpenMP runtime (libiomp5md.dll) - required by torch_cpu.dll when MKL
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

  # libuv (uv.dll) - required by torch distributed (gloo transport).
  # libuv_DLL_PATH is an optional CI hint (forwarded via EnvVarForwarding.cmake);
  # fall back to libuv_ROOT/bin/uv.dll which Windows CI sets.
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

  # CUDA runtime DLLs - only for CUDA builds.
  if(USE_CUDA AND CUDA_TOOLKIT_ROOT_DIR)
    # CUDA 13+ moves DLLs to an architecture-specific bin directory.
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
      set(_cuda_windows_arch "arm64")
    else()
      set(_cuda_windows_arch "x64")
    endif()

    if(IS_DIRECTORY "${CUDA_TOOLKIT_ROOT_DIR}/bin/${_cuda_windows_arch}")
      set(_cuda_bin "${CUDA_TOOLKIT_ROOT_DIR}/bin/${_cuda_windows_arch}")
    else()
      set(_cuda_bin "${CUDA_TOOLKIT_ROOT_DIR}/bin")
    endif()
    # CUPTI and its nvperf helper are not where they used to be. Through 13.2
    # they ship under extras/CUPTI/lib64; 13.4 drops that tree and puts them
    # beside the other runtime DLLs, and Windows Arm64 uses
    # extras/CUPTI/lib/<arch>. Search all of them and take whichever exists.
    # The filenames are unchanged (13.4's is cupti64_2026.1.1.dll, which the
    # same glob matches) -- only the directory moved.
    set(_cupti_dirs
      "${_cuda_bin}"
      "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib/${_cuda_windows_arch}"
    )
    # lib64 is the legacy x64-only tree: on Arm64 it holds x64 binaries.
    if(NOT _cuda_windows_arch STREQUAL "arm64")
      list(APPEND _cupti_dirs "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64")
    endif()
    set(_cupti_patterns "")
    set(_nvperf_patterns "")
    foreach(_dir ${_cupti_dirs})
      list(APPEND _cupti_patterns "${_dir}/cupti64_*.dll")
      list(APPEND _nvperf_patterns "${_dir}/nvperf_host*.dll")
    endforeach()

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
      ${_cupti_patterns}
      ${_nvperf_patterns}
    )
    foreach(_pattern ${_cuda_dll_patterns})
      file(GLOB _dlls "${_pattern}")
      if(_dlls)
        install(FILES ${_dlls} DESTINATION "${TORCH_INSTALL_LIB_DIR}")
      endif()
    endforeach()

    # A pattern matching nothing is exactly how a 13.4 wheel shipped without
    # CUPTI and still built green: file(GLOB) is silent and install() is simply
    # skipped. libkineto import-links CUPTI on Windows, so that wheel could not
    # be imported at all -- it failed with WinError 126 naming shm.dll, which is
    # merely the first entry in torch/__init__.py's load loop whose dependency
    # chain reaches torch_cuda.dll. Fail the build instead.
    if(USE_KINETO)
      file(GLOB _cupti_dlls ${_cupti_patterns})
      if(NOT _cupti_dlls)
        string(REPLACE ";" "\n  " _cupti_dirs_msg "${_cupti_dirs}")
        message(FATAL_ERROR
          "USE_KINETO is ON but no CUPTI DLL was found under:\n  "
          "${_cupti_dirs_msg}\n"
          "torch_cuda.dll import-links CUPTI, so the wheel would fail to "
          "import. Point CUDA_TOOLKIT_ROOT_DIR at a toolkit that ships CUPTI, "
          "or configure with USE_KINETO=OFF.")
      endif()
    endif()

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

# --- macOS OpenMP embedding ---
# Copy libomp.dylib / libiomp5.dylib into the wheel and fix rpaths so the
# wheel is self-contained (replicates setup.py's _embed_libomp).
# Gated on USE_OPENMP as well as OpenMP_FOUND so that a user-forced
# USE_OPENMP=OFF doesn't ship an orphan libomp.dylib in the wheel.
if(APPLE AND BUILD_PYTHON AND USE_OPENMP AND OpenMP_FOUND)
  # OpenMP_libomp_LIBRARY is set by our FindOpenMP module to the full path
  # of the OpenMP shared library (e.g. /path/to/libomp.dylib).
  if(OpenMP_libomp_LIBRARY AND EXISTS "${OpenMP_libomp_LIBRARY}")
    install(FILES "${OpenMP_libomp_LIBRARY}"
            DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    # Install omp.h so Inductor's C++ backend can find it at runtime.
    # FindOpenMP doesn't export an include-dir variable; OpenMP_C_FLAGS
    # carries -I<path> on macOS, so parse it for the header location.
    if(OpenMP_C_FLAGS)
      separate_arguments(_omp_c_flags UNIX_COMMAND "${OpenMP_C_FLAGS}")
      foreach(_flag IN LISTS _omp_c_flags)
        if(_flag MATCHES "^-I(.+)$")
          set(_omp_h_dir "${CMAKE_MATCH_1}")
          if(EXISTS "${_omp_h_dir}/omp.h")
            install(FILES "${_omp_h_dir}/omp.h"
                    DESTINATION "${TORCH_INSTALL_INCLUDE_DIR}")
            break()
          endif()
        endif()
      endforeach()
    endif()
    # Fix libtorch_cpu's load command and rpaths so the bundled libomp is
    # the only one resolved at runtime. See tools/embed_libomp_macos.py
    # for the two-case logic (homebrew abs-path vs. conda @rpath build).
    install(CODE "
      execute_process(
        COMMAND \"${_python_exe}\"
          \"${_project_src}/tools/embed_libomp_macos.py\"
          --libomp-path \"${OpenMP_libomp_LIBRARY}\"
          --lib-dir \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_LIB_DIR}\"
        COMMAND_ERROR_IS_FATAL ANY
      )
    ")
  endif()
endif()
