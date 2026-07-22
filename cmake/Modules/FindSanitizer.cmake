# Find sanitizers
#
# This module sets the following targets:
#  Sanitizer::address
#  Sanitizer::thread
#  Sanitizer::undefined
#  Sanitizer::leak
#  Sanitizer::memory
include_guard(GLOBAL)

option(UBSAN_FLAGS "additional UBSAN flags" OFF)

get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)

set(_source_code
    [==[
  #include <stdio.h>
  int main() {
  printf("hello world!");
  return 0;
  }
  ]==])

include(CMakePushCheckState)
cmake_push_check_state(RESET)
foreach(sanitizer_name IN ITEMS address thread undefined leak memory)
  if(TARGET Sanitizer::${sanitizer_name})
    continue()
  endif()

  set(CMAKE_REQUIRED_FLAGS
      "-fsanitize=${sanitizer_name};-fno-omit-frame-pointer")
  set(CMAKE_REQUIRED_LINK_OPTIONS "")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_C_COMPILER_ID STREQUAL
                                              "MSVC")
    if(sanitizer_name STREQUAL "address")
      set(CMAKE_REQUIRED_FLAGS "/fsanitize=${sanitizer_name}")
    else()
      continue()
    endif()
  endif()
  set(_asan_rpath_flag "")
  if(sanitizer_name STREQUAL "address")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL
                                                 "Clang")
      list(APPEND CMAKE_REQUIRED_FLAGS "-shared-libasan")
      # -shared-libasan needs libclang_rt.asan-<arch>.so at runtime. On toolchains
      # that keep it in a non-default directory (e.g. a ROCm SDK), the probe binary
      # below cannot load it, check_cxx_source_runs() fails, and ASAN is silently
      # disabled. Add an rpath to the runtime's directory so the probe runs; the
      # same flag is later attached to Sanitizer::address so real targets load it.
      execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libclang_rt.asan-${CMAKE_HOST_SYSTEM_PROCESSOR}.so
        OUTPUT_VARIABLE _asan_runtime_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(_asan_runtime_path AND NOT _asan_runtime_path STREQUAL "libclang_rt.asan-${CMAKE_HOST_SYSTEM_PROCESSOR}.so")
        get_filename_component(_asan_runtime_dir "${_asan_runtime_path}" DIRECTORY)
        set(_asan_rpath_flag "-Wl,-rpath,${_asan_runtime_dir}")
        list(APPEND CMAKE_REQUIRED_LINK_OPTIONS "${_asan_rpath_flag}")
      endif()
    endif()
  endif()
  if(sanitizer_name STREQUAL "undefined" AND UBSAN_FLAGS)
    list(APPEND CMAKE_REQUIRED_FLAGS "${UBSAN_FLAGS}")
  endif()
  if(sanitizer_name STREQUAL "memory")
    list(APPEND CMAKE_REQUIRED_FLAGS "-fsanitize-memory-track-origins=2")
  endif()

  set(CMAKE_REQUIRED_QUIET ON)
  set(_run_res 0)
  include(CheckCSourceRuns)
  include(CheckCXXSourceRuns)
  foreach(lang IN LISTS languages)
    if(lang STREQUAL C)
      check_c_source_runs("${_source_code}"
                        __${lang}_${sanitizer_name}_res)
      if(__${lang}_${sanitizer_name}_res)
        set(_run_res 1)
      endif()
    endif()
    if(lang STREQUAL CXX)
      check_cxx_source_runs("${_source_code}"
                        __${lang}_${sanitizer_name}_res)
      if(__${lang}_${sanitizer_name}_res)
        set(_run_res 1)
      endif()
    endif()
  endforeach()
  if(_run_res)
    add_library(Sanitizer::${sanitizer_name} INTERFACE IMPORTED GLOBAL)
    target_compile_options(
      Sanitizer::${sanitizer_name}
      INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:${CMAKE_REQUIRED_FLAGS}>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>>:${CMAKE_REQUIRED_FLAGS}>
    )
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND NOT CMAKE_C_COMPILER_ID
                                                     STREQUAL "MSVC")
      target_link_options(
        Sanitizer::${sanitizer_name}
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:${CMAKE_REQUIRED_FLAGS}>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>>:${CMAKE_REQUIRED_FLAGS}>
      )
    else()
      target_link_options(
        Sanitizer::${sanitizer_name}
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:/INCREMENTAL:NO>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>>:/INCREMENTAL:NO>
      )
    endif()

    if(sanitizer_name STREQUAL "address")
      # Include HIP language so HIP-side TUs update libstdc++ container
      # annotations consistently with cpu-side; without this the cpu-side
      # _M_realloc_insert sets up redzones that the inlined HIP-side
      # emplace_back fast path doesn't update, producing spurious
      # container-overflow reports.
      target_compile_definitions(
        Sanitizer::${sanitizer_name}
        INTERFACE
          $<$<AND:$<COMPILE_LANGUAGE:CXX,HIP>,$<BOOL:$__CXX_${sanitizer_name}_res>>:_GLIBCXX_SANITIZE_VECTOR>
          $<$<AND:$<COMPILE_LANGUAGE:CXX,HIP>,$<BOOL:$__CXX_${sanitizer_name}_res>>:_GLIBCXX_SANITIZE_STD_ALLOCATOR>
      )
      # Work around a Clang ASAN instrumentation issue where the global
      # metadata references the original (potentially unaligned) global
      # instead of the __sanitized_padded_global. Without private aliases,
      # globals can end up at non-8-byte-aligned offsets, causing an
      # unconditional alignment check failure in the ASAN runtime that is
      # not suppressible via detect_odr_violation=0. Especially needed
      # under ROCm Clang where ASAN + UBSan together exhibit the issue.
      if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
        target_compile_options(
          Sanitizer::${sanitizer_name}
          INTERFACE
          "SHELL:-mllvm -asan-use-private-alias=1"
        )
      endif()
      target_link_options(
        Sanitizer::${sanitizer_name}
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>,$<CXX_COMPILER_ID:GNU>>:-lasan>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>,$<C_COMPILER_ID:GNU>>:-lasan>
      )
      # Carry the rpath to the shared ASAN runtime (set above when probing with
      # -shared-libasan) onto real targets, so libraries and executables linking
      # Sanitizer::address can load libclang_rt.asan-<arch>.so from its
      # non-default toolchain directory at runtime.
      if(_asan_rpath_flag)
        target_link_options(Sanitizer::${sanitizer_name} INTERFACE "${_asan_rpath_flag}")
      endif()
    endif()
    if(sanitizer_name STREQUAL "undefined")
      target_link_options(
        Sanitizer::${sanitizer_name}
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>,$<CXX_COMPILER_ID:GNU>>:-lubsan>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>,$<C_COMPILER_ID:GNU>>:-lubsan>
      )
    endif()
  endif()
endforeach()

cmake_pop_check_state()
