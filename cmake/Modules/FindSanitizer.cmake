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
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_C_COMPILER_ID STREQUAL
                                              "MSVC")
    if(sanitizer_name STREQUAL "address")
      set(CMAKE_REQUIRED_FLAGS "/fsanitize=${sanitizer_name}")
    else()
      continue()
    endif()
  endif()
  if(sanitizer_name STREQUAL "address")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL
                                                 "Clang")
      list(APPEND CMAKE_REQUIRED_FLAGS "-shared-libasan")
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
      target_compile_definitions(
        Sanitizer::${sanitizer_name}
        INTERFACE
          $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:_GLIBCXX_SANITIZE_VECTOR>
          $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:_GLIBCXX_SANITIZE_STD_ALLOCATOR>
      )
      target_link_options(
        Sanitizer::${sanitizer_name}
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>,$<CXX_COMPILER_ID:GNU>>:-lasan>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>,$<C_COMPILER_ID:GNU>>:-lasan>
      )
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
