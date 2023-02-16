# Find google sanitizers
#
# This module sets the following targets: GoogleSanitizer::address
# GoogleSanitizer::thread GoogleSanitizer::undefined GoogleSanitizer::leak
# GoogleSanitizer::memory
include_guard(GLOBAL)

option(ADDITIONAL_UBSAN_OPTIONS "additional UBSAN options" OFF)

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
  if(TARGET GoogleSanitizer::${sanitizer_name})
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
  if(sanitizer_name STREQUAL "undefined" AND ADDITIONAL_UBSAN_OPTIONS)
    list(APPEND CMAKE_REQUIRED_FLAGS "${ADDITIONAL_UBSAN_OPTIONS}")
  endif()
  if(sanitizer_name STREQUAL "memory")
    list(APPEND CMAKE_REQUIRED_FLAGS "-fsanitize-memory-track-origins=2")
  endif()

  set(CMAKE_REQUIRED_QUIET ON)
  set(_run_res 0)
  include(CheckSourceRuns)
  foreach(lang IN LISTS languages)
    if(lang STREQUAL CXX OR lang STREQUAL C)
      check_source_runs(${lang} "${_source_code}"
                        __${lang}_${sanitizer_name}_res)
      if(__${lang}_${sanitizer_name}_res)
        set(_run_res 1)
      endif()
    endif()
  endforeach()
  if(_run_res)
    add_library(GoogleSanitizer::${sanitizer_name} INTERFACE IMPORTED GLOBAL)
    target_compile_options(
      GoogleSanitizer::${sanitizer_name}
      INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:${CMAKE_REQUIRED_FLAGS}>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>>:${CMAKE_REQUIRED_FLAGS}>
    )
    target_link_options(
      GoogleSanitizer::${sanitizer_name}
      INTERFACE
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:${CMAKE_REQUIRED_FLAGS}>
      $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>>:${CMAKE_REQUIRED_FLAGS}>
    )

    if(sanitizer_name STREQUAL "address")
      target_compile_definitions(
        GoogleSanitizer::${sanitizer_name}
        INTERFACE
          $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:_GLIBCXX_SANITIZE_VECTOR>
          $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>>:_GLIBCXX_SANITIZE_STD_ALLOCATOR>
      )
      target_link_options(
        GoogleSanitizer::${sanitizer_name}
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>,$<CXX_COMPILER_ID:GNU>>:-lasan>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>,$<C_COMPILER_ID:GNU>>:-lasan>
      )
    endif()
    if(sanitizer_name STREQUAL "undefined")
      target_link_options(
        GoogleSanitizer::${sanitizer_name}
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:$__CXX_${sanitizer_name}_res>,$<CXX_COMPILER_ID:GNU>>:-lubsan>
        $<$<AND:$<COMPILE_LANGUAGE:C>,$<BOOL:$__C_${sanitizer_name}_res>,$<C_COMPILER_ID:GNU>>:-lubsan>
      )
    endif()
  endif()
endforeach()

cmake_pop_check_state()
