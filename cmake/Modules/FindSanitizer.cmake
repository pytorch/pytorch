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

  set(CMAKE_REQUIRED_QUIET ON)
  foreach(lang IN LISTS languages)
    set(SANITIZER_FLAGS "-fsanitize=${sanitizer_name};-fno-omit-frame-pointer")
    if(sanitizer_name STREQUAL "address")
      if(CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC")
        set(SANITIZER_FLAGS "/fsanitize=${sanitizer_name}")
      endif()
    endif()
    if(sanitizer_name STREQUAL "undefined" AND UBSAN_FLAGS)
      list(APPEND SANITIZER_FLAGS "${UBSAN_FLAGS}")
    endif()
    if(sanitizer_name STREQUAL "memory")
      list(APPEND SANITIZER_FLAGS "-fsanitize-memory-track-origins=2")
    endif()
    string(REPLACE ";" " " CMAKE_REQUIRED_FLAGS "${SANITIZER_FLAGS}")

    if(lang STREQUAL C)
      include(CheckCSourceRuns)
      check_c_source_runs("${_source_code}" __res)
    elseif(lang STREQUAL CXX)
      include(CheckCXXSourceRuns)
      check_cxx_source_runs("${_source_code}" __res)
    else()
      continue()
    endif()
    if(NOT __res)
      continue()
    endif()
    if(NOT TARGET Sanitizer::${sanitizer_name})
      add_library(Sanitizer::${sanitizer_name} INTERFACE IMPORTED GLOBAL)
    endif()
    foreach(SANITIZER_FLAG IN LISTS SANITIZER_FLAGS)
      target_compile_options(
        Sanitizer::${sanitizer_name}
        INTERFACE $<$<COMPILE_LANGUAGE:${lang}>:${SANITIZER_FLAG}>)
      target_link_options(
        Sanitizer::${sanitizer_name}
        INTERFACE
        $<AND:$<NOT:$<${lang}_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:${lang}>>:${SANITIZER_FLAG}>
      )
    endforeach()
    target_link_options(
      Sanitizer::${sanitizer_name}
      INTERFACE
      $<AND:$<${lang}_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:${lang}>>:/INCREMENTAL:NO>
    )

    if(sanitizer_name STREQUAL "address")
      if(lang STREQUAL CXX)
        target_compile_definitions(
          Sanitizer::${sanitizer_name}
          INTERFACE
            $<$<COMPILE_LANGUAGE:${lang}>:_GLIBCXX_SANITIZE_VECTOR>
            $<$<COMPILE_LANGUAGE:${lang}>:_GLIBCXX_SANITIZE_STD_ALLOCATOR>)
      endif()
      target_link_options(
        Sanitizer::${sanitizer_name} INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:${lang}>,$<${lang}_COMPILER_ID:GNU>>:-lasan>)
      target_link_options(
        Sanitizer::${sanitizer_name}
        INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:${lang}>,$<${lang}_COMPILER_ID:Clang>>:-shared-libasan>
      )
    endif()
    if(sanitizer_name STREQUAL "undefined")
      target_link_options(
        Sanitizer::${sanitizer_name} INTERFACE
        $<$<AND:$<COMPILE_LANGUAGE:${lang}>,$<${lang}_COMPILER_ID:GNU>>:-lubsan>
      )
    endif()
  endforeach()
endforeach()

cmake_pop_check_state()
