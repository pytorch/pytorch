# Find sanitizers
#
# This module sets the following targets:
#  Sanitizer::address
#  Sanitizer::thread
#  Sanitizer::undefined
#  Sanitizer::memory
#  Sanitizer::type
include_guard(GLOBAL)

option(ASAN_FLAGS "additional ASAN flags" "")
option(UBSAN_FLAGS "additional UBSAN flags" "")
option(MSAN_FLAGS "additional MSAN flags" "-fsanitize-memory-track-origins=2")

get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)

set(_source_code
    [==[
  #include <stdio.h>
  int main() {
  printf("hello world!");
  return 0;
  }
  ]==])

set(_bug_address_code
    [==[
#include <stdlib.h>
int main(int argc, char **argv) {
  int *array = (int*)malloc(100*sizeof(int));
  array[0] = 0;
  int res = array[argc + 100];  // BOOM
  free(array);
  return res;
}
]==])

set(_bug_undefined_code
    [==[
int main(int argc, char **argv) {
  int k = 0x7fffffff;
  k += argc;
  return 0;
}
]==])

set(_bug_thread_code
    [==[
#include <pthread.h>
#include <stdio.h>
#include <string>
#include <map>

typedef std::map<std::string, std::string> map_t;

void *threadfunc(void *p) {
  map_t& m = *(map_t*)p;
  m["foo"] = "bar";
  return 0;
}

int main() {
  map_t m;
  pthread_t t;
  pthread_create(&t, 0, threadfunc, &m);
  printf("foo=%s\n", m["foo"].c_str());
  pthread_join(t, 0);
}
]==])

set(_bug_memory_code
    [==[
int main(int argc, char** argv) {
  int* a = new int[10];
  a[5] = 0;
  volatile int b = a[argc];
  if (b)
    printf("xx\n");
  return 0;
}
]==])

include(CMakePushCheckState)
foreach(lang IN LISTS languages)
  if(lang STREQUAL C OR lang STREQUAL CXX)
    include(CheckSourceCompiles)
    include(CheckSourceRuns)
  else()
    continue()
  endif()
  foreach(sanitizer_name IN ITEMS address thread undefined memory type)
    if(TARGET Sanitizer::${sanitizer_name}_${lang})
      continue()
    endif()
    if(CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC")
      if(sanitizer_name STREQUAL "address")
        set(SANITIZER_FLAGS "/fsanitize=${sanitizer_name}")
      else()
        continue()
      endif()
    else()
      set(SANITIZER_FLAGS
          "-fsanitize=${sanitizer_name};-fno-omit-frame-pointer")
    endif()
    if(sanitizer_name STREQUAL "address" AND ASAN_FLAGS)
      list(APPEND SANITIZER_FLAGS "${ASAN_FLAGS}")
    endif()
    if(sanitizer_name STREQUAL "thread" AND TSAN_FLAGS)
      list(APPEND SANITIZER_FLAGS "${TSAN_FLAGS}")
    endif()
    if(sanitizer_name STREQUAL "undefined" AND UBSAN_FLAGS)
      list(APPEND SANITIZER_FLAGS "${UBSAN_FLAGS}")
    endif()
    if(sanitizer_name STREQUAL "memory" AND MSAN_FLAGS)
      list(APPEND SANITIZER_FLAGS "${MSAN_FLAGS}")
    endif()
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_QUIET ON)
    string(REPLACE ";" " " CMAKE_REQUIRED_FLAGS "${SANITIZER_FLAGS}")

    set(SANITIZER_LINK_FLAGS)
    if(CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC")
      list(APPEND SANITIZER_LINK_FLAGS "/INCREMENTAL:NO")
    else()
      list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=${sanitizer_name}")
    endif()
    set(CMAKE_REQUIRED_LINK_OPTIONS "${SANITIZER_LINK_FLAGS}")

    unset(__res CACHE)
    if(CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC")
      check_source_compiles(${lang} "${_source_code}" __res)
    else()
      check_source_runs(${lang} "${_source_code}" __res)
    endif()
    if(NOT __res)
      # no memory sanitizer is common
      if(NOT sanitizer_name STREQUAL "memory")
        message(WARNING "Can't find ${sanitizer_name} in ${lang}")
      endif()
      cmake_pop_check_state()
      continue()
    endif()

    unset(__res CACHE)
    if(NOT CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC")
      set(CMAKE_REQUIRED_FLAGS
          "${CMAKE_REQUIRED_FLAGS} -fno-sanitize-recover=all")
      check_source_runs(${lang} "${_bug_${sanitizer_name}_code}" __res)
      if(__res)
        message(
          WARNING
            "Buffer overflow bug is not detected in ${lang} ${sanitizer_name}")
        cmake_pop_check_state()
        continue()
      endif()
    endif()

    add_library(Sanitizer::${sanitizer_name}_${lang} INTERFACE IMPORTED GLOBAL)
    if(NOT TARGET Sanitizer::${sanitizer_name})
      add_library(Sanitizer::${sanitizer_name} INTERFACE IMPORTED GLOBAL)
    endif()
    target_link_libraries(Sanitizer::${sanitizer_name}
                          INTERFACE Sanitizer::${sanitizer_name}_${lang})
    foreach(SANITIZER_FLAG IN LISTS SANITIZER_FLAGS)
      target_compile_options(
        Sanitizer::${sanitizer_name}_${lang}
        INTERFACE $<$<COMPILE_LANGUAGE:${lang}>:${SANITIZER_FLAG}>)
    endforeach()
    foreach(SANITIZER_FLAG IN LISTS SANITIZER_LINK_FLAGS)
      target_link_options(Sanitizer::${sanitizer_name}_${lang} INTERFACE
                          $<$<COMPILE_LANGUAGE:${lang}>:${SANITIZER_FLAG}>)
    endforeach()

    if(CMAKE_${lang}_COMPILER_ID STREQUAL "Clang")
      target_compile_options(
        Sanitizer::${sanitizer_name}_${lang}
        INTERFACE $<$<COMPILE_LANGUAGE:${lang}>:-shared-libsan>)
    endif()

    if(sanitizer_name STREQUAL "address" AND lang STREQUAL CXX)
      if(CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC")
        target_compile_definitions(
          Sanitizer::${sanitizer_name}_${lang}
          INTERFACE $<$<COMPILE_LANGUAGE:${lang}>:_DISABLE_VECTOR_ANNOTATION>
                    $<$<COMPILE_LANGUAGE:${lang}>:_DISABLE_STRING_ANNOTATION>)
      else()
        target_compile_definitions(
          Sanitizer::${sanitizer_name}_${lang}
          INTERFACE
            $<$<COMPILE_LANGUAGE:${lang}>:_GLIBCXX_SANITIZE_VECTOR>
            $<$<COMPILE_LANGUAGE:${lang}>:_GLIBCXX_SANITIZE_STD_ALLOCATOR>)
      endif()
    endif()
    cmake_pop_check_state()
  endforeach()
endforeach()
