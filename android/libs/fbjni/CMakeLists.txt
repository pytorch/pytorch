cmake_minimum_required(VERSION 3.5)
project(fbjni CXX)

file(GLOB fbjni_SOURCES
  cxx/fbjni/*.cpp
  cxx/fbjni/detail/*.cpp
  cxx/lyra/*.cpp
)
add_library(fbjni SHARED
  ${fbjni_SOURCES}
)

target_compile_options(fbjni PRIVATE
  -fno-omit-frame-pointer
  -fexceptions
  -O3
  -Wall
  -std=c++11
  -frtti
  -ffunction-sections
  -DDISABLE_CPUCAP
  -DDISABLE_XPLAT
  -DNDEBUG
)

target_include_directories(fbjni PUBLIC
  cxx
)

target_link_libraries(fbjni
  android
  log
)
