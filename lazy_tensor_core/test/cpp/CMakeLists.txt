cmake_minimum_required(VERSION 3.0)

set(GTEST_DIR "${CMAKE_BINARY_DIR}/gtest")

get_filename_component(PTLTC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(PT_DIR "${PTLTC_DIR}/.." ABSOLUTE)

file(GLOB PTLTC_LIBDIRS "${PTLTC_DIR}/build/lib.*")
list(GET PTLTC_LIBDIRS 0 PTLTC_LIBDIR)
message("Selected PT/LTC library folder ${PTLTC_LIBDIR}")

execute_process(COMMAND
  python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE PT_CXX_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

project(ptltc_test)

find_package(PythonLibs)
set(Torch_DIR "${PT_DIR}/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

include(ExternalProject)
set_directory_properties(PROPERTIES EP_PREFIX "${GTEST_DIR}")

ExternalProject_Add(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG 6f5fd0d7199b9a19faa
  SOURCE_DIR "${GTEST_DIR}/src/googletest-src"
  BINARY_DIR "${GTEST_DIR}/src/googletest-build"
  # Disable install step
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  CMAKE_ARGS
    "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${PT_CXX_ABI} -Wno-error=maybe-uninitialized")

ExternalProject_Get_Property(googletest SOURCE_DIR)

set(TORCH_LTC_TEST_SOURCES
  cpp_test_util.cpp
  main.cpp
  metrics_snapshot.cpp
  test_async_task.cpp
  test_ir.cpp
  test_mayberef.cpp
  test_ltc_util_cache.cpp
  test_aten_ltc_ts_tensor.cpp
  torch_ltc_ts_test.cpp
)

add_executable(test_ptltc ${TORCH_LTC_TEST_SOURCES})

set(TGT_OPTS
  -D_GLIBCXX_USE_CXX11_ABI=${PT_CXX_ABI}
  -Wno-sign-compare
  -Wno-deprecated-declarations
  -Wno-return-type
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # The -fsized-deallocation is required for Clang to prevent an error on
  # pytorch pybind11 about the operator delete being called inappropriately.
  list(APPEND TGT_OPTS
    -Wno-macro-redefined
    -Wno-return-std-move
    -fsized-deallocation)
endif()

target_compile_options(test_ptltc PRIVATE ${TGT_OPTS})

target_include_directories(
  test_ptltc
  PRIVATE
  "${PTLTC_DIR}"
  "${PTLTC_DIR}/torch_ltc/csrc"
)
target_include_directories(
  test_ptltc
  SYSTEM PUBLIC
  "${SOURCE_DIR}/googletest/include"
  "${PYTHON_INCLUDE_DIR}"
)

add_dependencies(test_ptltc googletest)

ExternalProject_Get_Property(googletest BINARY_DIR)

file(GLOB LTCC_LIBS "${PTLTC_LIBDIR}/_LAZYC.*.so")
list(GET LTCC_LIBS 0 LTCC_LIBRARY)
message("Selected LTCC library ${LTCC_LIBRARY}")

# The linker does not like the _LAZYC.cpython-36m-x86_64-linux-gnu.so name.
execute_process(COMMAND "ln" "-s" "-f"
  "${LTCC_LIBRARY}"
  "${PTLTC_LIBDIR}/libptltc.so")

find_library(PTLTC_LIB "libptltc.so"
  HINTS "${PTLTC_LIBDIR}")
find_library(PTPY_LIB "libtorch_python.so"
  HINTS "${PT_DIR}/torch/lib")

# Use --unresolved-symbols=ignore-all to get around the c10::Half::from_bits
# undefined symbol error at link time. At runtime everything resolves correctly.
target_link_libraries(
  test_ptltc
  -Wl,--unresolved-symbols=ignore-in-shared-libs
  "${TORCH_LIBRARIES}"
  "${PTLTC_LIB}"
  "${PTPY_LIB}"
  "${BINARY_DIR}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a"
  "${PYTHON_LIBRARY}"
  -lutil
  -pthread
  -lstdc++
  -ldl)
