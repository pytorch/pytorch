cmake_minimum_required(VERSION 3.1)

project(custom_build_project)

set(CMAKE_CXX_STANDARD 14)
set(TORCH_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../)
set(JIT_TEST_ROOT ${TORCH_ROOT}/test/cpp/jit)

# Find torch library
find_package(Torch REQUIRED)

# Predictor executable
if(BUILD_LITE_INTERPRETER)
  add_executable(Predictor lite_predictor.cpp)
else()
  add_executable(Predictor predictor.cpp)
endif()

target_include_directories(Predictor PUBLIC ${TORCH_INCLUDE_DIRS})

find_package(Threads REQUIRED)

# Avoid linker flags on Mac
if(${APPLE})
  set(LINKER_OPTIONS "")
else()
  set(LINKER_OPTIONS "-Wl,-s -Wl,--gc-sections -Wl,--whole-archive -Wl,--no-whole-archive")
endif()

target_link_libraries(Predictor
  ${LINKER_OPTIONS}
  ${TORCH_LIBRARIES}
  Threads::Threads
  ${CMAKE_DL_LIBS}
)

# Test executable
add_executable(test_codegen_unboxing
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${JIT_TEST_ROOT}/test_lite_interpreter.cpp
)

target_compile_definitions(test_codegen_unboxing PRIVATE USE_GTEST)

# set(TEST_UNBOXING_DEPENDENCIES ${TORCH_LIBRARIES})
set(TEST_UNBOXING_DEPENDENCIES torch gtest)

target_link_libraries(test_codegen_unboxing
  ${LINKER_OPTIONS}
  ${TEST_UNBOXING_DEPENDENCIES}
  Threads::Threads
  ${CMAKE_DL_LIBS}
)
message(STATUS "ATen_CPU_INCLUDE: ${ATen_CPU_INCLUDE}")
target_include_directories(test_codegen_unboxing PRIVATE ${ATen_CPU_INCLUDE})

if(INSTALL_TEST)
  install(TARGETS test_codegen_unboxing DESTINATION bin)
endif()