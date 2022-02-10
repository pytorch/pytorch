cmake_minimum_required(VERSION 3.1)

project(custom_build_project)

set(CMAKE_CXX_STANDARD 14)

# Find torch library
find_package(Torch REQUIRED)

# Main executable
add_executable(Predictor predictor.cpp)
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
