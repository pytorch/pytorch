add_definitions(-DBENCHMARK_TCP=1)
if(USE_IBVERBS)
  add_definitions(-DBENCHMARK_IBVERBS=1)
endif()

set(GLOO_BENCHMARK_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/main.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/options.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/runner.cc"
  )

add_executable(benchmark ${GLOO_BENCHMARK_SRCS})
target_link_libraries(benchmark gloo)
