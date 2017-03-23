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

if(GLOO_INSTALL)
  install(TARGETS benchmark DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endif()

if(USE_CUDA)
  set(GLOO_BENCHMARK_CUDA_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_main.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/options.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/runner.cc"
    )

  cuda_add_executable(benchmark_cuda ${GLOO_BENCHMARK_CUDA_SRCS})
  target_link_libraries(benchmark_cuda gloo_cuda)

  if(GLOO_INSTALL)
    install(TARGETS benchmark_cuda DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
  endif()
endif()
