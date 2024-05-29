# ---[ Benchmark binaries.

file(GLOB_RECURSE C10_ALL_BENCH_FILES *.cpp)
if(BUILD_TEST)
  foreach(bench_src ${C10_ALL_BENCH_FILES})
    get_filename_component(bench_file_name ${bench_src} NAME_WE)
    set(bench_name "c10_${bench_file_name}")
    add_executable(${bench_name} "${bench_src}")
    target_link_libraries(${bench_name} c10 benchmark)
    if(INSTALL_TEST)
      install(TARGETS ${bench_name} DESTINATION test)
    endif()
  endforeach()
endif()
