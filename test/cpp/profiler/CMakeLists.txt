# Test for PrivateUse1 profiler registration

if(USE_KINETO)
  set(PROFILER_TEST_ROOT ${TORCH_ROOT}/test/cpp/profiler)

  add_executable(test_privateuse1_profiler
    ${TORCH_ROOT}/test/cpp/common/main.cpp
    ${PROFILER_TEST_ROOT}/test_privateuse1_profiler.cpp
  )

  target_compile_definitions(test_privateuse1_profiler PRIVATE USE_GTEST USE_KINETO)

  set(PROFILER_TEST_DEPENDENCIES torch gtest_main fmt::fmt-header-only)

  target_link_libraries(test_privateuse1_profiler PRIVATE ${PROFILER_TEST_DEPENDENCIES})
  target_include_directories(test_privateuse1_profiler PRIVATE
    ${ATen_CPU_INCLUDE}
    ${KINETO_SOURCE_DIR}/include
  )

  if(INSTALL_TEST)
    set_target_properties(test_privateuse1_profiler PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
    install(TARGETS test_privateuse1_profiler DESTINATION bin)
  endif()
endif()
