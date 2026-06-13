if(USE_KINETO)
  set(PROFILER_TEST_ROOT ${TORCH_ROOT}/test/cpp/profiler)

  set(PROFILER_TEST_DEPENDENCIES torch gtest_main fmt::fmt-header-only)

  add_executable(test_privateuse1_profiler
    ${TORCH_ROOT}/test/cpp/common/main.cpp
    ${PROFILER_TEST_ROOT}/test_privateuse1_profiler.cpp
  )

  target_compile_definitions(test_privateuse1_profiler PRIVATE USE_GTEST USE_KINETO)

  target_link_libraries(test_privateuse1_profiler PRIVATE ${PROFILER_TEST_DEPENDENCIES})
  target_include_directories(test_privateuse1_profiler PRIVATE
    ${ATen_CPU_INCLUDE}
    ${KINETO_SOURCE_DIR}/include
  )

  if(INSTALL_TEST)
    set_target_properties(test_privateuse1_profiler PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
    install(TARGETS test_privateuse1_profiler DESTINATION bin)
  endif()

  add_executable(test_profiler_collection
    ${TORCH_ROOT}/test/cpp/common/main.cpp
    ${PROFILER_TEST_ROOT}/test_profiler_collection.cpp
  )

  target_compile_definitions(test_profiler_collection PRIVATE USE_GTEST USE_KINETO)

  # Link libkineto explicitly. The test calls libkineto::api() and uses
  # libkineto::GenericTraceActivity as a value type (which references its
  # vtable, including GenericTraceActivity::log). Needed for Windows.
  target_link_libraries(test_profiler_collection PRIVATE ${PROFILER_TEST_DEPENDENCIES} kineto)
  target_include_directories(test_profiler_collection PRIVATE
    ${ATen_CPU_INCLUDE}
    ${KINETO_SOURCE_DIR}/include
  )

  if(INSTALL_TEST)
    set_target_properties(test_profiler_collection PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
    install(TARGETS test_profiler_collection DESTINATION bin)
  endif()

  add_executable(test_global_state_manager
    ${TORCH_ROOT}/test/cpp/common/main.cpp
    ${PROFILER_TEST_ROOT}/test_global_state_manager.cpp
  )

  target_compile_definitions(test_global_state_manager PRIVATE USE_GTEST USE_KINETO)

  target_link_libraries(test_global_state_manager PRIVATE ${PROFILER_TEST_DEPENDENCIES})
  target_include_directories(test_global_state_manager PRIVATE
    ${ATen_CPU_INCLUDE}
    ${KINETO_SOURCE_DIR}/include
  )

  if(INSTALL_TEST)
    set_target_properties(test_global_state_manager PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
    install(TARGETS test_global_state_manager DESTINATION bin)
  endif()

  if(USE_DISTRIBUTED)
    add_executable(test_comms_id
      ${TORCH_ROOT}/test/cpp/common/main.cpp
      ${PROFILER_TEST_ROOT}/comms_id.cpp
    )

    target_compile_definitions(test_comms_id PRIVATE USE_GTEST USE_KINETO USE_DISTRIBUTED)

    target_link_libraries(test_comms_id PRIVATE ${PROFILER_TEST_DEPENDENCIES})
    target_include_directories(test_comms_id PRIVATE
      ${ATen_CPU_INCLUDE}
      ${KINETO_SOURCE_DIR}/include
    )

    if(INSTALL_TEST)
      set_target_properties(test_comms_id PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
      install(TARGETS test_comms_id DESTINATION bin)
    endif()
  endif()
endif()
