# ---[ Test binaries.

file(GLOB_RECURSE C10_ALL_TEST_FILES *_test.cpp)
if(BUILD_TEST)
  foreach(test_src ${C10_ALL_TEST_FILES})
    get_filename_component(test_file_name ${test_src} NAME_WE)
    set(test_name "c10_${test_file_name}")
    add_executable(${test_name} "${test_src}")
    if(NOT MSVC)
      target_compile_options(${test_name} PRIVATE -Wno-unused-variable)
    endif()
    target_link_libraries(${test_name} ${C10_LIB} gmock gtest gtest_main)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if(INSTALL_TEST)
      set_target_properties(${test_name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
      install(TARGETS ${test_name} DESTINATION test)
    endif()
  endforeach()
endif()

# ---[ C++17 header compilation test
if(BUILD_TEST)
  add_executable(c10_cpp17_header_build_test util/cpp17_header_build_check.cpp)
  target_link_libraries(c10_cpp17_header_build_test ${C10_LIB} gmock gtest gtest_main)
  set_target_properties(c10_cpp17_header_build_test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )
  add_test(NAME c10_cpp17_header_build_test COMMAND $<TARGET_FILE:c10_cpp17_header_build_test>)
endif()
