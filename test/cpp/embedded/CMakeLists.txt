set(EMBEDDED_TEST_ROOT ${TORCH_ROOT}/test/cpp/embedded)

# Build the cpp gtest binary containing the cpp-only tests.
set(EMBEDDED_TEST_SRCS
  ${EMBEDDED_TEST_ROOT}/test_embedded.cpp
)

add_executable(test_embedded
  ${EMBEDDED_TEST_SRCS}
)

target_link_libraries(
  test_embedded PRIVATE flatbuffers)


# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_embedded PRIVATE USE_GTEST)

set(EMBEDDED_TEST_DEPENDENCIES gtest)

target_link_libraries(test_embedded PRIVATE ${EMBEDDED_TEST_DEPENDENCIES})
#target_include_directories(test_embedded PRIVATE ${ATen_CPU_INCLUDE})
if(NOT MSVC)
  target_compile_options(test_embedded PRIVATE -Wno-unused-variable)
endif()

if(LINUX)
  #Update to target_link_options when CMake version can be upgraded
  target_link_libraries(test_embedded PRIVATE "-Wl,--no-as-needed")
endif()

if(INSTALL_TEST)
  install(TARGETS test_jit DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_jit> DESTINATION bin OPTIONAL)
  endif()
endif()
