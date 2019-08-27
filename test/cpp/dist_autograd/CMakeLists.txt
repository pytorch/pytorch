set(TORCH_DIST_AUTOGRAD_TEST_DIR "${TORCH_ROOT}/test/cpp/dist_autograd")
set(TORCH_API_TEST_SOURCES
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${TORCH_DIST_AUTOGRAD_TEST_DIR}/test_dist_autograd.cpp
)

add_executable(test_dist_autograd ${TORCH_API_TEST_SOURCES})
target_include_directories(test_dist_autograd PRIVATE ${ATen_CPU_INCLUDE})
target_link_libraries(test_dist_autograd PRIVATE torch gtest)

if (NOT MSVC)
  if (APPLE)
    target_compile_options(test_dist_autograd PRIVATE
      # Clang has an unfixed bug leading to spurious missing braces
      # warnings, see https://bugs.llvm.org/show_bug.cgi?id=21629
      -Wno-missing-braces)
  else()
    target_compile_options(test_dist_autograd PRIVATE
      # Considered to be flaky.  See the discussion at
      # https://github.com/pytorch/pytorch/pull/9608
      -Wno-maybe-uninitialized
      # gcc gives nonsensical warnings about variadic.h
      -Wno-unused-but-set-parameter)
  endif()
endif()

if (INSTALL_TEST)
  install(TARGETS test_dist_autograd DESTINATION bin)
  # Install PDB files for MSVC builds
  if (MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_dist_autograd> DESTINATION bin OPTIONAL)
  endif()
endif()
