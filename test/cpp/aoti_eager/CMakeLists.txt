set(AOTI_EAGER_TEST_ROOT ${TORCH_ROOT}/test/cpp/aoti_eager)

set(AOTI_EAGER_TEST_SRCS
  ${AOTI_EAGER_TEST_ROOT}/kernel_meta_info_test.cpp
)

add_executable(test_aoti_eager
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${AOTI_EAGER_TEST_SRCS}
)

target_compile_definitions(test_aoti_eager PRIVATE USE_GTEST)

target_link_libraries(test_aoti_eager PRIVATE
  torch
  gtest_main
)

if(INSTALL_TEST)
  install(TARGETS test_aoti_eager DESTINATION bin)
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_aoti_eager> DESTINATION bin OPTIONAL)
  endif()
endif()
