set(DEPLOY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(interpreter)

#set(DEPLOY_LIB_SOURCES
#  ${DEPLOY_DIR}/torchpy.cpp
#)
#add_library(deploy ${DEPLOY_LIB_SOURCES})
#target_include_directories(deploy PRIVATE ${DEPLOY_DIR} ${INTERPRETER_DIR})
#target_link_libraries(deploy PRIVATE torch)
#target_link_libraries(deploy PUBLIC shm)
#
#set(DEPLOY_TEST_SOURCES
#  ${DEPLOY_DIR}/test_main.cpp
#  ${DEPLOY_DIR}/example/test_threads.cpp
#  ${DEPLOY_DIR}/example/test_models.cpp
#)
#add_executable(deploy_test ${DEPLOY_TEST_SOURCES})
#target_include_directories(deploy_test PRIVATE ${DEPLOY_DIR})
#target_link_libraries(deploy_test PRIVATE deploy torch)
#target_link_libraries(deploy_test PUBLIC gtest)
#
#set(DEPLOY_APP_SOURCES
#  ${DEPLOY_DIR}/app_main.cpp
#)
#add_executable(deploy_app ${DEPLOY_APP_SOURCES})
#target_include_directories(deploy_app  PRIVATE ${DEPLOY_DIR})
#target_link_libraries(deploy_app PRIVATE dl)
#
#add_library(ctype ${DEPLOY_DIR}/libctype.c)
#target_compile_options(ctype PRIVATE -fPIC)
#
#if(USE_CUDA)
#  target_link_libraries(deploy_test PRIVATE
#    ${CUDA_LIBRARIES}
#    ${CUDA_NVRTC_LIB}
#    ${CUDA_CUDA_LIB}
#    ${TORCH_CUDA_LIBRARIES}
#    ctype
#    )
#
#  target_link_libraries(deploy_app PRIVATE
#    ${CUDA_LIBRARIES}
#    ${CUDA_NVRTC_LIB}
#    ${CUDA_CUDA_LIB}
#    ${TORCH_CUDA_LIBRARIES})
#
#  target_compile_definitions(deploy_test PRIVATE "USE_CUDA")
#  target_compile_definitions(deploy_app PRIVATE "USE_CUDA")
#endif()
#
#if(NOT MSVC)
#  if(APPLE)
#    target_compile_options(deploy_test PRIVATE
#      # Clang has an unfixed bug leading to spurious missing braces
#      # warnings, see https://bugs.llvm.org/show_bug.cgi?id=21629
#      -Wno-missing-braces)
#  else()
#    target_compile_options(deploy_test PRIVATE
#      # Considered to be flaky.  See the discussion at
#      # https://github.com/pytorch/pytorch/pull/9608
#      -Wno-maybe-uninitialized
#      # gcc gives nonsensical warnings about variadic.h
#      -Wno-unused-but-set-parameter)
#  endif()
#endif()
#
#if(INSTALL_TEST)
#  install(TARGETS deploy_test DESTINATION bin)
#  # Install PDB files for MSVC builds
#  if(MSVC AND BUILD_SHARED_LIBS)
#    install(FILES $<TARGET_PDB_FILE:deploy_test> DESTINATION bin OPTIONAL)
#  endif()
#endif()
