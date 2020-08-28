set(TORCHPY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(interpreter)

set(TORCHPY_LIB_SOURCES
  ${TORCHPY_DIR}/torchpy.cpp
)
add_library(torchpy ${TORCHPY_LIB_SOURCES})
target_include_directories(torchpy PRIVATE ${TORCHPY_DIR})
target_link_libraries(torchpy PRIVATE torch torch_python -Wl,-rpath,/home/whc/local/pytorch-env/lib/ ${PYTHON_LIBRARIES})

set(TORCHPY_TEST_SOURCES
  ${TORCHPY_DIR}/test_main.cpp
  ${TORCHPY_DIR}/example/test_threads.cpp
  ${TORCHPY_DIR}/example/test_models.cpp
)
add_executable(torchpy_test ${TORCHPY_TEST_SOURCES})
target_include_directories(torchpy_test PRIVATE ${TORCHPY_DIR})
target_link_libraries(torchpy_test PRIVATE torchpy torch -Wl,-rpath,/home/whc/local/pytorch-env/lib/ ${PYTHON_LIBRARIES})
target_link_libraries(torchpy_test PUBLIC gtest)

set(TORCHPY_APP_SOURCES
  ${TORCHPY_DIR}/app_main.cpp
)
add_executable(torchpy_app ${TORCHPY_APP_SOURCES})
target_include_directories(torchpy_app  PRIVATE ${TORCHPY_DIR})
target_link_libraries(torchpy_app PRIVATE dl)

if(USE_CUDA)
  target_link_libraries(torchpy_test PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  target_link_libraries(torchpy_app PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(torchpy_test PRIVATE "USE_CUDA")
  target_compile_definitions(torchpy_app PRIVATE "USE_CUDA")
endif()

if(NOT MSVC)
  if(APPLE)
    target_compile_options(torchpy_test PRIVATE
      # Clang has an unfixed bug leading to spurious missing braces
      # warnings, see https://bugs.llvm.org/show_bug.cgi?id=21629
      -Wno-missing-braces)
  else()
    target_compile_options(torchpy_test PRIVATE
      # Considered to be flaky.  See the discussion at
      # https://github.com/pytorch/pytorch/pull/9608
      -Wno-maybe-uninitialized
      # gcc gives nonsensical warnings about variadic.h
      -Wno-unused-but-set-parameter)
  endif()
endif()

if(INSTALL_TEST)
  install(TARGETS torchpy_test DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:torchpy_test> DESTINATION bin OPTIONAL)
  endif()
endif()
