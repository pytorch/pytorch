set(TENSOREXPR_TEST_ROOT ${TORCH_ROOT}/test/cpp/tensorexpr)

set(TENSOREXPR_TEST_SRCS
  ${TENSOREXPR_TEST_ROOT}/test_approx.cpp
  ${TENSOREXPR_TEST_ROOT}/test_aten.cpp
  ${TENSOREXPR_TEST_ROOT}/test_boundsinference.cpp
  ${TENSOREXPR_TEST_ROOT}/test_conv.cpp
  ${TENSOREXPR_TEST_ROOT}/test_cpp_codegen.cpp
  ${TENSOREXPR_TEST_ROOT}/test_dynamic_shapes.cpp
  ${TENSOREXPR_TEST_ROOT}/test_expr.cpp
  ${TENSOREXPR_TEST_ROOT}/test_external_calls.cpp
  ${TENSOREXPR_TEST_ROOT}/test_graph_opt.cpp
  ${TENSOREXPR_TEST_ROOT}/test_ir_printer.cpp
  ${TENSOREXPR_TEST_ROOT}/test_ir_verifier.cpp
  ${TENSOREXPR_TEST_ROOT}/test_kernel.cpp
  ${TENSOREXPR_TEST_ROOT}/test_loopnest.cpp
  ${TENSOREXPR_TEST_ROOT}/test_memdependency.cpp
  ${TENSOREXPR_TEST_ROOT}/test_ops.cpp
  ${TENSOREXPR_TEST_ROOT}/test_quantization.cpp
  ${TENSOREXPR_TEST_ROOT}/test_memplanning.cpp
  ${TENSOREXPR_TEST_ROOT}/test_reductions.cpp
  ${TENSOREXPR_TEST_ROOT}/test_registerizer.cpp
  ${TENSOREXPR_TEST_ROOT}/test_simplify.cpp
  ${TENSOREXPR_TEST_ROOT}/test_te_fuser_pass.cpp
  ${TENSOREXPR_TEST_ROOT}/test_type.cpp
  ${TENSOREXPR_TEST_ROOT}/test_type_specializations.cpp
)

if(USE_CUDA)
  list(APPEND TENSOREXPR_TEST_SRCS ${TENSOREXPR_TEST_ROOT}/test_cuda.cpp)
endif()

if(USE_LLVM AND LLVM_FOUND)
  list(APPEND TENSOREXPR_TEST_SRCS ${TENSOREXPR_TEST_ROOT}/test_llvm.cpp)
endif()

add_executable(test_tensorexpr
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${TENSOREXPR_TEST_ROOT}/padded_buffer.cpp
  ${TENSOREXPR_TEST_SRCS})

target_link_libraries(test_tensorexpr PRIVATE torch gtest)
target_include_directories(test_tensorexpr PRIVATE ${ATen_CPU_INCLUDE})
target_compile_definitions(test_tensorexpr PRIVATE USE_GTEST)
if(NOT MSVC)
  target_compile_options(test_tensorexpr PRIVATE -Wno-unused-variable)
endif()

add_executable(tutorial_tensorexpr ${TENSOREXPR_TEST_ROOT}/tutorial.cpp)
target_link_libraries(tutorial_tensorexpr PRIVATE torch)
target_include_directories(tutorial_tensorexpr PRIVATE ${ATen_CPU_INCLUDE})

# The test case depends on the xnnpack header which in turn depends on the
# pthreadpool header. For some build environment we need add the dependency
# explicitly.
if(USE_PTHREADPOOL)
  target_link_libraries(test_tensorexpr PRIVATE pthreadpool_interface)
endif()
if(USE_CUDA)
  target_compile_definitions(test_tensorexpr PRIVATE USE_CUDA)
  target_compile_definitions(tutorial_tensorexpr PRIVATE USE_CUDA)
elseif(USE_ROCM)
  target_link_libraries(test_tensorexpr PRIVATE
    ${ROCM_HIPRTC_LIB}
    ${PYTORCH_HIP_LIBRARIES}
    ${TORCH_CUDA_LIBRARIES})
  target_compile_definitions(test_tensorexpr PRIVATE USE_ROCM)

  target_link_libraries(tutorial_tensorexpr PRIVATE
    ${ROCM_HIPRTC_LIB}
    ${PYTORCH_HIP_LIBRARIES}
    ${TORCH_CUDA_LIBRARIES})
  target_compile_definitions(tutorial_tensorexpr PRIVATE USE_ROCM)
endif()

if(INSTALL_TEST)
  install(TARGETS test_tensorexpr DESTINATION bin)
  install(TARGETS tutorial_tensorexpr DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_tensorexpr> DESTINATION bin OPTIONAL)
    install(FILES $<TARGET_PDB_FILE:tutorial_tensorexpr> DESTINATION bin OPTIONAL)
  endif()
endif()
