set(JIT_TEST_ROOT ${TORCH_ROOT}/test/cpp/jit)

set(JIT_TEST_SRCS
  ${JIT_TEST_ROOT}/test_alias_analysis.cpp
  ${JIT_TEST_ROOT}/test_argument_spec.cpp
  ${JIT_TEST_ROOT}/test_autodiff.cpp
  ${JIT_TEST_ROOT}/test_backend.cpp
  ${JIT_TEST_ROOT}/test_base.cpp
  ${JIT_TEST_ROOT}/test_base.h
  ${JIT_TEST_ROOT}/test_class_import.cpp
  ${JIT_TEST_ROOT}/test_class_parser.cpp
  ${JIT_TEST_ROOT}/test_class_type.cpp
  ${JIT_TEST_ROOT}/test_code_template.cpp
  ${JIT_TEST_ROOT}/test_constant_pooling.cpp
  ${JIT_TEST_ROOT}/test_create_autodiff_subgraphs.cpp
  ${JIT_TEST_ROOT}/test_custom_class.cpp
  ${JIT_TEST_ROOT}/test_custom_operators.cpp
  ${JIT_TEST_ROOT}/test_dce.cpp
  ${JIT_TEST_ROOT}/test_fuser.cpp
  ${JIT_TEST_ROOT}/test_graph_executor.cpp
  ${JIT_TEST_ROOT}/test_inliner.cpp
  ${JIT_TEST_ROOT}/test_interface.cpp
  ${JIT_TEST_ROOT}/test_interpreter.cpp
  ${JIT_TEST_ROOT}/test_ir.cpp
  ${JIT_TEST_ROOT}/test_irparser.cpp
  ${JIT_TEST_ROOT}/test_jit_type.cpp
  ${JIT_TEST_ROOT}/test_lite_interpreter.cpp
  ${JIT_TEST_ROOT}/test_misc.cpp
  ${JIT_TEST_ROOT}/test_mobile_type_parser.cpp
  ${JIT_TEST_ROOT}/test_module_api.cpp
  ${JIT_TEST_ROOT}/test_peephole_optimize.cpp
  ${JIT_TEST_ROOT}/test_qualified_name.cpp
  ${JIT_TEST_ROOT}/test_save_load.cpp
  ${JIT_TEST_ROOT}/test_schema_matching.cpp
  ${JIT_TEST_ROOT}/test_subgraph_matcher.cpp
  ${JIT_TEST_ROOT}/test_subgraph_rewriter.cpp
  ${JIT_TEST_ROOT}/test_subgraph_utils.cpp
  ${JIT_TEST_ROOT}/test_utils.cpp
)

if(USE_CUDA)
  list(APPEND JIT_TEST_SRCS ${JIT_TEST_ROOT}/test_gpu.cpp)
endif()

set(JIT_TEST_SRCS ${JIT_TEST_SRCS} PARENT_SCOPE)

add_executable(test_jit
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${JIT_TEST_ROOT}/gtest.cpp
  ${JIT_TEST_SRCS})
if(USE_SYSTEM_ONNX)
  target_link_libraries(test_jit PRIVATE onnx_proto onnx)
endif()

set(JIT_TEST_DEPENDENCIES torch gtest)

if(MSVC)
  list(APPEND JIT_TEST_DEPENDENCIES onnx_library)
endif(MSVC)

target_link_libraries(test_jit PRIVATE ${JIT_TEST_DEPENDENCIES})
target_include_directories(test_jit PRIVATE ${ATen_CPU_INCLUDE})

if(USE_CUDA)
  target_link_libraries(test_jit PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_jit PRIVATE USE_CUDA)
elseif(USE_ROCM)
  target_link_libraries(test_jit PRIVATE
    ${ROCM_HIPRTC_LIB}
    ${PYTORCH_HIP_HCC_LIBRARIES}
    ${TORCH_CUDA_LIBRARIES})

  target_link_libraries(test_jit PRIVATE caffe2_gpu)

  target_compile_definitions(test_jit PRIVATE USE_ROCM)
endif()

if(INSTALL_TEST)
  install(TARGETS test_jit DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_jit> DESTINATION bin OPTIONAL)
  endif()
endif()
