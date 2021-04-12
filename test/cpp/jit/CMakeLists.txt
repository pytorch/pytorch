set(JIT_TEST_ROOT ${TORCH_ROOT}/test/cpp/jit)

# Build separate libraries the define custom classes/operators used from our Python tests.
# These are intended to be used with torch.ops.load_library() in our Python test suite.
add_library(torchbind_test SHARED
  ${JIT_TEST_ROOT}/test_custom_class_registrations.h
  ${JIT_TEST_ROOT}/test_custom_class_registrations.cpp
)
target_link_libraries(torchbind_test torch)

add_library(jitbackend_test SHARED ${JIT_TEST_ROOT}/test_backend_lib.cpp)
target_link_libraries(jitbackend_test torch)

add_library(backend_with_compiler SHARED
        ${JIT_TEST_ROOT}/test_backend_compiler_lib.cpp
        ${JIT_TEST_ROOT}/test_backend_compiler_preprocess.cpp
        )
target_link_libraries(backend_with_compiler torch)

if(INSTALL_TEST)
  install(TARGETS torchbind_test DESTINATION lib)
  install(TARGETS jitbackend_test DESTINATION lib)
  install(TARGETS backend_with_compiler DESTINATION lib)
endif()

# Build the cpp gtest binary containing the cpp-only tests.
set(JIT_TEST_SRCS
  ${JIT_TEST_ROOT}/test_alias_analysis.cpp
  ${JIT_TEST_ROOT}/test_argument_spec.cpp
  ${JIT_TEST_ROOT}/test_autodiff.cpp
  ${JIT_TEST_ROOT}/test_backend.cpp
  ${JIT_TEST_ROOT}/test_class_import.cpp
  ${JIT_TEST_ROOT}/test_class_parser.cpp
  ${JIT_TEST_ROOT}/test_class_type.cpp
  ${JIT_TEST_ROOT}/test_code_template.cpp
  ${JIT_TEST_ROOT}/test_constant_pooling.cpp
  ${JIT_TEST_ROOT}/test_cleanup_passes.cpp
  ${JIT_TEST_ROOT}/test_create_autodiff_subgraphs.cpp
  ${JIT_TEST_ROOT}/test_custom_class.cpp
  ${JIT_TEST_ROOT}/test_custom_class_registrations.h
  ${JIT_TEST_ROOT}/test_custom_class_registrations.cpp
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
  ${JIT_TEST_ROOT}/test_lite_trainer.cpp
  ${JIT_TEST_ROOT}/test_memory_dag.cpp
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

add_executable(test_jit
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${JIT_TEST_SRCS}
)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_jit PRIVATE USE_GTEST)

if(USE_SYSTEM_ONNX)
  target_link_libraries(test_jit PRIVATE onnx_proto onnx)
endif()

set(JIT_TEST_DEPENDENCIES torch gtest jitbackend_test backend_with_compiler)

if(MSVC)
  list(APPEND JIT_TEST_DEPENDENCIES onnx_library)
endif(MSVC)

target_link_libraries(test_jit PRIVATE ${JIT_TEST_DEPENDENCIES})
target_include_directories(test_jit PRIVATE ${ATen_CPU_INCLUDE})

if(LINUX)
  #Update to target_link_options when CMake version can be upgraded
  target_link_libraries(test_jit PRIVATE "-Wl,--no-as-needed,$<TARGET_FILE:jitbackend_test>,$<TARGET_FILE:backend_with_compiler>,--as-needed")
endif()

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

  target_compile_definitions(test_jit PRIVATE USE_ROCM)
endif()

if(INSTALL_TEST)
  install(TARGETS test_jit DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_jit> DESTINATION bin OPTIONAL)
  endif()
endif()
