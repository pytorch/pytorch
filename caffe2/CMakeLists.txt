# ---[ Generate and install header and cpp files
include(../cmake/Codegen.cmake)

# ---[ Declare source file lists

# ---[ ATen build
if (INTERN_BUILD_ATEN_OPS)
  set(__caffe2_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(../aten aten)
  set(CMAKE_POSITION_INDEPENDENT_CODE ${__caffe2_CMAKE_POSITION_INDEPENDENT_CODE})

  # Generate the headers wrapped by our operator
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/contrib/aten/aten_op.h
  COMMAND
  "${PYTHON_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten/gen_op.py
    --aten_root=${CMAKE_CURRENT_SOURCE_DIR}/../aten
    --template_dir=${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten
    --yaml_dir=${CMAKE_CURRENT_BINARY_DIR}/../aten/src/ATen
    --install_dir=${CMAKE_CURRENT_BINARY_DIR}/contrib/aten
  DEPENDS
  ATEN_CPU_FILES_GEN_TARGET
  ${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml
  ${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten/gen_op.py
  ${CMAKE_CURRENT_SOURCE_DIR}/contrib/aten/aten_op_template.h)

  add_custom_target(__aten_op_header_gen
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/contrib/aten/aten_op.h)
  add_library(aten_op_header_gen INTERFACE)
  add_dependencies(aten_op_header_gen __aten_op_header_gen)

  # Add source, includes, and libs to lists
  list(APPEND Caffe2_CPU_SRCS ${ATen_CPU_SRCS})
  list(APPEND Caffe2_GPU_SRCS ${ATen_CUDA_SRCS})
  list(APPEND Caffe2_HIP_SRCS ${ATen_HIP_SRCS})
  list(APPEND Caffe2_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS})
  list(APPEND Caffe2_GPU_TEST_SRCS ${ATen_CUDA_TEST_SRCS})
  list(APPEND Caffe2_HIP_TEST_SRCS ${ATen_HIP_TEST_SRCS})
  list(APPEND Caffe2_CPU_TEST_SRCS ${ATen_CORE_TEST_SRCS})
  list(APPEND Caffe2_CPU_INCLUDE ${ATen_CPU_INCLUDE})
  list(APPEND Caffe2_GPU_INCLUDE ${ATen_CUDA_INCLUDE})
  list(APPEND Caffe2_HIP_INCLUDE ${ATen_HIP_INCLUDE})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS})
  list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS})
  list(APPEND Caffe2_HIP_DEPENDENCY_LIBS ${ATen_HIP_DEPENDENCY_LIBS})
  list(APPEND Caffe2_DEPENDENCY_INCLUDE ${ATen_THIRD_PARTY_INCLUDE})
endif()

# ---[ Caffe2 build
# Note: the folders that are being commented out have not been properly
# addressed yet.

# For pthreadpool_new_if_impl. TODO: Remove when threadpools are unitied.
if (NOT MSVC)
  IF(NOT TARGET fxdiv)
    SET(FXDIV_BUILD_TESTS OFF CACHE BOOL "")
    SET(FXDIV_BUILD_BENCHMARKS OFF CACHE BOOL "")
    ADD_SUBDIRECTORY(
      "${FXDIV_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/FXdiv")
  ENDIF()
  if (NOT (INTERN_BUILD_MOBILE AND NOT BUILD_CAFFE2_MOBILE))
    set_source_files_properties(
        utils/threadpool/pthreadpool_new_if_impl.c PROPERTIES COMPILE_FLAGS -fno-openmp)
  endif()
endif()

add_subdirectory(core)
add_subdirectory(serialize)
add_subdirectory(utils)
add_subdirectory(perfkernels)

# Skip modules that are not used by libtorch mobile yet.
if (NOT INTERN_BUILD_MOBILE OR BUILD_CAFFE2_MOBILE)
  add_subdirectory(contrib)
  add_subdirectory(predictor)
  add_subdirectory(predictor/emulator)
  add_subdirectory(core/nomnigraph)
  if (USE_NVRTC)
    add_subdirectory(cuda_rtc)
  endif()
  add_subdirectory(db)
  add_subdirectory(distributed)
  # add_subdirectory(experiments) # note, we may remove this folder at some point
  add_subdirectory(ideep)
  add_subdirectory(image)
  add_subdirectory(video)
  add_subdirectory(mobile)
  add_subdirectory(mpi)
  add_subdirectory(observers)
  add_subdirectory(onnx)
  if (BUILD_CAFFE2_OPS)
    add_subdirectory(operators)
    add_subdirectory(operators/rnn)
    if (USE_FBGEMM)
      add_subdirectory(quantization)
      add_subdirectory(quantization/server)
    endif()
    if (USE_QNNPACK)
      add_subdirectory(operators/quantized)
    endif()
  endif()
  add_subdirectory(opt)
  add_subdirectory(proto)
  add_subdirectory(python)
  add_subdirectory(queue)
  add_subdirectory(sgd)
  add_subdirectory(share)
  # add_subdirectory(test) # todo: use caffe2_gtest_main instead of gtest_main because we will need to call GlobalInit
  add_subdirectory(transforms)
endif()

# Advanced: if we have white list specified, we will do intersections for all
# main lib srcs.
if (CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_CPU_SRCS CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_GPU_SRCS CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_HIP_SRCS CAFFE2_WHITELISTED_FILES)
endif()

# Debug messages - if you want to get a list of source files, enable the
# following.
if (FALSE)
  message(STATUS "CPU sources: ")
  foreach(tmp ${Caffe2_CPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU sources: ")
  foreach(tmp ${Caffe2_GPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU include: ")
  foreach(tmp ${Caffe2_CPU_INCLUDE})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU include: ")
  foreach(tmp ${Caffe2_GPU_INCLUDE})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU test sources: ")
  foreach(tmp ${Caffe2_CPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU test sources: ")
  foreach(tmp ${Caffe2_GPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "HIP sources: ")
  foreach(tmp ${Caffe2_HIP_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "HIP test sources: ")
  foreach(tmp ${Caffe2_HIP_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen CPU test sources: ")
  foreach(tmp ${ATen_CPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen CUDA test sources: ")
  foreach(tmp ${ATen_CUDA_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen HIP test sources: ")
  foreach(tmp ${ATen_HIP_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()
endif()

if (NOT INTERN_BUILD_MOBILE OR BUILD_CAFFE2_MOBILE)
  # ---[ List of libraries to link with
  add_library(caffe2_protos STATIC $<TARGET_OBJECTS:Caffe2_PROTO>)
  add_dependencies(caffe2_protos Caffe2_PROTO)
  # If we are going to link protobuf locally inside caffe2 libraries, what we will do is
  # to create a helper static library that always contains libprotobuf source files, and
  # link the caffe2 related dependent libraries to it.
  target_include_directories(caffe2_protos INTERFACE $<INSTALL_INTERFACE:include>)
  # Reason for this public dependency is as follows:
  # (1) Strictly speaking, we should not expose any Protobuf related functions. We should
  #     only use function interfaces wrapped with our own public API, and link protobuf
  #     locally.
  # (2) However, currently across the Caffe2 codebase, we have extensive use of protobuf
  #     functionalities. For example, not only libcaffe2.so uses it, but also other
  #     binaries such as python extensions etc. As a result, we will have to have a
  #     transitive dependency to libprotobuf.
  #
  # Good thing is that, if we specify CAFFE2_LINK_LOCAL_PROTOBUF, then we do not need to
  # separately deploy protobuf binaries - libcaffe2.so will contain all functionalities
  # one needs. One can verify this via ldd.
  #
  # TODO item in the future includes:
  # (1) Enable using lite protobuf
  # (2) Properly define public API that do not directly depend on protobuf itself.
  # (3) Expose the libprotobuf.a file for dependent libraries to link to.
  #
  # What it means for users/developers?
  # (1) Users: nothing affecting the users, other than the fact that CAFFE2_LINK_LOCAL_PROTOBUF
  #     avoids the need to deploy protobuf.
  # (2) Developers: if one simply uses core caffe2 functionality without using protobuf,
  #     nothing changes. If one has a dependent library that uses protobuf, then one needs to
  #     have the right protobuf version as well as linking to libprotobuf.a.
  target_link_libraries(caffe2_protos PUBLIC protobuf::libprotobuf)
  if (NOT BUILD_SHARED_LIBS)
    INSTALL(TARGETS caffe2_protos ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif()
endif()

# ==========================================================
# formerly-libtorch
# ==========================================================

set(TORCH_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../torch")
set(TORCH_ROOT "${TORCH_SRC_DIR}/..")

if(NOT TORCH_INSTALL_BIN_DIR)
  set(TORCH_INSTALL_BIN_DIR bin)
endif()

if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include)
endif()

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()



if (NOT INTERN_BUILD_MOBILE OR NOT BUILD_CAFFE2_MOBILE)


  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

  # Generate files
  set(TOOLS_PATH "${TORCH_ROOT}/tools")

  configure_file("${TORCH_ROOT}/aten/src/ATen/common_with_cwrap.py"
    "${TOOLS_PATH}/shared/cwrap_common.py"
    COPYONLY)

  configure_file("${TORCH_SRC_DIR}/_utils_internal.py"
    "${TOOLS_PATH}/shared/_utils_internal.py"
    COPYONLY)


  set(GENERATED_CXX_TORCH
    "${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.cpp"
    "${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops_0.cpp"
    "${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops_1.cpp"
    "${TORCH_SRC_DIR}/csrc/jit/generated/register_aten_ops_2.cpp"
    )

  if(NOT INTERN_DISABLE_AUTOGRAD)
    list(APPEND GENERATED_CXX_TORCH
      "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_0.cpp"
      "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_1.cpp"
      "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_2.cpp"
      "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_3.cpp"
      "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_4.cpp"
    )
  endif()

  set(GENERATED_H_TORCH
    "${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.h"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/variable_factories.h"
    )

  if(NOT INTERN_DISABLE_AUTOGRAD)
    list(APPEND GENERATED_H_TORCH
      "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType.h"
    )
  endif()

  set(GENERATED_CXX_PYTHON
    "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/python_variable_methods.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/python_torch_functions.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/python_nn_functions.cpp"
    )

  set(GENERATED_H_PYTHON
    "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions.h"
    )

  set(TORCH_GENERATED_CODE
    ${GENERATED_CXX_TORCH}
    ${GENERATED_H_TORCH}
    ${GENERATED_CXX_PYTHON}
    ${GENERATED_H_PYTHON}
    )

  add_custom_command(
    OUTPUT
    ${TORCH_GENERATED_CODE}
    COMMAND
    "${PYTHON_EXECUTABLE}" tools/setup_helpers/generate_code.py
      --declarations-path "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
      --nn-path "aten/src"
      $<$<BOOL:${INTERN_DISABLE_AUTOGRAD}>:--disable-autograd>
      $<$<BOOL:${SELECTED_OP_LIST}>:--selected-op-list-path="${SELECTED_OP_LIST}">
    DEPENDS
    "${CMAKE_BINARY_DIR}/aten/src/ATen/Declarations.yaml"
    "${TOOLS_PATH}/autograd/templates/VariableType.h"
    "${TOOLS_PATH}/autograd/templates/VariableType.cpp"
    "${TOOLS_PATH}/autograd/templates/Functions.h"
    "${TOOLS_PATH}/autograd/templates/Functions.cpp"
    "${TOOLS_PATH}/autograd/templates/python_functions.h"
    "${TOOLS_PATH}/autograd/templates/python_functions.cpp"
    "${TOOLS_PATH}/autograd/templates/python_variable_methods.cpp"
    "${TOOLS_PATH}/autograd/templates/python_torch_functions.cpp"
    "${TOOLS_PATH}/autograd/templates/python_nn_functions.cpp"
    "${TOOLS_PATH}/autograd/templates/variable_factories.h"
    "${TOOLS_PATH}/autograd/deprecated.yaml"
    "${TOOLS_PATH}/autograd/derivatives.yaml"
    "${TOOLS_PATH}/autograd/gen_autograd_functions.py"
    "${TOOLS_PATH}/autograd/gen_autograd.py"
    "${TOOLS_PATH}/autograd/gen_python_functions.py"
    "${TOOLS_PATH}/autograd/gen_variable_factories.py"
    "${TOOLS_PATH}/autograd/gen_variable_type.py"
    "${TOOLS_PATH}/autograd/load_derivatives.py"
    "${TOOLS_PATH}/autograd/nested_dict.py"
    "${TOOLS_PATH}/autograd/utils.py"
    "${TOOLS_PATH}/jit/gen_jit_dispatch.py"
    "${TOOLS_PATH}/jit/templates/register_aten_ops.cpp"
    WORKING_DIRECTORY "${TORCH_ROOT}")


  # Required workaround for libtorch_python.so build
  # see https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/#custom-commands-in-different-directories
  add_custom_target(
    generate-torch-sources
    DEPENDS ${TORCH_GENERATED_CODE}
    )


  set(TORCH_SRCS
    ${GENERATED_CXX_TORCH}
    ${GENERATED_H_TORCH}
    ${TORCH_SRC_DIR}/csrc/autograd/anomaly_mode.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/autograd.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/custom_function.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/cpp_hook.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/engine.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/function.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/function_hook.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/functions/accumulate_grad.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/functions/basic_ops.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/functions/tensor.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/functions/utils.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/input_buffer.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/profiler.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/record_function.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/record_function_ops.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/saved_variable.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/variable.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/autodiff.cpp
    ${TORCH_SRC_DIR}/csrc/jit/ir/attributes.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/argument_spec.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/pass_manager.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/pickler.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/unpickler.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/graph_executor.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/import_source.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/import.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/pickle.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/import_export_helpers.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/instruction.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/interpreter.cpp
    ${TORCH_SRC_DIR}/csrc/jit/ir/constants.cpp
    ${TORCH_SRC_DIR}/csrc/jit/ir/node_hashing.cpp
    ${TORCH_SRC_DIR}/csrc/jit/ir/type_hashing.cpp
    ${TORCH_SRC_DIR}/csrc/jit/ir/ir.cpp
    ${TORCH_SRC_DIR}/csrc/jit/ir/irparser.cpp
    ${TORCH_SRC_DIR}/csrc/jit/jit_log.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/operator.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/register_c10_ops.cpp
    ${TORCH_SRC_DIR}/csrc/jit/ir/subgraph_matcher.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/symbolic_script.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/profiling_record.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/profiling_graph_executor_impl.cpp
    ${TORCH_SRC_DIR}/csrc/jit/python/update_graph_executor_opt.cpp
    ${TORCH_SRC_DIR}/csrc/jit/ir/alias_analysis.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/batch_mm.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/bailout_graph.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/canonicalize.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/clear_undefinedness.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/constant_propagation.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/constant_pooling.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/common_subexpression_elimination.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/create_autodiff_subgraphs.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/inline_autodiff_subgraphs.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/insert_guards.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/inliner.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/lift_closures.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/inline_forked_closures.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/dead_code_elimination.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/decompose_ops.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/canonicalize_ops.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/erase_number_types.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/fixup_trace_scope_blocks.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/inline_fork_wait.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/graph_fuser.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/cuda_graph_fuser.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/guard_elimination.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/inplace_check.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/liveness.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/loop_unrolling.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/lower_grad_of.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/lower_graph.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/lower_tuples.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/peephole.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/remove_expands.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/remove_inplace_ops.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/shape_analysis.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/requires_grad_analysis.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/specialize_autogradzero.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/subgraph_rewrite.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/tensorexpr_fuser.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/python_print.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/utils/subgraph_utils.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/utils/check_alias_annotation.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/utils/memory_dag.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/quantization.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/fuse_linear.cpp
    ${TORCH_SRC_DIR}/csrc/jit/passes/freeze_module.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/print_handler.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/fuser/interface.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/register_prim_ops.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/register_prim_ops_c10.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/register_string_ops.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/register_special_ops.cpp
    ${TORCH_SRC_DIR}/csrc/jit/ir/scope.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/ir_emitter.cpp
    ${TORCH_SRC_DIR}/csrc/jit/testing/file_check.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/convert_to_ssa.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/exit_transforms.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/inline_loop_condition.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/schema_matching.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/script_type_parser.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/sugared_value.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/parser.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/builtin_functions.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/canonicalize_modified_loop.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/edit_distance.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/logging.cpp
    ${TORCH_SRC_DIR}/csrc/jit/api/module.cpp
    ${TORCH_SRC_DIR}/csrc/jit/api/object.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/jit_exception.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/string_to_type.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/source_range_serialization.cpp
    ${TORCH_SRC_DIR}/csrc/jit/frontend/tracer.cpp
    ${TORCH_SRC_DIR}/csrc/jit/testing/hooks_for_testing.cpp
    ${TORCH_SRC_DIR}/csrc/utils/tensor_flatten.cpp
    ${TORCH_SRC_DIR}/csrc/utils/variadic.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/fuser/kernel_cache.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/fuser/compiler.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/fuser/executor.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/fuser/codegen.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/fuser/fallback.cpp
    ${TORCH_SRC_DIR}/csrc/jit/api/function_impl.cpp
    ${TORCH_SRC_DIR}/csrc/jit/runtime/vararg_functions.cpp

    ${TORCH_SRC_DIR}/csrc/jit/tensorexpr/mem_arena.cpp
    ${TORCH_SRC_DIR}/csrc/jit/tensorexpr/codegen.cpp
    ${TORCH_SRC_DIR}/csrc/jit/tensorexpr/expr.cpp
    ${TORCH_SRC_DIR}/csrc/jit/tensorexpr/eval.cpp
    ${TORCH_SRC_DIR}/csrc/jit/tensorexpr/ir.cpp
    ${TORCH_SRC_DIR}/csrc/jit/tensorexpr/ir_visitor.cpp
    ${TORCH_SRC_DIR}/csrc/jit/tensorexpr/types.cpp
    ${TORCH_SRC_DIR}/csrc/jit/tensorexpr/ir_printer.cpp
    ${TORCH_SRC_DIR}/csrc/jit/tensorexpr/ir_mutator.cpp
    ${TORCH_SRC_DIR}/csrc/jit/tensorexpr/unique_name_manager.cpp
    ${TORCH_SRC_DIR}/csrc/jit/mobile/type_parser.cpp
    )

  if (NOT INTERN_DISABLE_MOBILE_INTERP)
    set (MOBILE_SRCS
        ${TORCH_SRC_DIR}/csrc/jit/mobile/function.cpp
        ${TORCH_SRC_DIR}/csrc/jit/mobile/import.cpp
        ${TORCH_SRC_DIR}/csrc/jit/mobile/module.cpp
        ${TORCH_SRC_DIR}/csrc/jit/mobile/interpreter.cpp
        )
    list (APPEND TORCH_SRCS ${MOBILE_SRCS})
  endif()

  if (NOT INTERN_DISABLE_AUTOGRAD)
    list(APPEND TORCH_SRCS
      ${TORCH_SRC_DIR}/csrc/autograd/VariableTypeManual.cpp
    )
  endif()

  if (NOT INTERN_BUILD_MOBILE)
    list(APPEND TORCH_SRCS
      ${TORCH_SRC_DIR}/csrc/api/src/jit.cpp
      ${TORCH_SRC_DIR}/csrc/jit/serialization/export.cpp
      ${TORCH_SRC_DIR}/csrc/jit/serialization/export_module.cpp
      ${TORCH_SRC_DIR}/csrc/jit/serialization/import_legacy.cpp
      ${TORCH_SRC_DIR}/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp
      ${TORCH_SRC_DIR}/csrc/jit/api/module_save.cpp
      ${TORCH_SRC_DIR}/csrc/utils/byte_order.cpp
    )
    if (USE_DISTRIBUTED)
      list(APPEND TORCH_SRCS
        ${TORCH_SRC_DIR}/csrc/distributed/autograd/context/container.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/autograd/context/context.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/autograd/engine/dist_engine.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/autograd/functions/recvrpc_backward.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/autograd/functions/sendrpc_backward.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/autograd/rpc_messages/autograd_metadata.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/autograd/utils.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/message.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/python_call.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/python_remote_call.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/python_resp.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/request_callback.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/rpc_agent.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/rref_context.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/rref_proto.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/rref_impl.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/torchscript_functions.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/script_call.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/script_remote_call.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/script_resp.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/types.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/rpc/utils.cpp
      )
    endif()
  endif()

  if (USE_CUDA)
    list(APPEND Caffe2_GPU_SRCS
      ${TORCH_SRC_DIR}/csrc/jit/codegen/fuser/cuda/fused_kernel.cpp
      ${TORCH_SRC_DIR}/csrc/autograd/profiler_cuda.cpp
      ${TORCH_SRC_DIR}/csrc/autograd/functions/comm.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/comm.cpp
    )
    add_library(caffe2_nvrtc SHARED ${ATen_NVRTC_STUB_SRCS})
    target_link_libraries(caffe2_nvrtc ${CUDA_NVRTC} ${CUDA_CUDA_LIB} ${CUDA_NVRTC_LIB})
    target_include_directories(caffe2_nvrtc PRIVATE ${CUDA_INCLUDE_DIRS})
    install(TARGETS caffe2_nvrtc DESTINATION "${TORCH_INSTALL_LIB_DIR}")
    if (USE_NCCL)
      list(APPEND Caffe2_GPU_SRCS
        ${TORCH_SRC_DIR}/csrc/cuda/nccl.cpp)
    endif()
  endif()

  if (USE_ROCM)
    list(APPEND Caffe2_HIP_SRCS
      ${TORCH_SRC_DIR}/csrc/jit/codegen/fuser/cuda/fused_kernel.cpp
      ${TORCH_SRC_DIR}/csrc/autograd/profiler_cuda.cpp
      ${TORCH_SRC_DIR}/csrc/autograd/functions/comm.cpp
      ${TORCH_SRC_DIR}/csrc/cuda/comm.cpp
    )
    if (USE_NCCL)
      list(APPEND Caffe2_HIP_SRCS
        ${TORCH_SRC_DIR}/csrc/cuda/nccl.cpp)
    endif()
    # caffe2_nvrtc's stubs to driver APIs are useful for HIP.
    # See NOTE [ ATen NVRTC Stub and HIP ]
    add_library(caffe2_nvrtc SHARED ${ATen_NVRTC_STUB_SRCS})
    target_link_libraries(caffe2_nvrtc ${PYTORCH_HIP_HCC_LIBRARIES} ${ROCM_HIPRTC_LIB})
    target_compile_definitions(caffe2_nvrtc PRIVATE USE_ROCM __HIP_PLATFORM_HCC__)
    install(TARGETS caffe2_nvrtc DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endif()

  if (NOT NO_API)
    list(APPEND TORCH_SRCS
      ${TORCH_SRC_DIR}/csrc/api/src/cuda.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/data/datasets/mnist.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/distributed.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/random.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/sequential.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/stream.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/enum.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/serialize.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/jit.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/init.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/module.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/_functions.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/activation.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/batchnorm.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/normalization.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/instancenorm.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/conv.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/dropout.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/distance.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/embedding.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/fold.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/linear.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/loss.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/padding.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/pixelshuffle.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/pooling.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/rnn.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/upsampling.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/container/functional.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/container/named_any.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/options/activation.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/options/batchnorm.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/options/embedding.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/options/instancenorm.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/options/normalization.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/options/conv.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/options/dropout.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/options/linear.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/options/padding.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/options/pooling.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/options/rnn.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/nn/options/vision.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/optim/adagrad.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/optim/adam.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/optim/lbfgs.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/optim/optimizer.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/optim/rmsprop.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/optim/serialize.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/optim/sgd.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/serialize/input-archive.cpp
      ${TORCH_SRC_DIR}/csrc/api/src/serialize/output-archive.cpp
    )
  endif()

  list(APPEND Caffe2_CPU_SRCS ${TORCH_SRCS})
endif()

# ==========================================================
# END formerly-libtorch sources
# ==========================================================


add_library(torch_cpu ${Caffe2_CPU_SRCS})
torch_compile_options(torch_cpu)  # see cmake/public/utils.cmake

# This is required for older versions of CMake, which don't allow
# specifying add_library() without a list of source files
set(DUMMY_EMPTY_FILE ${CMAKE_BINARY_DIR}/empty.cpp)

if (MSVC)
  set(DUMMY_FILE_CONTENT "__declspec(dllexport) int ignore_this_library_placeholder(){return 0\\;}")
else()
  set(DUMMY_FILE_CONTENT "")
endif()

file(WRITE ${DUMMY_EMPTY_FILE} ${DUMMY_FILE_CONTENT})

# Wrapper library for people who link against torch and expect both CPU and CUDA support
# Contains "torch_cpu" and "torch_cuda"
add_library(torch ${DUMMY_EMPTY_FILE})

if(USE_ROCM)
  filter_list(__caffe2_hip_srcs_cpp Caffe2_HIP_SRCS "\\.(cu|hip)$")
  set_source_files_properties(${__caffe2_hip_srcs_cpp} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
endif()

# Compile exposed libraries.
IF (USE_ROCM)
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  hip_add_library(torch_hip ${Caffe2_HIP_SRCS})
  set(CUDA_LINK_LIBRARIES_KEYWORD)
  torch_compile_options(torch_hip)  # see cmake/public/utils.cmake
  # TODO: Not totally sure if this is live or not
  if (USE_NCCL)
    target_link_libraries(torch_hip PRIVATE __caffe2_nccl)
    target_compile_definitions(torch_hip PRIVATE USE_NCCL)
  endif()
ELSEIF(USE_CUDA)
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  cuda_add_library(torch_cuda ${Caffe2_GPU_SRCS})
  set(CUDA_LINK_LIBRARIES_KEYWORD)
  torch_compile_options(torch_cuda)  # see cmake/public/utils.cmake
  if (USE_NCCL)
    target_link_libraries(torch_cuda PRIVATE __caffe2_nccl)
    target_compile_definitions(torch_cuda PRIVATE USE_NCCL)
  endif()
ENDIF()


if (NOT MSVC)
  TARGET_LINK_LIBRARIES(torch_cpu PRIVATE fxdiv)
endif()

# ==========================================================
# formerly-libtorch flags
# ==========================================================

if (NOT INTERN_BUILD_MOBILE)
  # Forces caffe2.pb.h to be generated before its dependents are compiled.
  # Adding the generated header file to the ${TORCH_SRCS} list is not sufficient
  # to establish the dependency, since the generation procedure is declared in a different CMake file.
  # See https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/#custom-commands-in-different-directories
  add_dependencies(torch_cpu Caffe2_PROTO)
endif()

if (NOT INTERN_BUILD_MOBILE OR NOT BUILD_CAFFE2_MOBILE)
  if (NOT NO_API)
    target_include_directories(torch_cpu PRIVATE
      ${TORCH_SRC_DIR}/csrc/api
      ${TORCH_SRC_DIR}/csrc/api/include)
  endif()

  if(USE_CUDA)
    if(MSVC)
      if (NOT NVTOOLEXT_HOME)
        set(NVTOOLEXT_HOME "C:/Program Files/NVIDIA Corporation/NvToolsExt")
      endif()
      if (DEFINED ENV{NVTOOLSEXT_PATH})
        set(NVTOOLEXT_HOME $ENV{NVTOOLSEXT_PATH})
        file(TO_CMAKE_PATH ${NVTOOLEXT_HOME} NVTOOLEXT_HOME)
      endif()
      set(TORCH_CUDA_LIBRARIES
        ${NVTOOLEXT_HOME}/lib/x64/nvToolsExt64_1.lib
        ${CUDA_LIBRARIES})
      target_include_directories(torch_cuda PUBLIC "${NVTOOLEXT_HOME}/include")

    elseif(APPLE)
      set(TORCH_CUDA_LIBRARIES
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/libcudart.dylib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvrtc.dylib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvToolsExt.dylib
        ${CUDA_LIBRARIES})

    else()
      find_library(LIBNVTOOLSEXT libnvToolsExt.so PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
      set(TORCH_CUDA_LIBRARIES
        ${LIBNVTOOLSEXT}
        ${CUDA_LIBRARIES})
    endif()
  endif()


  set(TH_CPU_INCLUDE
    # dense
    aten/src/TH
    ${CMAKE_CURRENT_BINARY_DIR}/aten/src/TH
    ${TORCH_ROOT}/aten/src
    ${CMAKE_CURRENT_BINARY_DIR}/aten/src
    ${CMAKE_BINARY_DIR}/aten/src)
  target_include_directories(torch_cpu PRIVATE ${TH_CPU_INCLUDE})

  set(ATen_CPU_INCLUDE
    ${TORCH_ROOT}/aten/src
    ${CMAKE_CURRENT_BINARY_DIR}/../aten/src
    ${CMAKE_CURRENT_BINARY_DIR}/../aten/src/ATen
    ${CMAKE_BINARY_DIR}/aten/src)

IF (USE_TBB)
  list(APPEND ATen_CPU_INCLUDE ${TBB_ROOT_DIR}/include)
  target_link_libraries(torch_cpu PUBLIC tbb)
ENDIF()


  target_include_directories(torch_cpu PRIVATE ${ATen_CPU_INCLUDE})

  target_include_directories(torch_cpu PRIVATE
    ${TORCH_SRC_DIR}/csrc)

  target_include_directories(torch_cpu PRIVATE
    ${TORCH_ROOT}/third_party/miniz-2.0.8)


  install(DIRECTORY "${TORCH_SRC_DIR}/csrc"
    DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch
    FILES_MATCHING PATTERN "*.h")
  install(FILES "${TORCH_SRC_DIR}/script.h" "${TORCH_SRC_DIR}/extension.h" "${TORCH_SRC_DIR}/custom_class.h" "${TORCH_SRC_DIR}/custom_class_detail.h"
    DESTINATION ${TORCH_INSTALL_INCLUDE_DIR}/torch)


  if (BUILD_TEST AND NOT MSVC AND NOT USE_ROCM)
    add_subdirectory(${TORCH_ROOT}/test/cpp/jit ${CMAKE_BINARY_DIR}/test_jit)
    add_subdirectory(${TORCH_ROOT}/test/cpp/tensorexpr ${CMAKE_BINARY_DIR}/test_tensorexpr)
    if (USE_DISTRIBUTED)
      add_subdirectory(${TORCH_ROOT}/test/cpp/rpc ${CMAKE_BINARY_DIR}/test_cpp_rpc)
    endif()
  endif()

  if (BUILD_TEST AND NOT NO_API)
    add_subdirectory(${TORCH_ROOT}/test/cpp/api ${CMAKE_BINARY_DIR}/test_api)
    add_subdirectory(${TORCH_ROOT}/test/cpp/dist_autograd ${CMAKE_BINARY_DIR}/dist_autograd)
  endif()

  # XXX This ABI check cannot be run with arm-linux-androideabi-g++
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (DEFINED GLIBCXX_USE_CXX11_ABI)
      message(STATUS "_GLIBCXX_USE_CXX11_ABI is already defined as a cmake variable")
    else()
      message(STATUS "${CMAKE_CXX_COMPILER} ${TORCH_SRC_DIR}/abi-check.cpp -o ${CMAKE_BINARY_DIR}/abi-check")
      execute_process(
        COMMAND
        "${CMAKE_CXX_COMPILER}"
        "${TORCH_SRC_DIR}/abi-check.cpp"
        "-o"
        "${CMAKE_BINARY_DIR}/abi-check"
        RESULT_VARIABLE ABI_CHECK_COMPILE_RESULT)
      if (ABI_CHECK_COMPILE_RESULT)
        message(FATAL_ERROR "Could not compile ABI Check: ${ABI_CHECK_COMPILE_RESULT}")
      endif()
      execute_process(
        COMMAND "${CMAKE_BINARY_DIR}/abi-check"
        RESULT_VARIABLE ABI_CHECK_RESULT
        OUTPUT_VARIABLE GLIBCXX_USE_CXX11_ABI)
      if (ABI_CHECK_RESULT)
        message(WARNING "Could not run ABI Check: ${ABI_CHECK_RESULT}")
      endif()
    endif()
    message(STATUS "Determined _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
  endif()

  # CMake config for external projects.
  configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/TorchConfigVersion.cmake.in
    ${PROJECT_BINARY_DIR}/TorchConfigVersion.cmake
    @ONLY)
  configure_file(
    ${TORCH_ROOT}/cmake/TorchConfig.cmake.in
    ${PROJECT_BINARY_DIR}/TorchConfig.cmake
    @ONLY)
  install(FILES
    ${PROJECT_BINARY_DIR}/TorchConfigVersion.cmake
    ${PROJECT_BINARY_DIR}/TorchConfig.cmake
    DESTINATION share/cmake/Torch)

  if (USE_DISTRIBUTED)
    if (NOT MSVC)
      add_subdirectory(${TORCH_SRC_DIR}/lib/c10d lib_c10d)
    endif()
  endif()


  # ---[ Torch python bindings build
  add_subdirectory(../torch torch)


endif()
# ==========================================================
# END formerly-libtorch flags
# ==========================================================






if (NOT NO_API)
  target_include_directories(torch_cpu PUBLIC
    $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/api>
    $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/api/include>)
endif()


if(USE_OPENMP)
  find_package(OpenMP QUIET)
endif()
if(USE_OPENMP AND OPENMP_FOUND)
  message(STATUS "pytorch is compiling with OpenMP. \n"
    "OpenMP CXX_FLAGS: ${OpenMP_CXX_FLAGS}. \n"
    "OpenMP libraries: ${OpenMP_CXX_LIBRARIES}.")
  target_compile_options(torch_cpu INTERFACE ${OpenMP_CXX_FLAGS})
  target_link_libraries(torch_cpu PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()


if(USE_ROCM)
  target_compile_definitions(torch_hip PRIVATE
    USE_ROCM
    __HIP_PLATFORM_HCC__
    )
  # NB: Massive hack.  torch/csrc/jit/codegen/fuser/codegen.cpp includes
  # torch/csrc/jit/codegen/fuser/cuda/resource_strings.h which changes the
  # strings depending on if you're __HIP_PLATFORM_HCC__ or not.
  # But that file is in torch_cpu!  So, against all odds, this macro
  # has to be set on torch_cpu too.  I also added it to torch for
  # better luck
  target_compile_definitions(torch_cpu PRIVATE
    USE_ROCM
    __HIP_PLATFORM_HCC__
    )
  target_compile_definitions(torch PRIVATE
    USE_ROCM
    __HIP_PLATFORM_HCC__
    )
  target_include_directories(torch_hip PRIVATE
    /opt/rocm/include
    /opt/rocm/hcc/include
    /opt/rocm/rocblas/include
    /opt/rocm/hipsparse/include
    )
endif()

# Pass USE_DISTRIBUTED to torch_cpu, as some codes in jit/pickler.cpp and
# jit/unpickler.cpp need to be compiled only when USE_DISTRIBUTED is set
if (USE_DISTRIBUTED)
  target_compile_definitions(torch_cpu PRIVATE
    USE_DISTRIBUTED
  )
endif()

if (NOT INTERN_BUILD_MOBILE OR BUILD_CAFFE2_MOBILE)
  caffe2_interface_library(caffe2_protos caffe2_protos_whole)
  target_link_libraries(torch_cpu PRIVATE caffe2_protos_whole)
  if (${CAFFE2_LINK_LOCAL_PROTOBUF})
    target_link_libraries(torch_cpu INTERFACE protobuf::libprotobuf)
  else()
    target_link_libraries(torch_cpu PUBLIC protobuf::libprotobuf)
  endif()
endif()

if (USE_OPENMP AND OPENMP_FOUND)
  message(STATUS "Caffe2 is compiling with OpenMP. \n"
    "OpenMP CXX_FLAGS: ${OpenMP_CXX_FLAGS}. \n"
    "OpenMP libraries: ${OpenMP_CXX_LIBRARIES}.")
  target_link_libraries(torch_cpu PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()

target_link_libraries(torch_cpu PUBLIC c10)
target_link_libraries(torch_cpu PUBLIC ${Caffe2_PUBLIC_DEPENDENCY_LIBS})
target_link_libraries(torch_cpu PRIVATE ${Caffe2_DEPENDENCY_LIBS})
target_link_libraries(torch_cpu PRIVATE ${Caffe2_DEPENDENCY_WHOLE_LINK_LIBS})
target_include_directories(torch_cpu INTERFACE $<INSTALL_INTERFACE:include>)
target_include_directories(torch_cpu PRIVATE ${Caffe2_CPU_INCLUDE})
target_include_directories(torch_cpu SYSTEM PRIVATE "${Caffe2_DEPENDENCY_INCLUDE}")
# Set standard properties on the target
torch_set_target_props(torch_cpu)


target_compile_options(torch_cpu PRIVATE "-DCAFFE2_BUILD_MAIN_LIB")
if(USE_CUDA)
  target_compile_options(torch_cuda PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
  # NB: This must be target_compile_definitions, not target_compile_options,
  # as the latter is not respected by nvcc
  target_compile_definitions(torch_cuda PRIVATE "-DTORCH_CUDA_BUILD_MAIN_LIB")
elseif(USE_ROCM)
  target_compile_options(torch_hip PRIVATE "-DTORCH_HIP_BUILD_MAIN_LIB")
  target_compile_definitions(torch_hip PRIVATE "-DTORCH_HIP_BUILD_MAIN_LIB")
endif()


# ATen parallelism settings
#  OMP - OpenMP for intra-op, native thread pool for inter-op parallelism
#  NATIVE - using native thread pool for intra- and inter-op parallelism
#  TBB - using TBB for intra- and native thread pool for inter-op parallelism
if (INTERN_BUILD_MOBILE AND NOT BUILD_CAFFE2_MOBILE)
  set(ATEN_THREADING "NATIVE" CACHE STRING "ATen parallel backend")
else()
  set(ATEN_THREADING "OMP" CACHE STRING "ATen parallel backend")
endif()

message(STATUS "Using ATen parallel backend: ${ATEN_THREADING}")
if ("${ATEN_THREADING}" STREQUAL "OMP")
  target_compile_definitions(torch_cpu PUBLIC "-DAT_PARALLEL_OPENMP=1")
elseif ("${ATEN_THREADING}" STREQUAL "NATIVE")
  target_compile_definitions(torch_cpu PUBLIC "-DAT_PARALLEL_NATIVE=1")
elseif ("${ATEN_THREADING}" STREQUAL "TBB")
  if (NOT USE_TBB)
    message(FATAL_ERROR "Using TBB backend but USE_TBB is off")
  endif()
  target_compile_definitions(torch_cpu PUBLIC "-DAT_PARALLEL_NATIVE_TBB=1")
else()
  message(FATAL_ERROR "Unknown ATen parallel backend: ${ATEN_THREADING}")
endif()
set(EXPERIMENTAL_SINGLE_THREAD_POOL "0" CACHE STRING
  "Experimental option to use a single thread pool for inter- and intra-op parallelism")
if ("${EXPERIMENTAL_SINGLE_THREAD_POOL}")
  target_compile_definitions(torch_cpu PUBLIC "-DAT_EXPERIMENTAL_SINGLE_THREAD_POOL=1")
endif()

if (MSVC AND NOT BUILD_SHARED_LIBS)
  # Note [Supporting both static and dynamic libraries on Windows]
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # A Windows library may be distributed as either a static or dynamic
  # library.  The chosen distribution mechanism affects how you setup
  # the headers for the library: if you statically link a function,
  # all you need is an ordinary signature:
  #
  #     void f();
  #
  # But if you *dynamically* link it, then you must provide a __declspec
  # specifying that it should be imported from a DLL:
  #
  #     __declspec(dllimport) void f();
  #
  # Mixing the two situations will not work: if you specify dllimport
  # while statically linking, the linker will complain it cannot find
  # the __imp_f symbol (which serve as the DLL entrypoint); if you
  # fail to specify dllimport for a symbol that's coming from a DLL,
  # the linker will complain that it can't find f.  Joy!
  #
  # Most places on the Internet, you will find people have written
  # their headers under the assumption that the application will
  # only ever be dynamically linked, as they define a macro which
  # tags a function as __declspec(dllexport) if you are actually
  # building the library, and __declspec(dllimport) otherwise.  But
  # if you want these headers to also work if you are linking against
  # a static library, you need a way to avoid adding these __declspec's
  # at all.  And that "mechanism" needs to apply to any downstream
  # libraries/executables which are going to link against your library.
  #
  #   As an aside, why do we need to support both modes?
  #   For historical reasons, PyTorch ATen on Windows is built dynamically,
  #   while Caffe2 on Windows is built statically (mostly because if
  #   we build it dynamically, we are over the DLL exported symbol limit--and
  #   that is because Caffe2 hasn't comprehensively annotated all symbols
  #   which cross the DLL boundary with CAFFE_API).  So any code
  #   which is used by both PyTorch and Caffe2 needs to support both
  #   modes of linking.
  #
  # So, you have a macro (call it AT_CORE_STATIC_WINDOWS) which you need to have
  # set for any downstream library/executable that transitively includes your
  # headers.  How are you going to do this?  You have two options:
  #
  #   1. Write out a config.h header which stores whether or not
  #      you are linking statically or dynamically.
  #
  #   2. Force all of users to set the the macro themselves.  If they
  #      use cmake, you can set -DAT_CORE_STATIC_WINDOWS=1 as a PUBLIC
  #      compile option, in which case cmake will automatically
  #      add the macro for you.
  #
  # Which one is better? Well, it depends: they trade off implementor
  # ease versus user ease: (1) is more work for the library author
  # but the user doesn't have to worry about it; (2) requires the user
  # to set the macro themselves... but only if they don't use cmake.
  #
  # So, which is appropriate in our situation?  In my mind, here is
  # the distinguishing factor: it is more common to distribute
  # DLLs, since they don't require you to line up the CRT version
  # (/MD, /MDd, /MT, /MTd) and MSVC version at the use site.  So,
  # if a user is already in the business of static linkage, they're
  # already in "expert user" realm.  So, I've decided that at this
  # point in time, the simplicity of implementation of (2) wins out.
  #
  # NB: This must be target_compile_definitions, not target_compile_options,
  # as the latter is not respected by nvcc
  target_compile_definitions(torch_cpu PUBLIC "AT_CORE_STATIC_WINDOWS=1")
endif()
if (MSVC AND BUILD_SHARED_LIBS)
  # ONNX is linked statically and needs to be exported from this library
  # to be used externally. Make sure that references match the export.
  target_compile_options(torch_cpu PRIVATE "-DONNX_BUILD_MAIN_LIB")
endif()

caffe2_interface_library(torch_cpu torch_cpu_library)

if (USE_CUDA)
  caffe2_interface_library(torch_cuda torch_cuda_library)
elseif (USE_ROCM)
  caffe2_interface_library(torch_hip torch_hip_library)
endif()

caffe2_interface_library(torch torch_library)

install(TARGETS torch_cpu torch_cpu_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")
if (USE_CUDA)
  install(TARGETS torch_cuda torch_cuda_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")
elseif (USE_ROCM)
  install(TARGETS torch_hip torch_hip_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()
install(TARGETS torch torch_library EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")

target_link_libraries(torch PUBLIC torch_cpu_library)
if(USE_CUDA)
  target_link_libraries(torch PUBLIC torch_cuda_library)
elseif(USE_ROCM)
  target_link_libraries(torch PUBLIC torch_hip_library)
endif()

list(APPEND Caffe2_MAIN_LIBS torch_library)
if (USE_TBB)
  list(APPEND Caffe2_MAIN_LIBS tbb)
endif()

# Install PDB files for MSVC builds
if (MSVC AND BUILD_SHARED_LIBS)
  install(FILES $<TARGET_PDB_FILE:torch_cpu> DESTINATION "${TORCH_INSTALL_LIB_DIR}" OPTIONAL)
  if(USE_CUDA)
    install(FILES $<TARGET_PDB_FILE:torch_cuda> DESTINATION "${TORCH_INSTALL_LIB_DIR}" OPTIONAL)
  elseif(USE_ROCM)
    install(FILES $<TARGET_PDB_FILE:torch_hip> DESTINATION "${TORCH_INSTALL_LIB_DIR}" OPTIONAL)
  endif()
endif()

# ---[ CUDA library.
if(USE_CUDA)

  target_link_libraries(torch_cuda INTERFACE torch::cudart)
  target_link_libraries(torch_cuda PUBLIC c10_cuda)

  target_link_libraries(torch_cuda PUBLIC ${TORCH_CUDA_LIBRARIES})

  target_include_directories(
      torch_cuda INTERFACE $<INSTALL_INTERFACE:include>)
  target_include_directories(
      torch_cuda PRIVATE ${Caffe2_GPU_INCLUDE})
  target_link_libraries(
      torch_cuda PRIVATE ${Caffe2_CUDA_DEPENDENCY_LIBS})

  # These public dependencies must go after the previous dependencies, as the
  # order of the libraries in the linker call matters here when statically
  # linking; libculibos and cublas must be last.
  target_link_libraries(torch_cuda PUBLIC torch_cpu_library ${Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS})


endif()


# Note [Global dependencies]
# Some libraries (e.g. OpenMPI) like to dlopen plugins after they're initialized,
# and they assume that all of their symbols will be available in the global namespace.
# On the other hand we try to be good citizens and avoid polluting the symbol
# namespaces, so libtorch is loaded with all its dependencies in a local scope.
# That usually leads to missing symbol errors at run-time, so to avoid a situation like
# this we have to preload those libs in a global namespace.
if (BUILD_SHARED_LIBS)
  add_library(torch_global_deps SHARED ${TORCH_SRC_DIR}/csrc/empty.c)
  set_target_properties(torch_global_deps PROPERTIES LINKER_LANGUAGE C)
  if (USE_MPI)
      target_link_libraries(torch_global_deps ${MPI_CXX_LIBRARIES})
  endif()
  target_link_libraries(torch_global_deps ${MKL_LIBRARIES})
  # The CUDA libraries are linked here for a different reason: in some
  # cases we load these libraries with ctypes, and if they weren't opened
  # with RTLD_GLOBAL, we'll do the "normal" search process again (and
  # not find them, because they're usually in non-standard locations)
  if (USE_CUDA)
    target_link_libraries(torch_global_deps ${TORCH_CUDA_LIBRARIES})
    target_link_libraries(torch_global_deps ${Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS})
    target_link_libraries(torch_global_deps torch::cudart)
  endif()

  install(TARGETS torch_global_deps DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()

# ---[ Caffe2 HIP sources.
if(USE_ROCM)
  # Call again since Caffe2_HIP_INCLUDE is extended with ATen include dirs.
  # Get Compile Definitions from the directory (FindHIP.cmake bug)
  get_directory_property(MY_DEFINITIONS COMPILE_DEFINITIONS)
  if(MY_DEFINITIONS)
    foreach(_item ${MY_DEFINITIONS})
      LIST(APPEND HIP_HCC_FLAGS "-D${_item}")
    endforeach()
  endif()

  # Call again since Caffe2_HIP_INCLUDE is extended with ATen include dirs.
  hip_include_directories(${Caffe2_HIP_INCLUDE})

  # Since PyTorch files contain HIP headers, these flags are required for the necessary definitions to be added.
  target_compile_options(torch_hip PUBLIC ${HIP_CXX_FLAGS})  # experiment
  target_link_libraries(torch_hip PUBLIC c10_hip)

  if(NOT INTERN_BUILD_MOBILE)
    # TODO: Cut this over to ATEN_HIP_FILES_GEN_LIB.  At the moment, we
    # only generate CUDA files
    # NB: This dependency must be PRIVATE, because we don't install
    # ATEN_CUDA_FILES_GEN_LIB (it's a synthetic target just to get the
    # correct dependency from generated files.)
    target_link_libraries(torch_hip PRIVATE ATEN_CUDA_FILES_GEN_LIB)
  endif()
  target_link_libraries(torch_hip PUBLIC torch_cpu_library ${Caffe2_HIP_DEPENDENCY_LIBS})

  # Since PyTorch files contain HIP headers, this is also needed to capture the includes.
  target_include_directories(torch_hip PRIVATE ${Caffe2_HIP_INCLUDE})
  target_include_directories(torch_hip INTERFACE $<INSTALL_INTERFACE:include>)
endif()

# ---[ Test binaries.
if (BUILD_TEST)
  foreach(test_src ${Caffe2_CPU_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} gtest_main)
    target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE})
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if (INSTALL_TEST)
      install(TARGETS ${test_name} DESTINATION test)
      # Install PDB files for MSVC builds
      if (MSVC AND BUILD_SHARED_LIBS)
        install(FILES $<TARGET_PDB_FILE:${test_name}> DESTINATION test OPTIONAL)
      endif()
    endif()
  endforeach()

  if (USE_CUDA)
    foreach(test_src ${Caffe2_GPU_TEST_SRCS})
      get_filename_component(test_name ${test_src} NAME_WE)
      cuda_add_executable(${test_name} "${test_src}")
      target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} gtest_main)
      target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
      target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE})
      add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
      if (INSTALL_TEST)
        install(TARGETS ${test_name} DESTINATION test)
        # Install PDB files for MSVC builds
        if (MSVC AND BUILD_SHARED_LIBS)
          install(FILES $<TARGET_PDB_FILE:${test_name}> DESTINATION test OPTIONAL)
        endif()
      endif()
    endforeach()
  endif()

  if(USE_ROCM)
    foreach(test_src ${Caffe2_HIP_TEST_SRCS})
      get_filename_component(test_name ${test_src} NAME_WE)
      add_executable(${test_name} "${test_src}")
      target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} gtest_main)
      target_include_directories(${test_name} PRIVATE $<INSTALL_INTERFACE:include>)
      target_include_directories(${test_name} PRIVATE ${Caffe2_CPU_INCLUDE} ${Caffe2_HIP_INCLUDE})
      target_compile_options(${test_name} PRIVATE ${HIP_CXX_FLAGS})
      add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
      if (INSTALL_TEST)
        install(TARGETS ${test_name} DESTINATION test)
      endif()
    endforeach()
  endif()

  # For special tests that explicitly uses dependencies, we add them here
  if (USE_MPI)
    target_link_libraries(mpi_test ${MPI_CXX_LIBRARIES})
    if (USE_CUDA)
      target_link_libraries(mpi_gpu_test ${MPI_CXX_LIBRARIES})
    endif()
  endif()
endif()

# Note: we only install the caffe2 python files if BUILD_CAFFE2_OPS is ON
# This is because the build rules here written in such a way that they always
# appear to need to be re-run generating >600 pieces of work during the pytorch
# rebuild step. The long-term fix should be to clean up these rules so they
# only rerun when needed.

if (BUILD_PYTHON)
  # Python site-packages
  # Get canonical directory for python site packages (relative to install
  # location).  It varies from system to system.
  # We should pin the path separator to the forward slash on Windows.
  # More details can be seen at
  # https://github.com/pytorch/pytorch/tree/master/tools/build_pytorch_libs.bat#note-backslash-munging-on-windows
  pycmd(PYTHON_SITE_PACKAGES "
      import os
      from distutils import sysconfig
      print(sysconfig.get_python_lib(prefix=''))
  ")
  file(TO_CMAKE_PATH ${PYTHON_SITE_PACKAGES} PYTHON_SITE_PACKAGES)
  SET(PYTHON_SITE_PACKAGES ${PYTHON_SITE_PACKAGES} PARENT_SCOPE) # for Summary
  # ---[ Options.
  SET(PYTHON_LIB_REL_PATH "${PYTHON_SITE_PACKAGES}" CACHE STRING "Python installation path (relative to CMake installation prefix)")
  message(STATUS "Using ${PYTHON_LIB_REL_PATH} as python relative installation path")
  # Python extension suffix
  # Try to get from python through sysconfig.get_env_var('EXT_SUFFIX') first,
  # fallback to ".pyd" if windows and ".so" for all others.
  pycmd(PY_EXT_SUFFIX "
      from distutils import sysconfig
      ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
      print(ext_suffix if ext_suffix else '')
  ")
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if (MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
  endif()

  # Allow different install locations for libcaffe2
  # For setuptools installs (that all build Python), install libcaffe2 into
  # site-packages, alongside the torch libraries. The pybind11 library needs
  # an rpath to the torch library folder
  # For cmake installs, including c++ only installs, install libcaffe2 into
  # CMAKE_INSTALL_PREFIX/lib . The pybind11 library can have a hardcoded
  # rpath
  set(caffe2_pybind11_rpath "${_rpath_portable_origin}")
  if(${BUILDING_WITH_TORCH_LIBS})
    # site-packages/caffe2/python/caffe2_pybind11_state
    # site-packages/torch/lib
    set(caffe2_pybind11_rpath "${_rpath_portable_origin}/../../torch/lib")
  endif(${BUILDING_WITH_TORCH_LIBS})

  # Must also include `CMAKE_SHARED_LINKER_FLAGS` in linker flags for
  # `caffe2_pybind11_state_*` targets because paths to required libraries may
  # need to be found there (e.g., specifying path to `libiomp5` with `LDFLAGS`).
  set(_caffe2_pybind11_state_linker_flags "${CMAKE_SHARED_LINKER_FLAGS}")
  if (APPLE)
    set(_caffe2_pybind11_state_linker_flags "${_caffe2_pybind11_state_linker_flags} -undefined dynamic_lookup")
  endif()

  # ---[ Python.
  add_library(caffe2_pybind11_state MODULE ${Caffe2_CPU_PYTHON_SRCS})
  if (NOT MSVC)
    set_target_properties(caffe2_pybind11_state PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  endif()
  set_target_properties(caffe2_pybind11_state PROPERTIES PREFIX "" DEBUG_POSTFIX "")
  set_target_properties(caffe2_pybind11_state PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  set_target_properties(caffe2_pybind11_state PROPERTIES LINK_FLAGS "${_caffe2_pybind11_state_linker_flags}")
  target_include_directories(caffe2_pybind11_state PRIVATE $<INSTALL_INTERFACE:include>)
  target_include_directories(caffe2_pybind11_state PRIVATE ${Caffe2_CPU_INCLUDE})

  target_link_libraries(
      caffe2_pybind11_state torch_library)
  if (WIN32)
    target_link_libraries(caffe2_pybind11_state ${PYTHON_LIBRARIES})
    target_link_libraries(caffe2_pybind11_state onnx_proto)
  endif(WIN32)

  # Install caffe2_pybind11_state(_gpu|hip) in site-packages/caffe2/python,
  # so it needs an rpath to find libcaffe2
  set_target_properties(
      caffe2_pybind11_state PROPERTIES LIBRARY_OUTPUT_DIRECTORY
      ${CMAKE_BINARY_DIR}/caffe2/python)
  install(TARGETS caffe2_pybind11_state DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python")
  if (MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:caffe2_pybind11_state> DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python" OPTIONAL)
  endif()
  set_target_properties(caffe2_pybind11_state PROPERTIES INSTALL_RPATH "${caffe2_pybind11_rpath}")

  if(USE_CUDA)
    add_library(caffe2_pybind11_state_gpu MODULE ${Caffe2_GPU_PYTHON_SRCS})
    if (NOT MSVC)
      set_target_properties(caffe2_pybind11_state_gpu PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
    endif()
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES PREFIX "" DEBUG_POSTFIX "")
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES LINK_FLAGS "${_caffe2_pybind11_state_linker_flags}")
    target_include_directories(caffe2_pybind11_state_gpu PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(caffe2_pybind11_state_gpu PRIVATE ${Caffe2_CPU_INCLUDE})
    target_link_libraries(caffe2_pybind11_state_gpu torch_library)
    if (WIN32)
      target_link_libraries(caffe2_pybind11_state_gpu ${PYTHON_LIBRARIES})
      target_link_libraries(caffe2_pybind11_state_gpu onnx_proto)
    endif(WIN32)

    # Install with same rpath as non-gpu caffe2_pybind11_state
    set_target_properties(
        caffe2_pybind11_state_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/caffe2/python)
    install(TARGETS caffe2_pybind11_state_gpu DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python")
    if (MSVC AND BUILD_SHARED_LIBS)
      install(FILES $<TARGET_PDB_FILE:caffe2_pybind11_state_gpu> DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python" OPTIONAL)
    endif()
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES INSTALL_RPATH "${caffe2_pybind11_rpath}")
  endif()

  if(USE_ROCM)
    add_library(caffe2_pybind11_state_hip MODULE ${Caffe2_HIP_PYTHON_SRCS})
    if (NOT MSVC)
      target_compile_options(caffe2_pybind11_state_hip PRIVATE ${HIP_CXX_FLAGS} -fvisibility=hidden)
    endif()
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES PREFIX "")
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES LINK_FLAGS "${_caffe2_pybind11_state_linker_flags}")
    target_include_directories(caffe2_pybind11_state_hip PRIVATE $<INSTALL_INTERFACE:include>)
    target_include_directories(caffe2_pybind11_state_hip PRIVATE ${Caffe2_CPU_INCLUDE} ${Caffe2_HIP_INCLUDE})
    target_link_libraries(caffe2_pybind11_state_hip torch_library)
    if (WIN32)
      target_link_libraries(caffe2_pybind11_state_hip ${PYTHON_LIBRARIES})
    endif(WIN32)

    # Install with same rpath as non-hip caffe2_pybind11_state
    set_target_properties(
        caffe2_pybind11_state_hip PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/caffe2/python)
    install(TARGETS caffe2_pybind11_state_hip DESTINATION "${PYTHON_LIB_REL_PATH}/caffe2/python")
    set_target_properties(caffe2_pybind11_state_hip PROPERTIES INSTALL_RPATH "${caffe2_pybind11_rpath}")
  endif()

  if (MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
    # If we are building under windows, we will copy the file from
    # build/caffe2/python/{Debug,Release}/caffe2_pybind11_state.pyd
    # to its parent folder so that we can do in-build execution.
    add_custom_target(windows_python_copy_lib ALL)
    add_dependencies(windows_python_copy_lib caffe2_pybind11_state)
    add_custom_command(
        TARGET windows_python_copy_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:caffe2_pybind11_state>
        ${CMAKE_BINARY_DIR}/caffe2/python)
    if (USE_CUDA)
      add_dependencies(windows_python_copy_lib caffe2_pybind11_state_gpu)
      add_custom_command(
          TARGET windows_python_copy_lib POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:caffe2_pybind11_state_gpu>
          ${CMAKE_BINARY_DIR}/caffe2/python)
    endif()
    if (USE_ROCM)
      add_dependencies(windows_python_copy_lib caffe2_pybind11_state_hip)
      add_custom_command(
          TARGET windows_python_copy_lib POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:caffe2_pybind11_state_hip>
          ${CMAKE_BINARY_DIR}/caffe2/python)
    endif()
  endif()

  # Finally, Copy all python files to build directory
  # Create a custom target that copies all python files.
  file(GLOB_RECURSE PYTHON_SRCS RELATIVE ${PROJECT_SOURCE_DIR}
       "${PROJECT_SOURCE_DIR}/caffe2/*.py")

  # generated pb files are copied from build/caffe2 to caffe2
  # if we copied them back to build this would create a build cycle
  # consider removing the need for globs
  filter_list_exclude(PYTHON_SRCS PYTHON_SRCS "proto/.*_pb")

  set(build_files)
  foreach(python_src ${PYTHON_SRCS})
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/${python_src}
                       DEPENDS ${PROJECT_SOURCE_DIR}/${python_src}
                       COMMAND ${CMAKE_COMMAND} -E copy
                       ${PROJECT_SOURCE_DIR}/${python_src}
                       ${CMAKE_BINARY_DIR}/${python_src})
    list(APPEND build_files ${CMAKE_BINARY_DIR}/${python_src})
  endforeach()

  add_custom_target(python_copy_files ALL DEPENDS ${build_files})


  # Install commands
  # Pick up static python files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe2 DESTINATION ${PYTHON_LIB_REL_PATH}
          FILES_MATCHING PATTERN "*.py")
  # Caffe proto files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe DESTINATION ${PYTHON_LIB_REL_PATH}
          FILES_MATCHING PATTERN "*.py")
  # Caffe2 proto files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe2 DESTINATION ${PYTHON_LIB_REL_PATH}
          FILES_MATCHING PATTERN "*.py")
endif()

# Finally, set the Caffe2_MAIN_LIBS variable in the parent scope.
set(Caffe2_MAIN_LIBS ${Caffe2_MAIN_LIBS} PARENT_SCOPE)
