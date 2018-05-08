if(BUILD_ATEN)
  add_custom_command(OUTPUT aten_op.h
  COMMAND
  python ${CMAKE_CURRENT_SOURCE_DIR}/gen_op.py
  --aten_root=${PROJECT_SOURCE_DIR}/aten
  --template_dir=${PROJECT_SOURCE_DIR}/caffe2/contrib/aten
  DEPENDS
  ATen
  ${CMAKE_CURRENT_SOURCE_DIR}/gen_op.py
  ${CMAKE_CURRENT_SOURCE_DIR}/aten_op_template.h)

  add_custom_target(__aten_op_header_gen DEPENDS aten_op.h)
  add_library(aten_op_header_gen INTERFACE)
  add_dependencies(aten_op_header_gen __aten_op_header_gen)
  list(APPEND Caffe2_DEPENDENCY_LIBS aten_op_header_gen ATen)
  if (USE_CUDA)
    list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS aten_op_header_gen ATen)
  endif()

  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/aten_op.cc PARENT_SCOPE)
  set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/aten_op_cuda.cc PARENT_SCOPE)

  include_directories(${PROJECT_BINARY_DIR}/caffe2/contrib/aten)
endif()
