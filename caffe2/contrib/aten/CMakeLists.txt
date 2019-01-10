if(BUILD_ATEN)
  # TODO: Combine this with caffe2/ version of the aten target
  # Unfortunately the custom command below requires this to be here
  set(__caffe2_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(ATEN_NO_TEST ON)
  set(AT_LINK_STYLE STATIC)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../aten aten)
  set(CMAKE_POSITION_INDEPENDENT_CODE ${__caffe2_CMAKE_POSITION_INDEPENDENT_CODE})

  add_custom_command(OUTPUT aten_op.h
  COMMAND
  python ${CMAKE_CURRENT_SOURCE_DIR}/gen_op.py
  --aten_root=${CMAKE_CURRENT_SOURCE_DIR}/../../../aten
  --template_dir=${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS
  ATen_cpu
  ${CMAKE_CURRENT_SOURCE_DIR}/gen_op.py
  ${CMAKE_CURRENT_SOURCE_DIR}/aten_op_template.h)

  add_custom_target(__aten_op_header_gen DEPENDS aten_op.h)
  add_library(aten_op_header_gen INTERFACE)
  add_dependencies(aten_op_header_gen __aten_op_header_gen)

  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/aten_op.cc PARENT_SCOPE)
  set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/aten_op_cuda.cc PARENT_SCOPE)
endif()
