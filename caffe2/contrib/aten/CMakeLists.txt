if(USE_ATEN)
  if(NOT USE_CUDA)
    set(NO_CUDA ON)
  endif()
  set(TORCH_CUDA_ARCH_LIST "3.5 5.2 6.0 6.1+PTX")
  set(TORCH_NVCC_FLAGS "-Xfatbin -compress-all")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(AT_LINK_STYLE STATIC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/aten aten EXCLUDE_FROM_ALL)

  add_custom_command(OUTPUT aten_op.h
  COMMAND
  python ${CMAKE_CURRENT_SOURCE_DIR}/gen_op.py
  --third_party_root=${PROJECT_SOURCE_DIR}/third_party
  --template_dir=${PROJECT_SOURCE_DIR}/caffe2/contrib/aten
  DEPENDS
  ATen
  ${CMAKE_CURRENT_SOURCE_DIR}/gen_op.py
  ${CMAKE_CURRENT_SOURCE_DIR}/aten_op_template.h)

  add_custom_target(__aten_op_header_gen DEPENDS aten_op.h)
  add_library(aten_op_header_gen INTERFACE)
  add_dependencies(aten_op_header_gen __aten_op_header_gen)

  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} "${CMAKE_CURRENT_SOURCE_DIR}/aten_op.cc" PARENT_SCOPE)
  set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} "${CMAKE_CURRENT_SOURCE_DIR}/aten_op_cuda.cc" PARENT_SCOPE)
endif()
