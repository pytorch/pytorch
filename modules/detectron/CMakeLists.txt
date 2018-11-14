file(GLOB Detectron_CPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
file(GLOB Detectron_GPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB_RECURSE Detectron_HIP_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*_hip.cc)

if (BUILD_CAFFE2_OPS)
  # Note(ilijar): Since Detectron ops currently have no
  # CPU implementation, we only build GPU ops for now.
  if (USE_CUDA)
    CUDA_ADD_LIBRARY(
        caffe2_detectron_ops_gpu SHARED
        ${Detectron_CPU_SRCS}
        ${Detectron_GPU_SRCS})

    target_link_libraries(caffe2_detectron_ops_gpu caffe2_gpu)
    install(TARGETS caffe2_detectron_ops_gpu DESTINATION lib)

  elseif(0)
  # To-do: Enable it with proper compilation flags for appropriate sources
  # elseif(USE_ROCM)
    set(Caffe2_HIP_INCLUDES ${Caffe2_HIP_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR})
    hip_include_directories(${Caffe2_HIP_INCLUDES})
    set_source_files_properties(${Detectron_HIP_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    HIP_ADD_LIBRARY(
        caffe2_detectron_ops_hip SHARED
        ${Detectron_CPU_SRCS}
        ${Detectron_HIP_SRCS})
    set_target_properties(caffe2_detectron_ops_hip PROPERTIES LINKER_LANGUAGE HIP)
    target_compile_options(caffe2_detectron_ops_hip PRIVATE ${HIP_HCC_FLAGS})
    target_link_libraries(caffe2_detectron_ops_hip caffe2_hip)
    install(TARGETS caffe2_detectron_ops_hip DESTINATION lib)

  elseif(NOT IOS_PLATFORM)
    add_library(caffe2_detectron_ops SHARED ${Detectron_CPU_SRCS})
    target_link_libraries(caffe2_detectron_ops caffe2)
    install(TARGETS caffe2_detectron_ops DESTINATION lib)
  endif()
endif()
