file(GLOB Detectron_CPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
file(GLOB Detectron_GPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

if (BUILD_CAFFE2_OPS)
  if (USE_OPENMP AND OPENMP_FOUND)
    Set(OpenMP_link ${OpenMP_CXX_LIBRARIES})
  endif()

  # Note(ilijar): Since Detectron ops currently have no
  # CPU implementation, we only build GPU ops for now.
  if (USE_CUDA)
    CUDA_ADD_LIBRARY(
        caffe2_detectron_ops_gpu SHARED
        ${Detectron_CPU_SRCS}
        ${Detectron_GPU_SRCS})

    target_link_libraries(caffe2_detectron_ops_gpu caffe2_gpu ${OpenMP_link})
    install(TARGETS caffe2_detectron_ops_gpu DESTINATION lib)
    if (MSVC)
      install(FILES $<TARGET_PDB_FILE:caffe2_detectron_ops_gpu> DESTINATION lib OPTIONAL)
    endif()
  elseif(NOT IOS_PLATFORM)
    add_library(caffe2_detectron_ops SHARED ${Detectron_CPU_SRCS})
    target_link_libraries(caffe2_detectron_ops caffe2 ${OpenMP_link})
    install(TARGETS caffe2_detectron_ops DESTINATION lib)
    if (MSVC)
      install(FILES $<TARGET_PDB_FILE:caffe2_detectron_ops> DESTINATION lib OPTIONAL)
    endif()
  endif()
endif()
