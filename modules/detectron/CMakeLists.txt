file(GLOB Detectron_CPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
file(GLOB Detectron_GPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

if (BUILD_CAFFE2_OPS)
  #cmake only check for separate OpenMP library on AppleClang 7+
  #https://github.com/Kitware/CMake/blob/42212f7539040139ecec092547b7d58ef12a4d72/Modules/FindOpenMP.cmake#L252
  if (WITH_OPENMP AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "7.0" OR
        CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "7.0")
      Set(OpenMP_link ${OpenMP_libomp_LIBRARY})
    endif()
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
  elseif(NOT IOS_PLATFORM)
    add_library(caffe2_detectron_ops SHARED ${Detectron_CPU_SRCS})
    target_link_libraries(caffe2_detectron_ops caffe2 ${OpenMP_link})
    install(TARGETS caffe2_detectron_ops DESTINATION lib)
  endif()
endif()
