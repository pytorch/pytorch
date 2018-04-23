# ---[ CPU files.
set(Caffe2_CPU_PYTHON_SRCS
    "/pybind_state.cc"
    "/pybind_state_dlpack.cc"
)

if(USE_MKL)
  if(MKL_USE_IDEEP)
  set(Caffe2_CPU_PYTHON_SRCS
      ${Caffe2_CPU_PYTHON_SRCS}
      "/pybind_state_ideep.cc"
  )
  endif()

  if(MKL_USE_MKLML)
  set(Caffe2_CPU_PYTHON_SRCS
      ${Caffe2_CPU_PYTHON_SRCS}
      "/pybind_state_mkl.cc"
  )
  endif()
endif()

# ---[ GPU files
set(Caffe2_GPU_PYTHON_SRCS
    ${Caffe2_CPU_PYTHON_SRCS}
    "/pybind_state_gpu.cc"
)

prepend(Caffe2_CPU_PYTHON_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_CPU_PYTHON_SRCS})
prepend(Caffe2_GPU_PYTHON_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_GPU_PYTHON_SRCS})

set(Caffe2_CPU_PYTHON_SRCS ${Caffe2_CPU_PYTHON_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_PYTHON_SRCS ${Caffe2_GPU_PYTHON_SRCS} PARENT_SCOPE)
