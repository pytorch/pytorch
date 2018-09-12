# ---[ CPU files.
set(Caffe2_CPU_PYTHON_SRCS
    "/pybind_state.cc"
    "/pybind_state_dlpack.cc"
    "/pybind_state_nomni.cc"
    "/pybind_state_registry.cc"
)

# ---[ MKL binding
if(USE_MKL AND USE_MKLML)
  set(Caffe2_CPU_PYTHON_SRCS
      ${Caffe2_CPU_PYTHON_SRCS}
      "/pybind_state_mkl.cc"
  )
endif()

# ---[ IDEEP binding
if(USE_MKL AND USE_IDEEP)
  set(Caffe2_CPU_PYTHON_SRCS
      ${Caffe2_CPU_PYTHON_SRCS}
      "/pybind_state_ideep.cc"
  )
endif()

# ---[ GPU files
set(Caffe2_GPU_PYTHON_SRCS
    ${Caffe2_CPU_PYTHON_SRCS}
    "/pybind_state_gpu.cc"
)

# ---[ HIP files
set(Caffe2_HIP_PYTHON_SRCS
    ${Caffe2_CPU_PYTHON_SRCS}
    "/pybind_state_hip.cc"
)

prepend(Caffe2_CPU_PYTHON_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_CPU_PYTHON_SRCS})
prepend(Caffe2_GPU_PYTHON_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_GPU_PYTHON_SRCS})
prepend(Caffe2_HIP_PYTHON_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_HIP_PYTHON_SRCS})

set(Caffe2_CPU_PYTHON_SRCS ${Caffe2_CPU_PYTHON_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_PYTHON_SRCS ${Caffe2_GPU_PYTHON_SRCS} PARENT_SCOPE)
set(Caffe2_HIP_PYTHON_SRCS ${Caffe2_HIP_PYTHON_SRCS} PARENT_SCOPE)

