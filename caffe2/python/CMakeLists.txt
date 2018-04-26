# ---[ CPU files.
set(Caffe2_CPU_PYTHON_SRCS
    "/pybind_state.cc"
    "/pybind_state_dlpack.cc"
)

# ---[ MKL binding
if(USE_MKL AND USE_MKLML)
  set(Caffe2_CPU_PYTHON_SRCS
      ${Caffe2_CPU_PYTHON_SRCS}
      "/pybind_state_mkl.cc"
  )
endif()

# ---[ GPU files
set(Caffe2_GPU_PYTHON_SRCS
    ${Caffe2_CPU_PYTHON_SRCS}
    "/pybind_state_gpu.cc"
)

prepend(Caffe2_CPU_PYTHON_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_CPU_PYTHON_SRCS})
prepend(Caffe2_GPU_PYTHON_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_GPU_PYTHON_SRCS})

# --[ Some special handling for ideep binding as we need to build with "-mavx2"
if(USE_MKL AND USE_IDEEP AND CAFFE2_COMPILER_SUPPORTS_AVX2_EXTENSIONS)
  file(GLOB_RECURSE ideep_srcs *_ideep.cc)
  add_library(Caffe2_ideep_pybind OBJECT ${ideep_srcs})
  add_dependencies(Caffe2_ideep_pybind Caffe_PROTO Caffe2_PROTO)
  set_target_properties(Caffe2_ideep_pybind PROPERTIES COMPILE_FLAGS "-mavx2")
  set(Caffe2_CPU_PYTHON_SRCS
      ${Caffe2_CPU_PYTHON_SRCS}
      $<TARGET_OBJECTS:Caffe2_ideep_pybind>
  )
endif()

set(Caffe2_CPU_PYTHON_SRCS ${Caffe2_CPU_PYTHON_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_PYTHON_SRCS ${Caffe2_GPU_PYTHON_SRCS} PARENT_SCOPE)
