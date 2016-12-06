set(Caffe2_PYTHON_CPU_SRC
    "/pybind_state.cc"
)
set(Caffe2_PYTHON_GPU_SRC
    "/pybind_state.cc"
    "/pybind_state_gpu.cc"
)

prepend(Caffe2_PYTHON_CPU_SRC ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_PYTHON_CPU_SRC})
prepend(Caffe2_PYTHON_GPU_SRC ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_PYTHON_GPU_SRC})

add_library(Caffe2_PYTHON_CPU SHARED ${Caffe2_PYTHON_CPU_SRC})
# target_link_libraries(Caffe2_PYTHON_CPU)

add_library(Caffe2_PYTHON_GPU_SHARED SHARED ${Caffe2_PYTHON_GPU_SRC})
