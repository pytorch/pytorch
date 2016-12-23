set(Caffe2_PYTHON_CPU_SRC
    "/pybind_state.cc"
)
set(Caffe2_PYTHON_GPU_SRC
    "/pybind_state.cc"
    "/pybind_state_gpu.cc"
)

prepend(Caffe2_PYTHON_CPU_SRC ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_PYTHON_CPU_SRC})
prepend(Caffe2_PYTHON_GPU_SRC ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_PYTHON_GPU_SRC})

add_library(caffe2_pybind11_state SHARED ${Caffe2_PYTHON_CPU_SRC})
set_target_properties(caffe2_pybind11_state PROPERTIES PREFIX "")
target_link_libraries(caffe2_pybind11_state ${Caffe2_LINK})
install(TARGETS caffe2_pybind11_state DESTINATION ${CMAKE_INSTALL_PREFIX}/caffe2/python)

add_library(caffe2_pybind11_state_gpu SHARED ${Caffe2_PYTHON_GPU_SRC})
set_target_properties(caffe2_pybind11_state_gpu PROPERTIES PREFIX "")
target_link_libraries(caffe2_pybind11_state_gpu ${Caffe2_LINK})
install(TARGETS caffe2_pybind11_state_gpu DESTINATION ${CMAKE_INSTALL_PREFIX}/caffe2/python)
