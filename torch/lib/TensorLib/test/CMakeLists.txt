ADD_EXECUTABLE(scalar_test scalar_test.cpp)
TARGET_LINK_LIBRARIES(scalar_test TensorLib)
TARGET_LINK_LIBRARIES(scalar_test ${CUDA_LIBRARIES})

ADD_EXECUTABLE(basic basic.cpp)
TARGET_LINK_LIBRARIES(scalar_test TensorLib)
TARGET_LINK_LIBRARIES(scalar_test ${CUDA_LIBRARIES})
