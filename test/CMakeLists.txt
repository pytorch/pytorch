ADD_EXECUTABLE(scalar_test scalar_test.cpp)
TARGET_LINK_LIBRARIES(scalar_test ATen)

ADD_EXECUTABLE(basic basic.cpp)
TARGET_LINK_LIBRARIES(basic ATen)

add_executable(atest atest.cpp)
TARGET_LINK_LIBRARIES(atest ATen)

add_executable(broadcast_test broadcast_test.cpp)
TARGET_LINK_LIBRARIES(broadcast_test ATen)

add_executable(wrapdim_test wrapdim_test.cpp)
TARGET_LINK_LIBRARIES(wrapdim_test ATen)

add_executable(dlconvertor dlconvertor.cpp)
TARGET_LINK_LIBRARIES(dlconvertor ATen)
