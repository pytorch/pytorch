ADD_EXECUTABLE(scalar_test scalar_test.cpp)
TARGET_LINK_LIBRARIES(scalar_test ATen)

ADD_EXECUTABLE(basic basic.cpp)
TARGET_LINK_LIBRARIES(basic ATen)

add_executable(atest atest.cpp)
target_link_libraries(atest ATen)

add_executable(broadcast_test broadcast_test.cpp)
target_link_libraries(broadcast_test ATen)

add_executable(wrapdim_test wrapdim_test.cpp)
target_link_libraries(wrapdim_test ATen)
