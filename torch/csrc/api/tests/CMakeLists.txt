ADD_EXECUTABLE(autograd_test
  test.cpp
  container_t.cpp
  misc_t.cpp
  rnn_t.cpp
  integration_t.cpp
  optim_t.cpp
  serialization_t.cpp
)

TARGET_COMPILE_OPTIONS(autograd_test
  PRIVATE
    -Dlest_FEATURE_AUTO_REGISTER=1
)

ADD_CUSTOM_COMMAND(
  TARGET autograd_test PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/mnist" "${CMAKE_CURRENT_BINARY_DIR}/mnist"
)

TARGET_LINK_LIBRARIES(autograd_test autogradpp)
