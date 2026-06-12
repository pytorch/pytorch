# Build separate libraries the define custom classes/operators used from our Python tests.
# These are intended to be used with torch.ops.load_library() in our Python test suite.
add_library(aoti_custom_ops SHARED custom_ops.cpp)
target_link_libraries(aoti_custom_ops torch)

include(GNUInstallDirs)

if(INSTALL_TEST)
  install(TARGETS aoti_custom_ops DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
