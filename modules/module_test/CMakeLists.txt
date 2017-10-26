add_library(
    caffe2_module_test_dynamic SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/module_test_dynamic.cc)

target_link_libraries(caffe2_module_test_dynamic caffe2)
install(TARGETS caffe2_module_test_dynamic DESTINATION lib)
