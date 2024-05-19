project(modules CXX C)
add_subdirectory(detectron)
add_subdirectory(module_test)
add_subdirectory(observers)

# Finally, set Caffe2_MODULES to parent scope.
set(Caffe2_MODULES ${Caffe2_MODULES} PARENT_SCOPE)
