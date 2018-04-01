# ---[ Add modules
add_subdirectory(detectron)
add_subdirectory(module_test)
add_subdirectory(observers)
add_subdirectory(rocksdb)

# Finally, set Caffe2_MODULES to parent scope.
set(Caffe2_MODULES ${Caffe2_MODULES} PARENT_SCOPE)