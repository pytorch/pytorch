# ---[ Add modules
# TODO(orionr): Enable Detectron ops for Windows DLL when we
# can figure out how to get it to build
if (NOT (MSVC AND BUILD_SHARED_LIBS))
  add_subdirectory(detectron)
endif()
add_subdirectory(module_test)
add_subdirectory(observers)
add_subdirectory(rocksdb)

# Finally, set Caffe2_MODULES to parent scope.
set(Caffe2_MODULES ${Caffe2_MODULES} PARENT_SCOPE)
