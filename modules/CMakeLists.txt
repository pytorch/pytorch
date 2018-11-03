# ---[ Add modules
# With build_windows_shared.bat file detectron ops build and work on MSVC 14 toolset
option(BUILD_DETECTRON_WITH_MSVC "This option is set to ON with build_windows_shared bat file" OFF)
# TODO(orionr): Enable Detectron ops for Windows DLL when we
# can figure out how to get it to build
if ((NOT (MSVC AND BUILD_SHARED_LIBS)) OR BUILD_DETECTRON_WITH_MSVC)
  add_subdirectory(detectron)
endif()
add_subdirectory(module_test)
add_subdirectory(observers)
add_subdirectory(rocksdb)

# Finally, set Caffe2_MODULES to parent scope.
set(Caffe2_MODULES ${Caffe2_MODULES} PARENT_SCOPE)
