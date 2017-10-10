# There is a linking issue that happens in some of the Windows builds.
# TODO(Yangqing): after the module redesing, enable this back.
if (NOT MSVC)
  add_subdirectory(contrib)
endif()
