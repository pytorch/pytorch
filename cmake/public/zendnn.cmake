if(NOT ZENDNN_FOUND)

  # Set source directory for ZenDNN
  set(ZENDNNL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/ZenDNN)
  if(EXISTS "${ZENDNNL_SOURCE_DIR}")
    add_subdirectory(${ZENDNNL_SOURCE_DIR})

    # Create and configure interface target if it doesn't exist
    if(NOT TARGET caffe2::zendnn)
      add_library(caffe2::zendnn INTERFACE IMPORTED)
      add_dependencies(caffe2::zendnn zendnnl::zendnnl_archive)
      target_link_libraries(caffe2::zendnn INTERFACE zendnnl::zendnnl_archive)
    endif()

    # Mark as found
    set(ZENDNN_FOUND TRUE)
    message(STATUS "  Found ZenDNN: TRUE")

    # Log target properties for debugging
    if(CMAKE_DEBUG_OUTPUT)
      get_target_property(_include_dirs caffe2::zendnn INTERFACE_INCLUDE_DIRECTORIES)
      get_target_property(_libs caffe2::zendnn INTERFACE_LINK_LIBRARIES)
      message(STATUS "  ZenDNN include dirs: ${_include_dirs}")
      message(STATUS "  ZenDNN libraries: ${_libs}")
    endif()
  else()
    set(ZENDNN_FOUND FALSE)
    message(WARNING "ZenDNN directory not found at ${ZENDNNL_SOURCE_DIR}.")
  endif()
endif()
