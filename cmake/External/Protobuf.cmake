if (NOT __PROTOBUF_INCLUDED)
  set(__PROTOBUF_INCLUDED TRUE)

  # try the system-wide nccl first
  if(USE_SYSTEM_PROTOBUF)
    find_package(Protobuf)
  endif()
  if (PROTOBUF_FOUND)
      set(PROTOBUF_EXTERNAL FALSE)
  else()
    message(STATUS "Building Protobuf at ${CMAKE_SOURCE_DIR}/third_party/protobuf")
    # build directory
    set(protobuf_prefix ${CMAKE_SOURCE_DIR}/third_party/protobuf)

    # we build glog statically, but want to link it into the caffe shared library
    # this requires position-independent code
    if (UNIX)
      set(PROTOBUF_EXTRA_COMPILER_FLAGS "-fPIC")
    endif()

    set(PROTOBUF_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${PROTOBUF_EXTRA_COMPILER_FLAGS})
    set(PROTOBUF_C_FLAGS ${CMAKE_C_FLAGS} ${PROTOBUF_EXTRA_COMPILER_FLAGS})

    ExternalProject_Add(protobuf_external
      SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/protobuf
      BUILD_IN_SOURCE 1
      CONFIGURE_COMMAND "./autogen.sh" COMMAND ./configure --prefix=${CMAKE_SOURCE_DIR}/third_party/protobuf/_build
      BUILD_COMMAND make 
      INSTALL_COMMAND make install
      )

    set(PROTOBUF_FOUND TRUE)
    set(PROTOBUF_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/protobuf/_build/include)
    set(PROTOBUF_LIBRARIES ${CMAKE_SOURCE_DIR}/third_party/protobuf/_build/lib/libprotobuf.so)
    set(PROTOBUF_LIBRARY_DIRS ${CMAKE_SOURCE_DIR}/third_party/protobuf/_build/lib)
    set(PROTOBUF_PROTOC_EXECUTABLE ${CMAKE_SOURCE_DIR}/third_party/protobuf/_build/bin/protoc)
    set(PROTOBUF_EXTERNAL TRUE)

    list(APPEND external_project_dependencies protobuf_external)
  endif()

  include("${CMAKE_SOURCE_DIR}/cmake/ProtoBuf.cmake")

endif()

