# Finds Google Protocol Buffers library and compilers and extends
# the standard cmake script with version and python generation support
function(custom_protobuf_find)
  message(STATUS "Use custom protobuf build.")
  # For a custom protobuf build, we will always use static protobuf.
  option(protobuf_BUILD_SHARED_LIBS "" OFF)
  option(protobuf_BUILD_TESTS "" OFF)
  option(protobuf_BUILD_EXAMPLES "" OFF)
  # MSVC protobuf built with static library explicitly uses /MT and /MTd which
  # makes things a bit tricky, so we set it off.
  #option(protobuf_MSVC_STATIC_RUNTIME "" OFF)
  if (APPLE)
    # Protobuf generated files triggers a deprecated atomic operation warning
    # so we turn it off here.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations" PARENT_SCOPE)
  endif()
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/protobuf/cmake)
  set(PROTOBUF_LIBRARIES libprotobuf PARENT_SCOPE)
  set(PROTOBUF_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/protobuf/src PARENT_SCOPE)
  set(Caffe2_DEPENDENCY_LIBS ${Caffe2_DEPENDENCY_LIBS} PARENT_SCOPE)
  if(NOT EXISTS ${PROTOBUF_PROTOC_EXECUTABLE})
    message(FATAL_ERROR
            "To build protobufs locally (required for Android/iOS/Windows), "
            "you will need to manually specify a PROTOBUF_PROTOC_EXECUTABLE. "
            "See scripts/build_host_protoc.{sh,bat} for more details.")
  else()
    message(STATUS "Using protobuf compiler ${PROTOBUF_PROTOC_EXECUTABLE}.")
  endif()
  set(Protobuf_FOUND TRUE PARENT_SCOPE)
endfunction()

if (WIN32)
  find_package( Protobuf NO_MODULE)
  if ( NOT (Protobuf_FOUND OR PROTOBUF_FOUND) )
    custom_protobuf_find()
  endif()
elseif (ANDROID OR IOS)
  custom_protobuf_find()
  if (IOS_PLATFORM STREQUAL "WATCHOS")
    # Unfortunately, WatchOS does not support building libprotoc and protoc,
    # so we will need to exclude it. The problem of using EXCLUDE_FROM_ALL is
    # that one is not going to be able to run cmake install. A proper solution
    # has to be implemented by protobuf since we derive our cmake files from
    # there.
    set_target_properties(
        libprotoc protoc PROPERTIES
        EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
  endif()
else()
  find_package( Protobuf )
endif()

# If Protobuf is not found, do custom protobuf find.
if ( NOT (Protobuf_FOUND OR PROTOBUF_FOUND) )
  custom_protobuf_find()
endif()

caffe2_include_directories(${PROTOBUF_INCLUDE_DIR})
# Adding PROTOBUF_LIBRARY for legacy support.
list(APPEND Caffe2_DEPENDENCY_LIBS ${PROTOBUF_LIBRARIES} ${PROTOBUF_LIBRARY})

if (NOT (Protobuf_FOUND OR PROTOBUF_FOUND) )
  message(FATAL_ERROR "Could not find Protobuf or compile local version.")
endif()

################################################################################################
# Modification of standard 'protobuf_generate_cpp()' with output dir parameter and python support
# Usage:
#   caffe2_protobuf_generate_cpp_py(<srcs_var> <hdrs_var> <python_var> <proto_files>)
function(caffe2_protobuf_generate_cpp_py srcs_var hdrs_var python_var)
  if(NOT ARGN)
    message(SEND_ERROR "Error: caffe_protobuf_generate_cpp_py() called without any proto files")
    return()
  endif()

  set(${srcs_var})
  set(${hdrs_var})
  set(${python_var})
  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)

    list(APPEND ${srcs_var} "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.cc")
    list(APPEND ${hdrs_var} "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h")
    list(APPEND ${python_var} "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}_pb2.py")

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h"
             "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}_pb2.py"
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} -I${PROJECT_SOURCE_DIR} --cpp_out    "${PROJECT_BINARY_DIR}" ${abs_fil}
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} -I${PROJECT_SOURCE_DIR} --python_out "${PROJECT_BINARY_DIR}" ${abs_fil}
      DEPENDS ${abs_fil}
      COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM )
  endforeach()

  set_source_files_properties(${${srcs_var}} ${${hdrs_var}} ${${python_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
  set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
  set(${python_var} ${${python_var}} PARENT_SCOPE)
endfunction()
