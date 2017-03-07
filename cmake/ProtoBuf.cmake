# Finds Google Protocol Buffers library and compilers and extends
# the standard cmake script with version and python generation support
function(custom_protobuf_find)
  option(protobuf_BUILD_SHARED_LIBS "" OFF)
  option(protobuf_BUILD_TESTS "" OFF)
  option(protobuf_BUILD_EXAMPLES "" OFF)
  # MSVC protobuf built with static library explicitly uses /MT and /MTd which
  # makes things a bit tricky, so we set it off.
  option(protobuf_MSVC_STATIC_RUNTIME "" OFF)
  if (APPLE)
    # Protobuf generated files triggers a deprecated atomic operation warning
    # so we turn it off here.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations" PARENT_SCOPE)
  endif()
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/protobuf/cmake)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/protobuf/src)
  list(APPEND Caffe2_DEPENDENCY_LIBS libprotobuf)
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

if (ANDROID OR IOS OR WIN32)
  custom_protobuf_find()
else()
  find_package( Protobuf )
  if ( NOT (Protobuf_FOUND OR PROTOBUF_FOUND) )
    custom_protobuf_find()
  else()
    # Adding PROTOBUF_LIBRARY for legacy support.
    list(APPEND Caffe2_DEPENDENCY_LIBS ${PROTOBUF_LIBRARIES} ${PROTOBUF_LIBRARY})
    include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIR})
  endif()
endif()

if (NOT (Protobuf_FOUND OR PROTOBUF_FOUND) )
  message(FATAL_ERROR "Could not find Protobuf or compile local version.")
endif()

# place where to generate protobuf sources
set(proto_gen_folder "${PROJECT_BINARY_DIR}/include/caffe/proto")
include_directories("${PROJECT_BINARY_DIR}/include")

set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)

################################################################################################
# Modification of standard 'protobuf_generate_cpp()' with output dir parameter and python support
# Usage:
#   caffe_protobuf_generate_cpp_py(<output_dir> <srcs_var> <hdrs_var> <python_var> <proto_files>)
function(caffe_protobuf_generate_cpp_py output_dir srcs_var hdrs_var python_var)
  if(NOT ARGN)
    message(SEND_ERROR "Error: caffe_protobuf_generate_cpp_py() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(fil ${ARGN})
      get_filename_component(abs_fil ${fil} ABSOLUTE)
      get_filename_component(abs_path ${abs_fil} PATH)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  else()
    set(_protoc_include -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
    foreach(dir ${PROTOBUF_IMPORT_DIRS})
      get_filename_component(abs_path ${dir} ABSOLUTE)
      list(FIND _protoc_include ${abs_path} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protoc_include -I ${abs_path})
      endif()
    endforeach()
  endif()

  set(${srcs_var})
  set(${hdrs_var})
  set(${python_var})
  foreach(fil ${ARGN})
    get_filename_component(abs_fil ${fil} ABSOLUTE)
    get_filename_component(fil_we ${fil} NAME_WE)

    list(APPEND ${srcs_var} "${output_dir}/${fil_we}.pb.cc")
    list(APPEND ${hdrs_var} "${output_dir}/${fil_we}.pb.h")
    list(APPEND ${python_var} "${output_dir}/${fil_we}_pb2.py")

    add_custom_command(
      OUTPUT "${output_dir}/${fil_we}.pb.cc"
             "${output_dir}/${fil_we}.pb.h"
             "${output_dir}/${fil_we}_pb2.py"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${output_dir}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out    ${output_dir} ${_protoc_include} ${abs_fil}
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --python_out ${output_dir} ${_protoc_include} ${abs_fil}
      DEPENDS ${abs_fil}
      COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM )
  endforeach()

  set_source_files_properties(${${srcs_var}} ${${hdrs_var}} ${${python_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
  set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
  set(${python_var} ${${python_var}} PARENT_SCOPE)
endfunction()
