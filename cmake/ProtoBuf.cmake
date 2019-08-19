# Finds Google Protocol Buffers library and compilers and extends
# the standard cmake script with version and python generation support
macro(custom_protobuf_find)
  message(STATUS "Use custom protobuf build.")
  option(protobuf_BUILD_TESTS "" OFF)
  option(protobuf_BUILD_EXAMPLES "" OFF)
  option(protobuf_WITH_ZLIB "" OFF)
  if (APPLE)
    # Protobuf generated files triggers a deprecated atomic operation warning
    # so we turn it off here.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
  endif()
  if (${CAFFE2_LINK_LOCAL_PROTOBUF})
    # If we are going to link protobuf locally, we will need to turn off
    # shared libs build for protobuf.
    option(protobuf_BUILD_SHARED_LIBS "" OFF)
  else()
    # If we are building Caffe2 as shared libs, we will also build protobuf as
    # shared libs.
    option(protobuf_BUILD_SHARED_LIBS "" ${BUILD_SHARED_LIBS})
  endif()
  # We will make sure that protobuf and caffe2 uses the same msvc runtime.
  option(protobuf_MSVC_STATIC_RUNTIME "" ${CAFFE2_USE_MSVC_STATIC_RUNTIME})

  if (${CAFFE2_LINK_LOCAL_PROTOBUF})
    set(__caffe2_CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ${CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS})
    set(__caffe2_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    set(BUILD_SHARED_LIBS OFF)
    if (${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    endif()
    if (${COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
    endif()
  endif()

  set(__caffe2_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  if (MSVC)
    if(MSVC_Z7_OVERRIDE)
      foreach(flag_var
          CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
          CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
          CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/Z[iI]")
          string(REGEX REPLACE "/Z[iI]" "/Z7" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/Z[iI]")
      endforeach(flag_var)
    endif(MSVC_Z7_OVERRIDE)
  endif(MSVC)

  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/protobuf/cmake)

  set(CMAKE_POSITION_INDEPENDENT_CODE ${__caffe2_CMAKE_POSITION_INDEPENDENT_CODE})

  if (${CAFFE2_LINK_LOCAL_PROTOBUF})
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ${__caffe2_CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS})
    set(BUILD_SHARED_LIBS ON)
    set(CMAKE_CXX_FLAGS ${__caffe2_CMAKE_CXX_FLAGS})
  endif()

  # Protobuf "namespaced" target is only added post protobuf 3.5.1. As a
  # result, for older versions, we will manually add alias.
  if (NOT TARGET protobuf::libprotobuf)
    add_library(protobuf::libprotobuf ALIAS libprotobuf)
    add_library(protobuf::libprotobuf-lite ALIAS libprotobuf-lite)
    # There is link error when cross compiling protoc on mobile:
    # https://github.com/protocolbuffers/protobuf/issues/2719
    # And protoc is very unlikely needed for mobile builds.
    if (NOT (ANDROID OR IOS))
      add_executable(protobuf::protoc ALIAS protoc)
    endif()
  endif()
endmacro()

# Main entry for protobuf. If we are building on Android, iOS or we have hard
# coded BUILD_CUSTOM_PROTOBUF, we will hard code the use of custom protobuf
# in the submodule.
if (ANDROID OR IOS)
  if (NOT ${BUILD_CUSTOM_PROTOBUF})
    message(WARNING
        "For Android and iOS cross compilation, I am automatically using "
        "custom protobuf under third party. Note that this behavior may "
        "change in the future, and you will need to specify "
        "-DBUILD_CUSTOM_PROTOBUF=ON explicitly.")
  endif()
  # There is link error when cross compiling protoc on mobile:
  # https://github.com/protocolbuffers/protobuf/issues/2719
  # And protoc is very unlikely needed for mobile builds.
  set(__caffe2_protobuf_BUILD_PROTOC_BINARIES ${protobuf_BUILD_PROTOC_BINARIES})
  set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "" FORCE)
  custom_protobuf_find()
  set(protobuf_BUILD_PROTOC_BINARIES ${__caffe2_protobuf_BUILD_PROTOC_BINARIES} CACHE BOOL "" FORCE)
elseif (BUILD_CUSTOM_PROTOBUF)
  message(STATUS "Building using own protobuf under third_party per request.")
  custom_protobuf_find()
else()
  include(cmake/public/protobuf.cmake)
endif()

if ((NOT TARGET protobuf::libprotobuf) AND (NOT TARGET protobuf::libprotobuf-lite))
  message(WARNING
      "Protobuf cannot be found. Caffe2 will automatically switch to use "
      "own protobuf under third_party. Note that this behavior may change in "
      "the future, and you will need to specify -DBUILD_CUSTOM_PROTOBUF=ON "
      "explicitly.")
  custom_protobuf_find()

  # TODO(jiayq): enable this in the future, when Jenkins Mac support is
  # properly set up with protobuf installs.

  # message(FATAL_ERROR
  #     "Protobuf cannot be found. Caffe2 will have to build with libprotobuf. "
  #     "Please set the proper paths so that I can find protobuf correctly.")
endif()

get_target_property(__tmp protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Caffe2 protobuf include directory: " ${__tmp})
include_directories(BEFORE SYSTEM ${__tmp})

# If Protobuf_VERSION is known (true in most cases, false if we are building
# local protobuf), then we will add a protobuf version check in
# Caffe2Config.cmake.in.
if (DEFINED ${Protobuf_VERSION})
  set(CAFFE2_KNOWN_PROTOBUF_VERSION TRUE)
else()
  set(CAFFE2_KNOWN_PROTOBUF_VERSION FALSE)
  set(Protobuf_VERSION "Protobuf_VERSION_NOTFOUND")
endif()


# Figure out which protoc to use.
# If CAFFE2_CUSTOM_PROTOC_EXECUTABLE is set, we assume the user knows
# what they're doing and we blindly use the specified protoc. This
# is typically the case when cross-compiling where protoc must be
# compiled for the host architecture and libprotobuf must be
# compiled for the target architecture.
# If CAFFE2_CUSTOM_PROTOC_EXECUTABLE is NOT set, we use the protoc
# target that is built as part of including the protobuf project.
if(EXISTS "${CAFFE2_CUSTOM_PROTOC_EXECUTABLE}")
  set(CAFFE2_PROTOC_EXECUTABLE ${CAFFE2_CUSTOM_PROTOC_EXECUTABLE})
else()
  set(CAFFE2_PROTOC_EXECUTABLE protobuf::protoc)
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

    # Add CAFFE2_API prefix to protobuf classes and methods in all cases
    set(DLLEXPORT_STR "dllexport_decl=CAFFE2_API:")

    # Note: the following depends on PROTOBUF_PROTOC_EXECUTABLE. This
    # is done to make sure protoc is built before attempting to
    # generate sources if we're using protoc from the third_party
    # directory and are building it as part of the Caffe2 build. If
    # points to an existing path, it is a no-op.

    if (${CAFFE2_LINK_LOCAL_PROTOBUF})
      # We need to rewrite the pb.h files to route GetEmptyStringAlreadyInited
      # through our wrapper in proto_utils so the memory location test
      # is correct.
      add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h"
               "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}_pb2.py"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
        COMMAND ${CAFFE2_PROTOC_EXECUTABLE} -I${PROJECT_SOURCE_DIR} --cpp_out=${DLLEXPORT_STR}${PROJECT_BINARY_DIR} ${abs_fil}
        COMMAND ${CAFFE2_PROTOC_EXECUTABLE} -I${PROJECT_SOURCE_DIR} --python_out "${PROJECT_BINARY_DIR}" ${abs_fil}

        # If we remove all reference to these pb.h files from external
        # libraries and binaries this rewrite can be removed.
        COMMAND ${CMAKE_COMMAND} -DFILENAME=${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h -DNAMESPACES=caffe\;caffe2\;onnx\;torch -P ${PROJECT_SOURCE_DIR}/cmake/ProtoBufPatch.cmake

        DEPENDS ${CAFFE2_PROTOC_EXECUTABLE} ${abs_fil}
        COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM )
    else()
      add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}.pb.h"
               "${CMAKE_CURRENT_BINARY_DIR}/${fil_we}_pb2.py"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
        COMMAND ${CAFFE2_PROTOC_EXECUTABLE} -I${PROJECT_SOURCE_DIR} --cpp_out=${DLLEXPORT_STR}${PROJECT_BINARY_DIR} ${abs_fil}
        COMMAND ${CAFFE2_PROTOC_EXECUTABLE} -I${PROJECT_SOURCE_DIR} --python_out "${PROJECT_BINARY_DIR}" ${abs_fil}
        DEPENDS ${CAFFE2_PROTOC_EXECUTABLE} ${abs_fil}
        COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM )
    endif()
  endforeach()

  set_source_files_properties(${${srcs_var}} ${${hdrs_var}} ${${python_var}} PROPERTIES GENERATED TRUE)
  set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
  set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
  set(${python_var} ${${python_var}} PARENT_SCOPE)
endfunction()
