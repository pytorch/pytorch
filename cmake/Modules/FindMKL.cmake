# Find the MKL libraries
#
# Options:
#
#   MKL_USE_IDEEP                     : use IDEEP interface
#   MKL_USE_MKLML                     : use MKLML interface
#   MKLML_USE_SINGLE_DYNAMIC_LIBRARY  : use single dynamic library interface
#   MKLML_USE_STATIC_LIBS             : use static libraries
#   MKLML_MULTI_THREADED              : use multi-threading
#
# This module defines the following variables:
#
#   MKL_FOUND            : True mkl is found
#   MKL_INCLUDE_DIR      : include directory
#   MKL_LIBRARIES        : the libraries to link against.

# ---[ Options
include(CMakeDependentOption)

if(MKL_USE_IDEEP)
  set(IDEEP_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep")
  set(MKLDNN_ROOT "${IDEEP_ROOT}/mkl-dnn")
  set(__ideep_looked_for IDEEP_ROOT)

  find_path(IDEEP_INCLUDE_DIR ideep.hpp PATHS ${IDEEP_ROOT} PATH_SUFFIXES include)
  find_path(MKLDNN_INCLUDE_DIR mkldnn.hpp mkldnn.h PATHS ${MKLDNN_ROOT} PATH_SUFFIXES include)
  if (NOT MKLDNN_INCLUDE_DIR)
    execute_process(COMMAND git submodule update --init mkl-dnn WORKING_DIRECTORY ${IDEEP_ROOT})
    find_path(MKLDNN_INCLUDE_DIR mkldnn.hpp mkldnn.h PATHS ${MKLDNN_ROOT} PATH_SUFFIXES include)
  endif()

  if (MKLDNN_INCLUDE_DIR)
    # to avoid adding conflicting submodels
    set(ORIG_WITH_TEST ${WITH_TEST})
    set(WITH_TEST OFF)
    add_subdirectory(${IDEEP_ROOT})
    set(WITH_TEST ${ORIG_WITH_TEST})

    file(GLOB_RECURSE MKLML_INNER_INCLUDE_DIR ${MKLDNN_ROOT}/external/*/mkl_vsl.h)
    if(MKLML_INNER_INCLUDE_DIR)
      # if user has multiple version under external/ then guess last
      # one alphabetically is "latest" and warn
      list(LENGTH MKLML_INNER_INCLUDE_DIR MKLINCLEN)
      if(MKLINCLEN GREATER 1)
        list(SORT MKLML_INNER_INCLUDE_DIR)
        list(REVERSE MKLML_INNER_INCLUDE_DIR)
        list(GET MKLML_INNER_INCLUDE_DIR 0 MKLINCLST)
        set(MKLML_INNER_INCLUDE_DIR "${MKLINCLST}")
      endif()
      get_filename_component(MKLML_INNER_INCLUDE_DIR ${MKLML_INNER_INCLUDE_DIR} DIRECTORY)
      list(APPEND IDEEP_INCLUDE_DIR ${MKLDNN_INCLUDE_DIR} ${MKLML_INNER_INCLUDE_DIR})
      list(APPEND __ideep_looked_for IDEEP_INCLUDE_DIR)

      if(APPLE)
        set(__mklml_inner_libs mklml iomp5)
      else()
        set(__mklml_inner_libs mklml_intel iomp5)
      endif()

      set(IDEEP_LIBRARIES "")
      foreach (__mklml_inner_lib ${__mklml_inner_libs})
        string(TOUPPER ${__mklml_inner_lib} __mklml_inner_lib_upper)
        find_library(${__mklml_inner_lib_upper}_LIBRARY
              NAMES ${__mklml_inner_lib}
              PATHS  "${MKLML_INNER_INCLUDE_DIR}/../lib"
              DOC "The path to Intel(R) MKLML ${__mklml_inner_lib} library")
        mark_as_advanced(${__mklml_inner_lib_upper}_LIBRARY)
        list(APPEND IDEEP_LIBRARIES ${${__mklml_inner_lib_upper}_LIBRARY})
        list(APPEND __ideep_looked_for ${__mklml_inner_lib_upper}_LIBRARY)
      endforeach()

      include(FindPackageHandleStandardArgs)
      find_package_handle_standard_args(IDEEP DEFAULT_MSG ${__ideep_looked_for})

      if(IDEEP_FOUND)
        set(MKLDNN_LIB "${CMAKE_SHARED_LIBRARY_PREFIX}mkldnn${CMAKE_SHARED_LIBRARY_SUFFIX}")
        list(APPEND IDEEP_LIBRARIES "${PROJECT_BINARY_DIR}/lib/${MKLDNN_LIB}")
        set(CAFFE2_USE_IDEEP 1)
        # Do NOT use MPI if IDEEP is enabled
        set(USE_MPI OFF)
        message(STATUS "Found IDEEP (include: ${IDEEP_INCLUDE_DIR}, lib: ${IDEEP_LIBRARIES})")
      endif()

      caffe_clear_vars(__ideep_looked_for __mklml_inner_libs)
    endif()
  endif()

  if(NOT IDEEP_FOUND)
    message(FATAL_ERROR "Did not find IDEEP files!")
  endif()
endif()

if(MKL_USE_MKLML)

  # ---[ Options
  option(MKLML_USE_SINGLE_DYNAMIC_LIBRARY "Use single dynamic library interface" ON)
  cmake_dependent_option(
    MKLML_USE_STATIC_LIBS "Use static libraries" OFF
      "NOT MKLML_USE_SINGLE_DYNAMIC_LIBRARY" OFF)
  cmake_dependent_option(
    MKLML_MULTI_THREADED  "Use multi-threading" ON
      "NOT MKLML_USE_SINGLE_DYNAMIC_LIBRARY" OFF)

  # ---[ Root folders
  if(MSVC)
    set(INTEL_ROOT_DEFAULT "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows")
  else()
    set(INTEL_ROOT_DEFAULT "/opt/intel")
  endif()
  set(INTEL_ROOT ${INTEL_ROOT_DEFAULT} CACHE PATH "Folder contains intel libs")
  find_path(MKLML_ROOT include/mkl.h PATHS $ENV{MKLMLROOT} ${INTEL_ROOT}/mkl
    DOC "Folder contains MKLML")

  # ---[ Find include dir
  find_path(MKLML_INCLUDE_DIR mkl.h PATHS ${MKLML_ROOT} PATH_SUFFIXES include)
  set(__looked_for MKLML_INCLUDE_DIR)

  # ---[ Find libraries
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(__path_suffixes lib lib/ia32)
  else()
    set(__path_suffixes lib lib/intel64)
  endif()

  set(__mklml_libs "")
  if(MKLML_USE_SINGLE_DYNAMIC_LIBRARY)
    list(APPEND __mklml_libs rt)
  else()
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
      if(WIN32)
        list(APPEND __mklml_libs intel_c)
      else()
        list(APPEND __mklml_libs intel gf)
      endif()
    else()
      list(APPEND __mklml_libs intel_lp64 gf_lp64)
    endif()

    if(MKLML_MULTI_THREADED)
      list(APPEND __mklml_libs intel_thread)
    else()
       list(APPEND __mklml_libs sequential)
    endif()

    list(APPEND __mklml_libs core cdft_core)
  endif()

  foreach (__lib ${__mklml_libs})
    set(__mklml_lib "mkl_${__lib}")
    string(TOUPPER ${__mklml_lib} __mklml_lib_upper)

    if(MKLML_USE_STATIC_LIBS)
      set(__mklml_lib "lib${__mklml_lib}.a")
    endif()

    find_library(${__mklml_lib_upper}_LIBRARY
          NAMES ${__mklml_lib}
          PATHS ${MKLML_ROOT} "${MKLML_INCLUDE_DIR}/.."
          PATH_SUFFIXES ${__path_suffixes}
          DOC "The path to Intel(R) MKLML ${__mklml_lib} library")
    mark_as_advanced(${__mklml_lib_upper}_LIBRARY)

    list(APPEND __looked_for ${__mklml_lib_upper}_LIBRARY)
    list(APPEND MKLML_LIBRARIES ${${__mklml_lib_upper}_LIBRARY})
  endforeach()

  if(NOT MKLML_USE_SINGLE_DYNAMIC_LIBRARY)
    if (MKLML_USE_STATIC_LIBS)
      set(__iomp5_libs iomp5 libiomp5mt.lib)
    else()
      set(__iomp5_libs iomp5 libiomp5md.lib)
    endif()

    if(WIN32)
      find_path(INTEL_INCLUDE_DIR omp.h PATHS ${INTEL_ROOT} PATH_SUFFIXES include)
      list(APPEND __looked_for INTEL_INCLUDE_DIR)
    endif()

    find_library(MKLML_RTL_LIBRARY ${__iomp5_libs}
      PATHS ${INTEL_RTL_ROOT} ${INTEL_ROOT}/compiler ${MKLML_ROOT}/.. ${MKLML_ROOT}/../compiler
       PATH_SUFFIXES ${__path_suffixes}
       DOC "Path to OpenMP runtime library")

     list(APPEND __looked_for MKLML_RTL_LIBRARY)
     list(APPEND MKLML_LIBRARIES ${MKLML_RTL_LIBRARY})
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(MKLML DEFAULT_MSG ${__looked_for})

  if(MKLML_FOUND)
    set(CAFFE2_USE_MKL 1)
    message(STATUS "Found MKLML (include: ${MKLML_INCLUDE_DIR}, lib: ${MKLML_LIBRARIES})")
  endif()

  caffe_clear_vars(__looked_for __mklml_libs __path_suffixes __iomp5_libs)

endif()

if(IDEEP_FOUND OR MKLML_FOUND)
  set(USE_MKL ON)
  set(MKL_FOUND True)
  list(APPEND MKL_INCLUDE_DIR ${IDEEP_INCLUDE_DIR} ${MKLML_INCLUDE_DIR})
  list(APPEND MKL_LIBRARIES ${IDEEP_LIBRARIES} ${MKLML_LIBRARIES})
else()
  set(USE_MKL OFF)
  set(MKL_USE_IDEEP OFF)
  set(MKL_USE_MKLML OFF)
endif()
