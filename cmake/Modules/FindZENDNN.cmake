include_guard(GLOBAL)
include(ExternalProject)

# declare a zendnnl dependency
macro(zendnnl_add_dependency )
  set(options INCLUDE_ONLY)
  set(oneValueArgs NAME PATH LIB_SUFFIX INCLUDE_SUFFIX ARCHIVE_FILE ALIAS)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(_zad "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  string(TOUPPER ${_zad_NAME} _ZAD_UNAME)

  if(DEFINED _zad_INCLUDE_SUFFIX)
    set(ZENDNNL_${_ZAD_UNAME}_INC_DIR "${_zad_PATH}/${_zad_INCLUDE_SUFFIX}")
  else()
    set(ZENDNNL_${_ZAD_UNAME}_INC_DIR "${_zad_PATH}/include")
  endif()

  if(DEFINED _zad_LIB_SUFFIX)
    set(ZENDNNL_${_ZAD_UNAME}_LIB_DIR "${_zad_PATH}/${_zad_LIB_SUFFIX}")
  else()
    set(ZENDNNL_${_ZAD_UNAME}_LIB_DIR "${_zad_PATH}/lib")
  endif()

  if(NOT EXISTS ${ZENDNNL_${_ZAD_UNAME}_INC_DIR})
    file(MAKE_DIRECTORY ${ZENDNNL_${_ZAD_UNAME}_INC_DIR})
  endif()

  if(${_zad_INCLUDE_ONLY})
    add_library(zendnnl_${_zad_NAME}_deps INTERFACE IMPORTED GLOBAL)
    #add_dependencies(zendnnl_${_zad_NAME}_deps ${_zad_DEPENDS})

    set_target_properties(zendnnl_${_zad_NAME}_deps
      PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_${_ZAD_UNAME}_INC_DIR}")
  else()

    add_library(zendnnl_${_zad_NAME}_deps STATIC IMPORTED GLOBAL)
    #add_dependencies(zendnnl_${_zad_NAME}_deps ${_zad_DEPENDS})

    set_target_properties(zendnnl_${_zad_NAME}_deps
      PROPERTIES
      IMPORTED_LOCATION "${ZENDNNL_${_ZAD_UNAME}_LIB_DIR}/${_zad_ARCHIVE_FILE}"
      INCLUDE_DIRECTORIES "${ZENDNNL_${_ZAD_UNAME}_INC_DIR}"
      INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_${_ZAD_UNAME}_INC_DIR}")
  endif()

  add_library(${_zad_ALIAS} ALIAS zendnnl_${_zad_NAME}_deps)

  list(APPEND ZNL_BYPRODUCTS "${ZENDNNL_${_ZAD_UNAME}_LIB_DIR}/${_zad_ARCHIVE_FILE}")
endmacro()

macro(zendnnl_add_option )
  set(options EXECLUDE_FROM_COMMAND_LIST FORCE)
  set(oneValueArgs NAME VALUE TYPE CACHE_STRING COMMAND_LIST)
  set(multiValueArgs "")
  cmake_parse_arguments(_zao "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(${_zao_FORCE})
    set(${_zao_NAME} ${_zao_VALUE} CACHE ${_zao_TYPE} ${_zao_CACHE_STRING} FORCE)
  else()
    set(${_zao_NAME} ${_zao_VALUE} CACHE ${_zao_TYPE} ${_zao_CACHE_STRING})
  endif()

  if (NOT ${_zao_EXECLUDE_FROM_COMMAND_LIST})
    list(APPEND ${_zao_COMMAND_LIST} "-D${_zao_NAME}:${_zao_TYPE}=${_zao_VALUE}")
  endif()
endmacro()

message(AUTHOR_WARNING "(ZENDNNL) please ensure all zendnnl variables are set properly.")

if(NOT ZENDNN_FOUND)
  # find openmp
  find_package(OpenMP REQUIRED QUIET)

  # set zendnnl source dir, where zendnnl has been downloaded.
  zendnnl_add_option(NAME ZENDNNL_SOURCE_DIR
    VALUE ${PROJECT_SOURCE_DIR}/third_party/ZenDNN
    TYPE PATH
    CACHE_STRING "zendnnl_source_dir"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set zendnnl binary dir, if unsure set ${CMAKE_CURRENT_BINARY_DIR}/zendnnl.
  zendnnl_add_option(NAME ZENDNNL_BINARY_DIR
    VALUE ${ZENDNNL_SOURCE_DIR}/build
    TYPE PATH
    CACHE_STRING "zendnnl_binary_dir"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set zendnnl install dir, if unsure set ${CMAKE_INSTALL_PREFIX}/zendnnl.
  zendnnl_add_option(NAME ZENDNNL_INSTALL_PREFIX
    VALUE ${ZENDNNL_BINARY_DIR}/install
    TYPE PATH
    CACHE_STRING "zendnnl_install_dir"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  ## general zendnnl options
  # set ZenDNNL framework build, this should on ON to avoid standalone build.
  zendnnl_add_option(NAME ZENDNNL_FWK_BUILD
    VALUE ON
    TYPE BOOL
    CACHE_STRING "zendnnl framework build"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set zendnnl build option, default is Release.
  zendnnl_add_option(NAME ZENDNNL_BUILD_TYPE
    VALUE "Release"
    TYPE STRING
    CACHE_STRING "zendnnl build type"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set zendnnl log level.
  zendnnl_add_option(NAME ZENDNNL_MESSAGE_LOG_LEVEL
    VALUE "DEBUG"
    TYPE STRING
    CACHE_STRING "zendnnl message log level"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set zendnnl verbose makefile option.
  zendnnl_add_option(NAME ZENDNNL_VERBOSE_MAKEFILE
    VALUE ON
    TYPE BOOL
    CACHE_STRING "zendnnl verbose makefile"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  ## components options
  # set building zendnnl examples, default os OFF.
  zendnnl_add_option(NAME ZENDNNL_BUILD_EXAMPLES
    VALUE OFF
    TYPE BOOL
    CACHE_STRING "build zendnnl examples"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set building zendnnl gtests, default os OFF.
  zendnnl_add_option(NAME ZENDNNL_BUILD_GTEST
    VALUE OFF
    TYPE BOOL
    CACHE_STRING "build zendnnl gtests"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set building zendnnl doxygen documentation, default os OFF.
  zendnnl_add_option(NAME ZENDNNL_BUILD_DOXYGEN
    VALUE OFF
    TYPE BOOL
    CACHE_STRING "build zendnnl doxygen documentation"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set building zendnnl benchmarking tool, default os OFF.
  zendnnl_add_option(NAME ZENDNNL_BUILD_BENCHDNN
    VALUE OFF
    TYPE BOOL
    CACHE_STRING "build zendnnl benchdnn"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set zendnnl code coverage option, default os OFF.
  zendnnl_add_option(NAME ZENDNNL_CODE_COVERAGE
    VALUE OFF
    TYPE BOOL
    CACHE_STRING "build zendnnl code coverage"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  ## dependencies
  # set if zendnnl depends on amdblis. this should bf OFF only if
  # aocldlp dependency is ON.
  zendnnl_add_option(NAME ZENDNNL_DEPENDS_AMDBLIS
    VALUE OFF
    TYPE BOOL
    CACHE_STRING "zendnnl amdblis dependency"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set if zendnnl depends on aocldlp. this should bf ON only if
  # amdblis dependency is OFF.
  zendnnl_add_option(NAME ZENDNNL_DEPENDS_AOCLDLP
    VALUE ON
    TYPE BOOL
    CACHE_STRING "zendnnl aocldlp dependency"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set if zendnnl depends on onednn, default is OFF.
  zendnnl_add_option(NAME ZENDNNL_DEPENDS_ONEDNN
    VALUE OFF
    TYPE BOOL
    CACHE_STRING "zendnnl onednn dependency"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set if zendnnl depends on libxsmm, default is OFF.
  zendnnl_add_option(NAME ZENDNNL_DEPENDS_LIBXSMM
    VALUE ON
    TYPE BOOL
    CACHE_STRING "zendnnl libxsmm dependency"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set path of amdblis if amdblis is injected. if the framework
  # does not inject it, set it to "" (empty string).
  zendnnl_add_option(NAME ZENDNNL_AMDBLIS_FWK_DIR
    VALUE ""
    TYPE PATH
    CACHE_STRING "zendnnl amdblis framework path"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set path of aocldlp if aocldlp is injected. if the framework
  # does not inject it, set it to "" (empty string).
  zendnnl_add_option(NAME ZENDNNL_AOCLDLP_FWK_DIR
    VALUE ""
    TYPE PATH
    CACHE_STRING "zendnnl aocldlp framework path"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set path of onednn if onednn is injected. if the framework
  # does not inject it, set it to "" (empty string).
  zendnnl_add_option(NAME ZENDNNL_ONEDNN_FWK_DIR
    VALUE ""
    TYPE PATH
    CACHE_STRING "zendnnl onednnn framework path"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # set path of libxsmm if libxsmm is injected. if the framework
  # does not inject it, set it to "" (empty string).
  zendnnl_add_option(NAME ZENDNNL_LIBXSMM_FWK_DIR
    VALUE ""
    TYPE PATH
    CACHE_STRING "zendnnl libxsmm framework path"
    COMMAND_LIST ZNL_CMAKE_ARGS)

  # try to find pre-built package
  set(zendnnl_ROOT "${ZENDNNL_INSTALL_PREFIX}/zendnnl")
  set(zendnnl_DIR "${zendnnl_ROOT}/lib/cmake")
  find_package(zendnnl QUIET)
  if(zendnnl_FOUND)
    message(STATUS "(ZENDNNL) ZENDNNL FOUND AT ${zendnnl_ROOT}")
    message(STATUS "(ZENDNNL) if zendnnl options are changed from previous build,")
    message(STATUS "(ZENDNNL) they will not be reflected")
    message(STATUS "(ZENDNNL) If options are changed, please do a clean build.")
    if(TARGET zendnnl::zendnnl_archive)
      set_target_properties(zendnnl::zendnnl_archive
        PROPERTIES IMPORTED_GLOBAL ON)
    else()
      message(FATAL_ERROR "(ZENDNNL) zendnnl installation does not have imported target zendnnl::zendnnl_archive")
    endif()
  else()
    message(STATUS "(ZENDNNL) ZENDNNL NOT FOUND, will be built as an external project.")

    # declare zendnnl library
    set(ZENDNNL_LIBRARY_INC_DIR "${ZENDNNL_INSTALL_PREFIX}/zendnnl/include")
    set(ZENDNNL_LIBRARY_LIB_DIR "${ZENDNNL_INSTALL_PREFIX}/zendnnl/lib")

    if(NOT EXISTS ${ZENDNNL_LIBRARY_INC_DIR})
      file(MAKE_DIRECTORY ${ZENDNNL_LIBRARY_INC_DIR})
    endif()

    add_library(zendnnl_library STATIC IMPORTED GLOBAL)
    add_dependencies(zendnnl_library fwk_zendnnl)
    set_target_properties(zendnnl_library
      PROPERTIES
      IMPORTED_LOCATION "${ZENDNNL_LIBRARY_LIB_DIR}/libzendnnl_archive.a"
      INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}"
      INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}")

    target_link_options(zendnnl_library INTERFACE "-fopenmp")
    target_link_libraries(zendnnl_library
      INTERFACE OpenMP::OpenMP_CXX
      INTERFACE ${CMAKE_DL_LIBS})

    add_library(zendnnl::zendnnl_archive ALIAS zendnnl_library)

    list(APPEND ZNL_BYPRODUCTS "${ZENDNNL_LIBRARY_LIB_DIR}/libzendnnl_archive.a")

    # declare all dependencies

    # json dependency
    zendnnl_add_dependency(NAME json
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/json"
      ALIAS "nlohmann_json::nlohmann_json"
      INCLUDE_ONLY)

    target_link_libraries(zendnnl_library INTERFACE nlohmann_json::nlohmann_json)

    # aoclutils dependency
    if (DEFINED ENV{ZENDNNL_MANYLINUX_BUILD})

      zendnnl_add_dependency(NAME aoclutils
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
        LIB_SUFFIX lib64
        ARCHIVE_FILE "libaoclutils.a"
        ALIAS "au::aoclutils")

      target_link_libraries(zendnnl_library INTERFACE au::aoclutils)

      zendnnl_add_dependency(NAME aucpuid
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
        LIB_SUFFIX lib64
        ARCHIVE_FILE "libau_cpuid.a"
        ALIAS "au::au_cpuid")

      target_link_libraries(zendnnl_library INTERFACE au::au_cpuid)

    else()
      zendnnl_add_dependency(NAME aoclutils
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
        ARCHIVE_FILE "libaoclutils.a"
        ALIAS "au::aoclutils")

      target_link_libraries(zendnnl_library INTERFACE au::aoclutils)

      zendnnl_add_dependency(NAME aucpuid
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
        ARCHIVE_FILE "libau_cpuid.a"
        ALIAS "au::au_cpuid")

      target_link_libraries(zendnnl_library INTERFACE au::au_cpuid)

    endif()

    # amdblis dependency
    if (ZENDNNL_DEPENDS_AMDBLIS)
      zendnnl_add_dependency(NAME amdblis
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/amdblis"
        ARCHIVE_FILE "libblis-mt.a"
        ALIAS "amdblis::amdblis_archive")

      target_link_libraries(zendnnl_library INTERFACE amdblis::amdblis_archive)
    endif()

    if (ZENDNNL_DEPENDS_AOCLDLP)
        zendnnl_add_dependency(NAME aocldlp
          PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aocldlp"
          ARCHIVE_FILE "libaocl-dlp.a"
          ALIAS "aocldlp::aocl_dlp_static")

        target_link_libraries(zendnnl_library INTERFACE aocldlp::aocl_dlp_static)
    endif()

    if (ZENDNNL_DEPENDS_ONEDNN)
        zendnnl_add_dependency(NAME onednn
          PATH "${ZENDNNL_INSTALL_PREFIX}/deps/onednn"
          ARCHIVE_FILE "libdnnl.a"
          ALIAS "DNNL::dnnl")

        target_link_libraries(zendnnl_library INTERFACE DNNL::dnnl)
    endif()

      # libxsmm dependency
    if (ZENDNNL_DEPENDS_LIBXSMM)
        zendnnl_add_dependency(NAME libxsmm
          PATH "${ZENDNNL_INSTALL_PREFIX}/deps/libxsmm"
          ARCHIVE_FILE "libxsmm.a"
          ALIAS "libxsmm::libxsmm_archive")

        target_link_libraries(zendnnl_library INTERFACE libxsmm::libxsmm_archive)
    endif()

    message(STATUS "(ZENDNNL) ZNL_BYPRODUCTS=${ZNL_BYPRODUCTS}")
    message(STATUS "(ZENDNNL) ZNL_CMAKE_ARGS=${ZNL_CMAKE_ARGS}")

    ExternalProject_ADD(fwk_zendnnl
      SOURCE_DIR  "${ZENDNNL_SOURCE_DIR}"
      BINARY_DIR  "${ZENDNNL_BINARY_DIR}"
      CMAKE_ARGS  "${ZNL_CMAKE_ARGS}"
      BUILD_COMMAND cmake --build . --target all -j
      INSTALL_COMMAND ""
      BUILD_BYPRODUCTS ${ZNL_BYPRODUCTS})

    list(APPEND ZENDNNL_CLEAN_FILES "${ZENDNNL_BINARY_DIR}")
    list(APPEND ZENDNNL_CLEAN_FILES "${ZENDNNL_INSTALL_PREFIX}")
    set_target_properties(fwk_zendnnl
      PROPERTIES
      ADDITIONAL_CLEAN_FILES "${ZENDNNL_CLEAN_FILES}")

    # framework dependencies
    # add_dependencies(fwk_zendnnl <injected dependency targets>)
    get_target_property(FWK_ZENDNNL_DEPENDS fwk_zendnnl MANUALLY_ADDED_DEPENDENCIES)
    if(${FWK_ZENDNNL_DEPENDS} STREQUAL "FWK_ZENDNNL_DEPENDS-NOTFOUND")
      message(AUTHOR_WARNING "(ZENDNNL) please ensure fwk_zendnnl depends on injected dependencies targets")
    else()
      message(STATUS "fwk_zendnnl dependencies : ${FWK_ZENDNNL_DEPENDS}")
    endif()

    # make library and its dependencies depend on fwk_zendnnl
    add_dependencies(zendnnl_library fwk_zendnnl)
    add_dependencies(zendnnl_json_deps fwk_zendnnl)
    add_dependencies(zendnnl_aoclutils_deps fwk_zendnnl)
    add_dependencies(zendnnl_aucpuid_deps fwk_zendnnl)

    if(ZENDNNL_DEPENDS_AMDBLIS)
      add_dependencies(zendnnl_amdblis_deps fwk_zendnnl)
    endif()

    if(ZENDNNL_DEPENDS_AOCLDLP)
      add_dependencies(zendnnl_aocldlp_deps fwk_zendnnl)
    endif()

    if(ZENDNNL_DEPENDS_ONEDNN)
      add_dependencies(zendnnl_onednn_deps fwk_zendnnl)
    endif()

    if(ZENDNNL_DEPENDS_LIBXSMM)
      add_dependencies(zendnnl_libxsmm_deps fwk_zendnnl)
    endif()
  endif()
  set(ZENDNN_FOUND TRUE)

endif(NOT ZENDNN_FOUND)
