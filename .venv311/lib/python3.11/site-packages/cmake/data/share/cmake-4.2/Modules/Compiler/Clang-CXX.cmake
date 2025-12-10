include(Compiler/Clang)
__compiler_clang(CXX)
__compiler_clang_cxx_standards(CXX)

if("x${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" STREQUAL "xGNU")
  if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
      AND CMAKE_GENERATOR MATCHES "Makefiles|WMake"
      AND CMAKE_DEPFILE_FLAGS_CXX)
    # dependencies are computed by the compiler itself
    set(CMAKE_CXX_DEPFILE_FORMAT gcc)
    set(CMAKE_CXX_DEPENDS_USE_COMPILER TRUE)
  endif()

  set(CMAKE_CXX_COMPILE_OPTIONS_EXPLICIT_LANGUAGE -x c++)
  set(CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-fvisibility-inlines-hidden")
endif()

if("x${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" STREQUAL "xMSVC")
  set(CMAKE_CXX_CLANG_TIDY_DRIVER_MODE "cl")
  set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE_DRIVER_MODE "cl")
  if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
      AND CMAKE_GENERATOR MATCHES "Makefiles"
      AND CMAKE_DEPFILE_FLAGS_CXX)
    set(CMAKE_CXX_DEPENDS_USE_COMPILER TRUE)
  endif()
endif()

if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16.0)
  if("x${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" STREQUAL "xGNU")
    if (CMAKE_CXX_COMPILER_CLANG_RESOURCE_DIR)
      set(_clang_scan_deps_resource_dir
        " -resource-dir \"${CMAKE_CXX_COMPILER_CLANG_RESOURCE_DIR}\"")
    else()
      set(_clang_scan_deps_resource_dir "")
    endif ()
    if (CMAKE_HOST_WIN32)
      # `rename` doesn't overwrite and doesn't retry in case of "target file is
      # busy".
      set(_clang_scan_deps_mv "\"${CMAKE_COMMAND}\" -E rename")
    else ()
      set(_clang_scan_deps_mv "mv")
    endif ()
    string(CONCAT CMAKE_CXX_SCANDEP_SOURCE
      "\"${CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS}\""
      " -format=p1689"
      " --"
      " <CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS>"
      " -x c++ <SOURCE> -c -o <OBJECT>"
      "${_clang_scan_deps_resource_dir}"
      " -MT <DYNDEP_FILE>"
      " -MD -MF <DEP_FILE>"
      # Write to a temporary file. If the scan fails, we do not want to update
      # the actual output file as `ninja` (at least) assumes that failed
      # commands either delete or leave output files alone. See Issue#25419.
      " > <DYNDEP_FILE>.tmp"
      # We cannot use `copy_if_different` as the rule does not have a feature
      # analogous to `ninja`'s `restat = 1`. It would also leave behind the
      # `.tmp` file.
      " && ${_clang_scan_deps_mv} <DYNDEP_FILE>.tmp <DYNDEP_FILE>")
    unset(_clang_scan_deps_resource_dir)
    unset(_clang_scan_deps_mv)
    set(CMAKE_CXX_MODULE_MAP_FORMAT "clang")
    set(CMAKE_CXX_MODULE_MAP_FLAG "@<MODULE_MAP_FILE>")
    set(CMAKE_CXX_MODULE_BMI_ONLY_FLAG "--precompile")
  endif()
endif()
