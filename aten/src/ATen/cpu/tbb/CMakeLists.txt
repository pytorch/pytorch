# Based on https://github.com/wjakob/tbb/blob/master/CMakeLists.txt
# All credit goes to Wenzel Jakob!

cmake_minimum_required (VERSION 2.8.12 FATAL_ERROR)
project (tbb CXX)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

if(POLICY CMP0058)
  cmake_policy(SET CMP0058 NEW)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(NOT TBB_ROOT_DIR)
  set(TBB_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
if(NOT TBB_INSTALL_EXPORT_NAME)
  set(TBB_INSTALL_EXPORT_NAME "Caffe2Targets")
endif()
if(NOT TBB_INSTALL_EXPORT_DESTINATION)
  set(TBB_INSTALL_EXPORT_DESTINATION lib)
endif()
if(NOT TBB_INSTALL_RUNTIME_DIR)
  set(TBB_INSTALL_RUNTIME_DIR bin)
endif()
if(NOT TBB_INSTALL_LIBRARY_DIR)
  set(TBB_INSTALL_LIBRARY_DIR lib)
endif()
if(NOT TBB_INSTALL_ARCHIVE_DIR)
  set(TBB_INSTALL_ARCHIVE_DIR lib)
endif()
if(NOT TBB_INSTALL_INCLUDE_DIR)
  set(TBB_INSTALL_INCLUDE_DIR "${TBB_ROOT_DIR}/include")
endif()

set(TBB_INCLUDES
  "${TBB_ROOT_DIR}/include"
  "${TBB_ROOT_DIR}/src"
  "${TBB_ROOT_DIR}/src/rml/include"
  ${CMAKE_CURRENT_BINARY_DIR})

option(TBB_BUILD_SHARED          "Build TBB shared library" ON)
option(TBB_BUILD_STATIC          "Build TBB static library" ON)
option(TBB_BUILD_TBBMALLOC       "Build TBB malloc library" ON)
option(TBB_BUILD_TBBMALLOC_PROXY "Build TBB malloc proxy library" ON)
option(TBB_BUILD_TESTS           "Build TBB tests and enable testing infrastructure" ON)
option(TBB_CI_BUILD              "Is this a continuous integration build?" OFF)

if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()

file(GLOB tbb_src "${TBB_ROOT_DIR}/src/tbb/*.cpp" "${TBB_ROOT_DIR}/src/old/*.cpp")
list(APPEND tbb_src ${TBB_ROOT_DIR}/src/rml/client/rml_tbb.cpp)
file(GLOB to_remove "${TBB_ROOT_DIR}/src/old/test*.cpp")
if (NOT "${to_remove}" STREQUAL "")
  list(REMOVE_ITEM tbb_src ${to_remove})
endif()

set(tbbmalloc_static_src
  src/tbbmalloc/backend.cpp
  src/tbbmalloc/large_objects.cpp
  src/tbbmalloc/backref.cpp
  src/tbbmalloc/tbbmalloc.cpp
  src/tbbmalloc/frontend.cpp
  src/tbb/itt_notify.cpp)

set(tbbmalloc_src ${tbbmalloc_static_src})

set(tbbmalloc_proxy_src
  src/tbbmalloc/proxy.cpp
  src/tbbmalloc/tbb_function_replacement.cpp)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(i386|x86_64)")
  if (NOT APPLE AND NOT MINGW)
    add_definitions(-DDO_ITT_NOTIFY)
  endif()
endif()

if (APPLE)
  # Disable annoying "has no symbols" warnings
  set(CMAKE_C_ARCHIVE_CREATE   "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_C_ARCHIVE_FINISH   "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
  set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
endif()

macro(CHECK_CXX_COMPILER_AND_LINKER_FLAGS _RESULT _CXX_FLAGS _LINKER_FLAGS)
  set(CMAKE_REQUIRED_FLAGS ${_CXX_FLAGS})
  set(CMAKE_REQUIRED_LIBRARIES ${_LINKER_FLAGS})
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_source_runs("#include <iostream>\nint main(int argc, char **argv) { std::cout << \"test\"; return 0; }" ${_RESULT})
  set(CMAKE_REQUIRED_FLAGS "")
  set(CMAKE_REQUIRED_LIBRARIES "")
endmacro()

# Prefer libc++ in conjunction with Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if (CMAKE_CXX_FLAGS MATCHES "-stdlib=libc\\+\\+")
    message(STATUS "TBB: using libc++.")
  else()
    CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPP "-stdlib=libc++" "-stdlib=libc++")
    if (HAS_LIBCPP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -D_LIBCPP_VERSION")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
      message(STATUS "TBB: using libc++.")
    else()
      message(STATUS "TBB: NOT using libc++.")
    endif()
  endif()
endif()

if (UNIX)
  add_definitions (-DUSE_PTHREAD)

   check_cxx_compiler_flag ("-std=c++11" SUPPORTS_STDCXX11)
  if (SUPPORTS_STDCXX11)
    set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
  endif ()

   check_cxx_compiler_flag ("-mrtm -Werror" SUPPORTS_MRTM)
  if (SUPPORTS_MRTM)
    set (CMAKE_CXX_FLAGS "-mrtm ${CMAKE_CXX_FLAGS}")
  endif ()

elseif(WIN32)
  if (MSVC)
    cmake_minimum_required (VERSION 3.1)
    enable_language(ASM_MASM)
    set(CMAKE_CXX_FLAGS "/GS- /Zc:wchar_t /Zc:forScope /DUSE_WINTHREAD ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "/D_CRT_SECURE_NO_DEPRECATE /D_WIN32_WINNT=0x0600 ${CMAKE_CXX_FLAGS}")
    check_cxx_compiler_flag ("/volatile:iso" SUPPORTS_VOLATILE_FLAG)
    if (SUPPORTS_VOLATILE_FLAG)
      set(CMAKE_CXX_FLAGS "/volatile:iso ${CMAKE_CXX_FLAGS}")
    endif ()
    set(CMAKE_CXX_FLAGS "/wd4267 /wd4800 /wd4146 /wd4244 /wd4577 /wd4018 ${CMAKE_CXX_FLAGS}")
    if (NOT CMAKE_SIZEOF_VOID_P)
       message(FATAL_ERROR "'CMAKE_SIZEOF_VOID_P' is undefined. Please delete your build directory and rerun CMake again!")
    endif()

     if (CMAKE_SIZEOF_VOID_P EQUAL 8)
      list(APPEND tbb_src "${TBB_ROOT_DIR}/src/tbb/intel64-masm/atomic_support.asm")
      list(APPEND tbb_src "${TBB_ROOT_DIR}/src/tbb/intel64-masm/itsx.asm")
      list(APPEND tbb_src "${TBB_ROOT_DIR}/src/tbb/intel64-masm/intel64_misc.asm")
      list(APPEND tbbmalloc_src "${TBB_ROOT_DIR}/src/tbb/intel64-masm/atomic_support.asm")
      set(CMAKE_ASM_MASM_FLAGS "/DEM64T=1 ${CMAKE_ASM_MASM_FLAGS}")
    else()
      list(APPEND tbb_src "${TBB_ROOT_DIR}/src/tbb/ia32-masm/atomic_support.asm"
        "${TBB_ROOT_DIR}/src/tbb/ia32-masm/itsx.asm src/tbb/ia32-masm/lock_byte.asm")
      # Enable SAFESEH feature for assembly (x86 builds only).
      set(CMAKE_ASM_MASM_FLAGS "/safeseh ${CMAKE_ASM_MASM_FLAGS}")
    endif()
  elseif (MINGW)
    add_definitions(-DUSE_WINTHREAD)
    add_definitions(-D_WIN32_WINNT=0x0502)
    set(CMAKE_CXX_FLAGS "-mthreads ${CMAKE_CXX_FLAGS}")
  endif ()
endif()

if (MSVC)
  set(ENABLE_RTTI "/EHsc /GR ")
  set(DISABLE_RTTI "/EHs- /GR- ")
elseif (UNIX)
  set(ENABLE_RTTI "-frtti -fexceptions ")
  set(DISABLE_RTTI "-fno-rtti -fno-exceptions ")
endif ()

##--------
#   - Added TBB_USE_GLIBCXX_VERSION macro to specify the version of GNU
#     libstdc++ when it cannot be properly recognized, e.g. when used
#     with Clang on Linux* OS. Inspired by a contribution from David A.
if (NOT TBB_USE_GLIBCXX_VERSION AND UNIX AND NOT APPLE)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    string(REPLACE "." "0" TBB_USE_GLIBCXX_VERSION ${CMAKE_CXX_COMPILER_VERSION})
  endif()
endif()

if (TBB_USE_GLIBCXX_VERSION)
   add_definitions(-DTBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION})
endif()

##-------

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  check_cxx_compiler_flag ("-flifetime-dse=1" SUPPORTS_FLIFETIME)
  if (SUPPORTS_FLIFETIME)
    add_definitions(-flifetime-dse=1)
  endif()
endif()

# Linker export definitions
if (APPLE)
  set (ARCH_PREFIX "mac")
elseif(WIN32)
  set (ARCH_PREFIX "win")
else()
  set (ARCH_PREFIX "lin")
endif()

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(ARCH_PREFIX "${ARCH_PREFIX}64")
else()
  set(ARCH_PREFIX "${ARCH_PREFIX}32")
endif()

if (MINGW)
  set (ARCH_PREFIX "${ARCH_PREFIX}-gcc")
  # there's no win32-gcc-tbb-export.def, use lin32-tbb-export.def
  execute_process (COMMAND ${CMAKE_COMMAND} -E copy ${TBB_ROOT_DIR}/src/tbb/lin32-tbb-export.def ${TBB_ROOT_DIR}/src/tbb/win32-gcc-tbb-export.def)
endif()

if (MSVC)
  add_custom_command(OUTPUT tbb.def
    COMMAND ${CMAKE_CXX_COMPILER} /TC /EP ${TBB_ROOT_DIR}/src/tbb/${ARCH_PREFIX}-tbb-export.def  -I ${TBB_ROOT_DIR}/include > tbb.def
    MAIN_DEPENDENCY ${TBB_ROOT_DIR}/src/tbb/${ARCH_PREFIX}-tbb-export.def
    COMMENT "Preprocessing tbb.def"
  )

  add_custom_command(OUTPUT tbbmalloc.def
    COMMAND ${CMAKE_CXX_COMPILER} /TC /EP ${TBB_ROOT_DIR}/src/tbbmalloc/${ARCH_PREFIX}-tbbmalloc-export.def  -I ${TBB_ROOT_DIR}/include >   tbbmalloc.def
    MAIN_DEPENDENCY ${TBB_ROOT_DIR}/src/tbbmalloc/${ARCH_PREFIX}-tbbmalloc-export.def
    COMMENT "Preprocessing tbbmalloc.def"
  )
else()
  add_custom_command(OUTPUT tbb.def
    COMMAND ${CMAKE_CXX_COMPILER} -xc++ -E ${TBB_ROOT_DIR}/src/tbb/${ARCH_PREFIX}-tbb-export.def  -I ${TBB_ROOT_DIR}/include -o tbb.def
    MAIN_DEPENDENCY ${TBB_ROOT_DIR}/src/tbb/${ARCH_PREFIX}-tbb-export.def
    COMMENT "Preprocessing tbb.def"
  )

  add_custom_command(OUTPUT tbbmalloc.def
    COMMAND ${CMAKE_CXX_COMPILER} -xc++ -E ${TBB_ROOT_DIR}/src/tbbmalloc/${ARCH_PREFIX}-tbbmalloc-export.def  -I ${TBB_ROOT_DIR}/include -o   tbbmalloc.def
    MAIN_DEPENDENCY ${TBB_ROOT_DIR}/src/tbbmalloc/${ARCH_PREFIX}-tbbmalloc-export.def
    COMMENT "Preprocessing tbbmalloc.def"
  )
endif()

add_custom_target(tbb_def_files DEPENDS tbb.def tbbmalloc.def)

# TBB library
if (TBB_BUILD_STATIC)
  add_library(tbb_static STATIC ${tbb_src})
  target_include_directories(tbb_static PRIVATE ${TBB_INCLUDES})
  set_property(TARGET tbb_static APPEND PROPERTY COMPILE_DEFINITIONS "__TBB_BUILD=1")
  set_property(TARGET tbb_static APPEND_STRING PROPERTY COMPILE_FLAGS ${ENABLE_RTTI})
  install(TARGETS tbb_static
          EXPORT ${TBB_INSTALL_EXPORT_NAME} DESTINATION ${TBB_INSTALL_EXPORT_DESTINATION}
          ARCHIVE DESTINATION ${TBB_INSTALL_ARCHIVE_DIR})
  if (MSVC)
    target_compile_definitions(tbb_static PUBLIC __TBB_NO_IMPLICIT_LINKAGE=1)
  endif()

  if (UNIX AND NOT APPLE)
    target_link_libraries(tbb_static PUBLIC pthread dl)
  endif()
endif()

if (TBB_BUILD_SHARED)
  add_library(tbb SHARED ${tbb_src})
  target_include_directories(tbb PRIVATE ${TBB_INCLUDES})
  set_property(TARGET tbb APPEND PROPERTY COMPILE_DEFINITIONS "__TBB_BUILD=1")
  set_property(TARGET tbb APPEND_STRING PROPERTY COMPILE_FLAGS ${ENABLE_RTTI})
  add_dependencies(tbb tbb_def_files)

  if (APPLE)
    set_property(TARGET tbb APPEND PROPERTY LINK_FLAGS "-Wl,-exported_symbols_list,\"${CMAKE_CURRENT_BINARY_DIR}/tbb.def\"")
  elseif (MSVC)
    set_property(TARGET tbb APPEND PROPERTY LINK_FLAGS "/DEF:\"${CMAKE_CURRENT_BINARY_DIR}/tbb.def\"")
  else ()
    set_property(TARGET tbb APPEND PROPERTY LINK_FLAGS "-Wl,-version-script,\"${CMAKE_CURRENT_BINARY_DIR}/tbb.def\"")
  endif()

  install(TARGETS tbb
          EXPORT ${TBB_INSTALL_EXPORT_NAME} DESTINATION ${TBB_INSTALL_EXPORT_DESTINATION}
          LIBRARY DESTINATION ${TBB_INSTALL_LIBRARY_DIR}
          ARCHIVE DESTINATION ${TBB_INSTALL_ARCHIVE_DIR}
          RUNTIME DESTINATION ${TBB_INSTALL_RUNTIME_DIR})
  if (UNIX AND NOT APPLE)
    target_link_libraries(tbb PUBLIC pthread dl)
  endif()
  if (MSVC)
    target_compile_definitions(tbb PUBLIC __TBB_NO_IMPLICIT_LINKAGE=1)
  endif()
endif()


if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # Quench a warning on GCC
  set_source_files_properties(${TBB_ROOT_DIR}/src/tbb/governor.cpp COMPILE_FLAGS "-Wno-missing-field-initializers ")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Quench a warning on Clang
  set_source_files_properties(${TBB_ROOT_DIR}/src/tbb/itt_notify.cpp COMPILE_FLAGS "-Wno-varargs ")
elseif(MSVC)
  # Quench a warning on MSVC
  set_source_files_properties(${TBB_ROOT_DIR}/src/tbb/scheduler.cpp COMPILE_FLAGS "/wd4458 ")
endif()

if(TBB_BUILD_TBBMALLOC)
  # TBB malloc library
  if (TBB_BUILD_STATIC)
    add_library(tbbmalloc_static STATIC ${tbbmalloc_static_src})
    target_include_directories(tbbmalloc_static PRIVATE ${TBB_INCLUDES})
    set_property(TARGET tbbmalloc_static APPEND PROPERTY COMPILE_DEFINITIONS "__TBBMALLOC_BUILD=1")
    set_property(TARGET tbbmalloc_static APPEND_STRING PROPERTY COMPILE_FLAGS ${DISABLE_RTTI})
    if (MSVC)
      target_compile_definitions(tbbmalloc_static PUBLIC __TBB_NO_IMPLICIT_LINKAGE=1 __TBBMALLOC_NO_IMPLICIT_LINKAGE=1)
    endif()
    install(TARGETS tbbmalloc_static
            EXPORT ${TBB_INSTALL_EXPORT_NAME} DESTINATION ${TBB_INSTALL_EXPORT_DESTINATION}
            ARCHIVE DESTINATION ${TBB_INSTALL_ARCHIVE_DIR})
  endif()

  if (TBB_BUILD_SHARED)
    add_library(tbbmalloc SHARED ${tbbmalloc_src})
    target_include_directories(tbbmalloc PRIVATE ${TBB_INCLUDES})
    set_property(TARGET tbbmalloc APPEND PROPERTY COMPILE_DEFINITIONS "__TBBMALLOC_BUILD=1")
    set_property(TARGET tbbmalloc APPEND_STRING PROPERTY COMPILE_FLAGS ${DISABLE_RTTI})
    add_dependencies(tbbmalloc tbb_def_files)
    if (APPLE)
      set_property(TARGET tbbmalloc APPEND PROPERTY LINK_FLAGS "-Wl,-exported_symbols_list,\"${CMAKE_CURRENT_BINARY_DIR}/tbbmalloc.def\"")
    elseif (MSVC)
      set_property(TARGET tbbmalloc APPEND PROPERTY LINK_FLAGS "/DEF:\"${CMAKE_CURRENT_BINARY_DIR}/tbbmalloc.def\"")
    else ()
      set_property(TARGET tbbmalloc APPEND PROPERTY LINK_FLAGS "-Wl,-version-script,\"${CMAKE_CURRENT_BINARY_DIR}/tbbmalloc.def\"")
    endif()
    if (MSVC)
      target_compile_definitions(tbbmalloc PUBLIC __TBB_NO_IMPLICIT_LINKAGE=1 __TBBMALLOC_NO_IMPLICIT_LINKAGE=1)
    endif()
    install(TARGETS tbbmalloc
            EXPORT ${TBB_INSTALL_EXPORT_NAME} DESTINATION ${TBB_INSTALL_EXPORT_DESTINATION}
            LIBRARY DESTINATION ${TBB_INSTALL_LIBRARY_DIR}
            ARCHIVE DESTINATION ${TBB_INSTALL_ARCHIVE_DIR}
            RUNTIME DESTINATION ${TBB_INSTALL_RUNTIME_DIR})
    if (UNIX AND NOT APPLE)
      target_link_libraries(tbbmalloc PUBLIC pthread dl)
    endif()
  endif()
endif()

if(TBB_BUILD_TBBMALLOC_PROXY)
  # TBB malloc proxy library
  if (TBB_BUILD_STATIC)
    add_library(tbbmalloc_proxy_static STATIC ${tbbmalloc_proxy_src})
    set_property(TARGET tbbmalloc_proxy_static APPEND PROPERTY COMPILE_DEFINITIONS "__TBBMALLOC_BUILD=1")
    set_property(TARGET tbbmalloc_proxy_static APPEND_STRING PROPERTY COMPILE_FLAGS ${DISABLE_RTTI})
    install(TARGETS tbbmalloc_proxy_static
            EXPORT ${TBB_INSTALL_EXPORT_NAME} DESTINATION ${TBB_INSTALL_EXPORT_DESTINATION}
            ARCHIVE DESTINATION ${TBB_INSTALL_ARCHIVE_DIR})
  endif()

  if (TBB_BUILD_SHARED)
    add_library(tbbmalloc_proxy SHARED ${tbbmalloc_proxy_src})
    set_property(TARGET tbbmalloc_proxy APPEND PROPERTY COMPILE_DEFINITIONS "__TBBMALLOC_BUILD=1")
    set_property(TARGET tbbmalloc_proxy APPEND_STRING PROPERTY COMPILE_FLAGS ${DISABLE_RTTI})
    target_link_libraries(tbbmalloc_proxy PUBLIC tbbmalloc)
    install(TARGETS tbbmalloc_proxy
            EXPORT ${TBB_INSTALL_EXPORT_NAME} DESTINATION ${TBB_INSTALL_EXPORT_DESTINATION}
            LIBRARY DESTINATION ${TBB_INSTALL_LIBRARY_DIR}
            ARCHIVE DESTINATION ${TBB_INSTALL_ARCHIVE_DIR}
            RUNTIME DESTINATION ${TBB_INSTALL_RUNTIME_DIR})
    if (UNIX AND NOT APPLE)
      target_link_libraries(tbbmalloc_proxy PUBLIC pthread dl)
    endif()
  endif()
endif()

install(DIRECTORY "${TBB_ROOT_DIR}/include/tbb" DESTINATION ${TBB_INSTALL_INCLUDE_DIR})

# version_string.ver
if (UNIX)
  execute_process (COMMAND date "+%a, %d %b %Y %H:%M:%S %z"
                   OUTPUT_VARIABLE _configure_date
                   OUTPUT_STRIP_TRAILING_WHITESPACE)
elseif (WIN32)
  execute_process (COMMAND cmd " /C date /T"
                   OUTPUT_VARIABLE _configure_date
                   OUTPUT_STRIP_TRAILING_WHITESPACE)
else ()
  set (_configure_date "Unknown")
endif()
include_directories (${CMAKE_BINARY_DIR})
configure_file (extra/version_string.ver.in version_string.ver @ONLY)
