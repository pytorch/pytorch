# ##############################################################################
# Exclude and prepend functionalities
function(exclude OUTPUT INPUT)
  set(EXCLUDES ${ARGN})
  foreach(EXCLUDE ${EXCLUDES})
    list(REMOVE_ITEM INPUT "${EXCLUDE}")
  endforeach()
  set(${OUTPUT}
      ${INPUT}
      PARENT_SCOPE)
endfunction(exclude)

function(prepend OUTPUT PREPEND)
  set(OUT "")
  foreach(ITEM ${ARGN})
    list(APPEND OUT "${PREPEND}${ITEM}")
  endforeach()
  set(${OUTPUT}
      ${OUT}
      PARENT_SCOPE)
endfunction(prepend)

# ##############################################################################
# Parses a version string that might have values beyond major, minor, and patch
# and set version variables for the library. Usage:
# caffe2_parse_version_str(<library_name> <version_string>)
function(caffe2_parse_version_str LIBNAME VERSIONSTR)
  string(REGEX REPLACE "^([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MAJOR
                       "${VERSIONSTR}")
  string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MINOR
                       "${VERSIONSTR}")
  string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1"
                       ${LIBNAME}_VERSION_PATCH "${VERSIONSTR}")
  set(${LIBNAME}_VERSION_MAJOR
      ${${LIBNAME}_VERSION_MAJOR} ${ARGN}
      PARENT_SCOPE)
  set(${LIBNAME}_VERSION_MINOR
      ${${LIBNAME}_VERSION_MINOR} ${ARGN}
      PARENT_SCOPE)
  set(${LIBNAME}_VERSION_PATCH
      ${${LIBNAME}_VERSION_PATCH} ${ARGN}
      PARENT_SCOPE)
  set(${LIBNAME}_VERSION
      "${${LIBNAME}_VERSION_MAJOR}.${${LIBNAME}_VERSION_MINOR}.${${LIBNAME}_VERSION_PATCH}"
      PARENT_SCOPE)
endfunction()

#
# Removes common indentation from a block of text to produce code suitable for
# setting to `python -c`, or using with pycmd. This allows multiline code to be
# nested nicely in the surrounding code structure.
#
# This function respsects Python_EXECUTABLE if it defined, otherwise it uses
# `python` and hopes for the best. An error will be thrown if it is not found.
#
# Args: outvar : variable that will hold the stdout of the python command text :
# text to remove indentation from
#
function(dedent outvar text)
  # Use Python_EXECUTABLE if it is defined, otherwise default to python
  if("${Python_EXECUTABLE}" STREQUAL "")
    set(_python_exe "python3")
  else()
    set(_python_exe "${Python_EXECUTABLE}")
  endif()
  set(_fixup_cmd
      "import sys; from textwrap import dedent; print(dedent(sys.stdin.read()))"
  )
  file(WRITE "${CMAKE_BINARY_DIR}/indented.txt" "${text}")
  execute_process(
    COMMAND "${_python_exe}" -c "${_fixup_cmd}"
    INPUT_FILE "${CMAKE_BINARY_DIR}/indented.txt"
    RESULT_VARIABLE _dedent_exitcode
    OUTPUT_VARIABLE _dedent_text)
  if(NOT _dedent_exitcode EQUAL 0)
    message(ERROR " Failed to remove indentation from: \n\"\"\"\n${text}\n\"\"\"
    Python dedent failed with error code: ${_dedent_exitcode}")
    message(
      FATAL_ERROR " Python dedent failed with error code: ${_dedent_exitcode}")
  endif()
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_dedent_text}" _dedent_text)
  set(${outvar}
      "${_dedent_text}"
      PARENT_SCOPE)
endfunction()

function(pycmd_no_exit outvar exitcode cmd)
  # Use Python_EXECUTABLE if it is defined, otherwise default to python
  if("${Python_EXECUTABLE}" STREQUAL "")
    set(_python_exe "python")
  else()
    set(_python_exe "${Python_EXECUTABLE}")
  endif()
  # run the actual command
  execute_process(
    COMMAND "${_python_exe}" -c "${cmd}"
    RESULT_VARIABLE _exitcode
    OUTPUT_VARIABLE _output)
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_output}" _output)
  set(${outvar}
      "${_output}"
      PARENT_SCOPE)
  set(${exitcode}
      "${_exitcode}"
      PARENT_SCOPE)
endfunction()

#
# Helper function to run `python -c "<cmd>"` and capture the results of stdout
#
# Runs a python command and populates an outvar with the result of stdout.
# Common indentation in the text of `cmd` is removed before the command is
# executed, so the caller does not need to worry about indentation issues.
#
# This function respsects Python_EXECUTABLE if it defined, otherwise it uses
# `python` and hopes for the best. An error will be thrown if it is not found.
#
# Args: outvar : variable that will hold the stdout of the python command cmd :
# text representing a (possibly multiline) block of python code
#
function(pycmd outvar cmd)
  dedent(_dedent_cmd "${cmd}")
  pycmd_no_exit(_output _exitcode "${_dedent_cmd}")

  if(NOT _exitcode EQUAL 0)
    message(ERROR
            " Failed when running python code: \"\"\"\n${_dedent_cmd}\n\"\"\"")
    message(FATAL_ERROR " Python command failed with error code: ${_exitcode}")
  endif()
  # Remove supurflous newlines (artifacts of print)
  string(STRIP "${_output}" _output)
  set(${outvar}
      "${_output}"
      PARENT_SCOPE)
endfunction()

# ##############################################################################
# Macro to update cached options.
macro(caffe2_update_option variable value)
  if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
    get_property(
      __help_string
      CACHE ${variable}
      PROPERTY HELPSTRING)
    set(${variable}
        ${value}
        CACHE BOOL ${__help_string} FORCE)
  else()
    set(${variable} ${value})
  endif()
endmacro()

# ##############################################################################
# Add an interface library definition that is dependent on the source.
#
# It's probably easiest to explain why this macro exists, by describing what
# things would look like if we didn't have this macro.
#
# Let's suppose we want to statically link against torch.  We've defined a
# library in cmake called torch, and we might think that we just
# target_link_libraries(my-app PUBLIC torch).  This will result in a linker
# argument 'libtorch.a' getting passed to the linker.
#
# Unfortunately, this link command is wrong!  We have static initializers in
# libtorch.a that would get improperly pruned by the default link settings. What
# we actually need is for you to do -Wl,--whole-archive,libtorch.a
# -Wl,--no-whole-archive to ensure that we keep all symbols, even if they are
# (seemingly) not used.
#
# What caffe2_interface_library does is create an interface library that
# indirectly depends on the real library, but sets up the link arguments so that
# you get all of the extra link settings you need. The result is not a "real"
# library, and so we have to manually copy over necessary properties from the
# original target.
#
# (The discussion above is about static libraries, but a similar situation
# occurs for dynamic libraries: if no symbols are used from a dynamic library,
# it will be pruned unless you are --no-as-needed)
macro(caffe2_interface_library SRC DST)
  add_library(${DST} INTERFACE)
  add_dependencies(${DST} ${SRC})
  # Depending on the nature of the source library as well as the compiler,
  # determine the needed compilation flags.
  get_target_property(__src_target_type ${SRC} TYPE)
  # Depending on the type of the source library, we will set up the link command
  # for the specific SRC library.
  if(${__src_target_type} STREQUAL "STATIC_LIBRARY")
    # In the case of static library, we will need to add whole-static flags.
    target_link_libraries(${DST} INTERFACE $<LINK_LIBRARY:WHOLE_ARCHIVE,${SRC}>)
    # Link all interface link libraries of the src target as well. For static
    # library, we need to explicitly depend on all the libraries that are the
    # dependent library of the source library. Note that we cannot use the
    # populated INTERFACE_LINK_LIBRARIES property, because if one of the
    # dependent library is not a target, cmake creates a $<LINK_ONLY:src>
    # wrapper and then one is not able to find target "src". For more
    # discussions, check https://cmake.org/Bug/print_bug_page.php?bug_id=15415
    # https://cmake.org/pipermail/cmake-developers/2013-May/019019.html
    # Specifically the following quote
    #
    # """ For STATIC libraries we can define that the PUBLIC/PRIVATE/INTERFACE
    # keys are ignored for linking and that it always populates both
    # LINK_LIBRARIES LINK_INTERFACE_LIBRARIES.  Note that for STATIC libraries
    # the LINK_LIBRARIES property will not be used for anything except
    # build-order dependencies. """
    target_link_libraries(${DST}
                          INTERFACE $<TARGET_PROPERTY:${SRC},LINK_LIBRARIES>)
  elseif(${__src_target_type} STREQUAL "SHARED_LIBRARY")
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
      target_link_libraries(
        ${DST}
        INTERFACE "-Wl,--no-as-needed,\"$<TARGET_FILE:${SRC}>\" -Wl,--as-needed"
      )
    else()
      target_link_libraries(${DST} INTERFACE ${SRC})
    endif()
    # Link all interface link libraries of the src target as well. For shared
    # libraries, we can simply depend on the INTERFACE_LINK_LIBRARIES property
    # of the target.
    target_link_libraries(
      ${DST} INTERFACE $<TARGET_PROPERTY:${SRC},INTERFACE_LINK_LIBRARIES>)
  else()
    message(
      FATAL_ERROR
        "You made a CMake build file error: target " ${SRC}
        " must be of type either STATIC_LIBRARY or SHARED_LIBRARY. However, "
        "I got " ${__src_target_type} ".")
  endif()
  # For all other interface properties, manually inherit from the source target.
  set_target_properties(
    ${DST}
    PROPERTIES INTERFACE_COMPILE_DEFINITIONS
               $<TARGET_PROPERTY:${SRC},INTERFACE_COMPILE_DEFINITIONS>
               INTERFACE_COMPILE_OPTIONS
               $<TARGET_PROPERTY:${SRC},INTERFACE_COMPILE_OPTIONS>
               INTERFACE_INCLUDE_DIRECTORIES
               $<TARGET_PROPERTY:${SRC},INTERFACE_INCLUDE_DIRECTORIES>
               INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
               $<TARGET_PROPERTY:${SRC},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>)
endmacro()

# ##############################################################################
# Creating a Caffe2 binary target with sources specified with relative path.
# Usage: caffe2_binary_target(target_name_or_src <src1> [<src2>] [<src3>] ...)
# If only target_name_or_src is specified, this target is build with one single
# source file and the target name is autogen from the filename. Otherwise, the
# target name is given by the first argument and the rest are the source files
# to build the target.
function(caffe2_binary_target target_name_or_src)
  # https://cmake.org/cmake/help/latest/command/function.html Checking that ARGC
  # is greater than # is the only way to ensure that ARGV# was passed to the
  # function as an extra argument.
  if(ARGC GREATER 1)
    set(__target ${target_name_or_src})
    prepend(__srcs "${CMAKE_CURRENT_SOURCE_DIR}/" "${ARGN}")
  else()
    get_filename_component(__target ${target_name_or_src} NAME_WE)
    prepend(__srcs "${CMAKE_CURRENT_SOURCE_DIR}/" "${target_name_or_src}")
  endif()
  add_executable(${__target} ${__srcs})
  target_link_libraries(${__target} torch_library)
  # If we have Caffe2_MODULES defined, we will also link with the modules.
  if(DEFINED Caffe2_MODULES)
    target_link_libraries(${__target} ${Caffe2_MODULES})
  endif()
  install(TARGETS ${__target} DESTINATION bin)
endfunction()

function(caffe2_hip_binary_target target_name_or_src)
  if(ARGC GREATER 1)
    set(__target ${target_name_or_src})
    prepend(__srcs "${CMAKE_CURRENT_SOURCE_DIR}/" "${ARGN}")
  else()
    get_filename_component(__target ${target_name_or_src} NAME_WE)
    prepend(__srcs "${CMAKE_CURRENT_SOURCE_DIR}/" "${target_name_or_src}")
  endif()

  caffe2_binary_target(${target_name_or_src})

  target_compile_options(${__target} PRIVATE ${HIP_CXX_FLAGS})
  target_include_directories(${__target} PRIVATE ${Caffe2_HIP_INCLUDE})
endfunction()

# ##############################################################################
# Multiplex between adding libraries for CUDA versus HIP (AMD Software Stack).
# Usage: torch_cuda_based_add_library(cuda_target)
#
macro(torch_cuda_based_add_library cuda_target)
  if(USE_ROCM)
    hip_add_library(${cuda_target} ${ARGN})
  elseif(USE_CUDA)
    add_library(${cuda_target} ${ARGN})
  else()

  endif()
endmacro()

# ##############################################################################
# Get the HIP arch flags specified by PYTORCH_ROCM_ARCH. Usage:
# torch_hip_get_arch_list(variable_to_store_flags)
#
macro(torch_hip_get_arch_list store_var)
  if(DEFINED ENV{PYTORCH_ROCM_ARCH})
    set(_TMP $ENV{PYTORCH_ROCM_ARCH})
  else()
    # Use arch of installed GPUs as default
    execute_process(
      COMMAND "rocm_agent_enumerator"
      COMMAND bash "-c" "grep -v gfx000 | sort -u | xargs | tr -d '\n'"
      RESULT_VARIABLE ROCM_AGENT_ENUMERATOR_RESULT
      OUTPUT_VARIABLE ROCM_ARCH_INSTALLED)
    if(NOT ROCM_AGENT_ENUMERATOR_RESULT EQUAL 0)
      message(
        FATAL_ERROR
          " Could not detect ROCm arch for GPUs on machine. Result: '${ROCM_AGENT_ENUMERATOR_RESULT}'"
      )
    endif()
    set(_TMP ${ROCM_ARCH_INSTALLED})
  endif()
  string(REPLACE " " ";" ${store_var} "${_TMP}")
endmacro()

# ##############################################################################
# Get the XPU arch flags specified by TORCH_XPU_ARCH_LIST. Usage:
# torch_xpu_get_arch_list(variable_to_store_flags)
#
macro(torch_xpu_get_arch_list store_var)
  if(DEFINED ENV{TORCH_XPU_ARCH_LIST})
    set(${store_var} $ENV{TORCH_XPU_ARCH_LIST})
  endif()
endmacro()

# ##############################################################################
# Get the NVCC arch flags specified by TORCH_CUDA_ARCH_LIST and CUDA_ARCH_NAME.
# Usage: torch_cuda_get_nvcc_gencode_flag(variable_to_store_flags)
#
macro(torch_cuda_get_nvcc_gencode_flag store_var)
  # setting nvcc arch flags We need to support the explicitly and conveniently
  # defined TORCH_CUDA_ARCH_LIST
  if((NOT DEFINED TORCH_CUDA_ARCH_LIST) AND (DEFINED ENV{TORCH_CUDA_ARCH_LIST}))
    set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
  endif()
  if(DEFINED CUDA_ARCH_NAME)
    message(
      WARNING
        "CUDA_ARCH_NAME is no longer used. Use TORCH_CUDA_ARCH_LIST instead. "
        "Right now, CUDA_ARCH_NAME is ${CUDA_ARCH_NAME} and "
        "TORCH_CUDA_ARCH_LIST is ${TORCH_CUDA_ARCH_LIST}.")
    if(NOT TORCH_CUDA_ARCH_LIST)
      set(TORCH_CUDA_ARCH_LIST ${CUDA_ARCH_NAME})
    else()
      list(APPEND TORCH_CUDA_ARCH_LIST ${CUDA_ARCH_NAME})
    endif()
  endif()

  # Invoke cuda_select_nvcc_arch_flags from proper cmake FindCUDA.
  cuda_select_nvcc_arch_flags(${store_var} ${TORCH_CUDA_ARCH_LIST})
endmacro()

# ##############################################################################
# Add standard compile options. Usage: torch_compile_options(lib_name)
function(torch_compile_options libname)
  set_property(TARGET ${libname} PROPERTY CXX_STANDARD 17)

  # until they can be unified, keep these lists synced with setup.py
  if(MSVC)

    if(MSVC_Z7_OVERRIDE)
      set(MSVC_DEBINFO_OPTION "/Z7")
    else()
      set(MSVC_DEBINFO_OPTION "/Zi")
    endif()

    if(${MSVC_TOOLSET_VERSION} GREATER_EQUAL 142)
      # Add /permissive- flag for conformance mode to the compiler. This will
      # force more strict check to the code standard. 1. From MS official doc:
      # https://learn.microsoft.com/en-us/cpp/build/reference/permissive-standards-conformance?view=msvc-170#remarks
      # By default, the /permissive- option is set in new projects created by
      # Visual Studio 2017 version 15.5 and later versions. We set the
      # /permissive- flag from VS 2019 (MSVC_TOOLSET_VERSION 142) to avoid
      # compiling issues for old toolkit. 2. For MSVC VERSION:
      # https://cmake.org/cmake/help/latest/variable/MSVC_TOOLSET_VERSION.html
      target_compile_options(${libname}
                             PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
    endif()
    # This option enables a token-based preprocessor that conforms to C99 and
    # C++11 and later standards. This option is available since VS 2017. For MS
    # official doc:
    # https://learn.microsoft.com/en-us/cpp/build/reference/zc-preprocessor
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} /Zc:preprocessor"
        PARENT_SCOPE)

    if(${MSVC_TOOLSET_VERSION} GREATER_EQUAL 143)
      # Add /d2implyavx512upperregs- to disable compiler over-aggressive
      # optimization, which caused involeved AVX512 register on AVX2 machine.
      # Reference:
      # https://github.com/pytorch/pytorch/issues/145702#issuecomment-2874029459
      target_compile_options(
        ${libname} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/d2implyavx512upperregs->)
    endif()

    target_compile_options(
      ${libname}
      PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>: ${MSVC_RUNTIME_LIBRARY_OPTION}
        $<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${MSVC_DEBINFO_OPTION}>
        /EHsc /bigobj>)
  else()
    set(private_compile_options
        -Wall
        -Wextra
        -Wdeprecated
        -Wunused
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        -Wno-array-bounds
        -Wno-unknown-pragmas
        -Wno-strict-overflow
        -Wno-strict-aliasing)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      list(APPEND private_compile_options -Wredundant-move)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      list(APPEND private_compile_options -Wextra-semi -Wmove)
    else()
      list(
        APPEND
        private_compile_options
        # Considered to be flaky.  See the discussion at
        # https://github.com/pytorch/pytorch/pull/9608
        -Wno-maybe-uninitialized)
    endif()

    if(WERROR)
      list(
        APPEND
        private_compile_options
        -Werror
        -Werror=ignored-attributes
        -Werror=inconsistent-missing-override
        -Werror=inconsistent-missing-destructor-override
        -Werror=pedantic
        -Werror=unused
        -Wno-error=unused-parameter)
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND private_compile_options -Werror=unused-but-set-variable)
      endif()
    endif()
  endif()

  target_compile_options(
    ${libname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${private_compile_options}>)
  if(USE_CUDA)
    foreach(option IN LISTS private_compile_options)
      if(CMAKE_CUDA_HOST_COMPILER_ID STREQUAL "GNU")
        if("${option}" STREQUAL "-Wextra-semi")
          continue()
        endif()
        if("${option}" STREQUAL "-Wunused-private-field")
          continue()
        endif()
      endif()
      target_compile_options(
        ${libname} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler ${option}>)
    endforeach()
  endif()

  if(NOT WIN32 AND NOT USE_ASAN)
    # Enable hidden visibility by default to make it easier to debug issues with
    # TORCH_API annotations. Hidden visibility with selective default visibility
    # behaves close enough to Windows' dllimport/dllexport.
    #
    # Unfortunately, hidden visibility messes up some ubsan warnings because
    # templated classes crossing library boundary get duplicated (but identical)
    # definitions. It's easier to just disable it.
    target_compile_options(${libname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
                                              -fvisibility=hidden>)
  endif()

  # Use -O2 for release builds (-O3 doesn't improve perf, and -Os results in
  # perf regression)
  target_compile_options(
    ${libname}
    PRIVATE
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-O2>
  )

endfunction()

# ##############################################################################
# Set old-style FindCuda.cmake compile flags from modern CMake cuda flags.
# Usage: torch_update_find_cuda_flags()
function(torch_update_find_cuda_flags)
  # Convert -O2 -Xcompiler="-O2 -Wall" to "-O2;-Xcompiler=-O2,-Wall"
  if(USE_CUDA)
    separate_arguments(FLAGS UNIX_COMMAND "${CMAKE_CUDA_FLAGS}")
    string(REPLACE " " "," FLAGS "${FLAGS}")
    set(CUDA_NVCC_FLAGS
        ${FLAGS}
        PARENT_SCOPE)

    separate_arguments(FLAGS_DEBUG UNIX_COMMAND "${CMAKE_CUDA_FLAGS_DEBUG}")
    string(REPLACE " " "," FLAGS_DEBUG "${FLAGS_DEBUG}")
    set(CUDA_NVCC_FLAGS_DEBUG
        "${FLAGS_DEBUG}"
        PARENT_SCOPE)

    separate_arguments(FLAGS_RELEASE UNIX_COMMAND "${CMAKE_CUDA_FLAGS_RELEASE}")
    string(REPLACE " " "," FLAGS_RELEASE "${FLAGS_RELEASE}")
    set(CUDA_NVCC_FLAGS_RELEASE
        "${FLAGS_RELEASE}"
        PARENT_SCOPE)

    separate_arguments(FLAGS_MINSIZEREL UNIX_COMMAND
                       "${CMAKE_CUDA_FLAGS_MINSIZEREL}")
    string(REPLACE " " "," FLAGS_MINSIZEREL "${FLAGS_MINSIZEREL}")
    set(CUDA_NVCC_FLAGS_MINSIZEREL
        "${FLAGS_MINSIZEREL}"
        PARENT_SCOPE)

    separate_arguments(FLAGS_RELWITHDEBINFO UNIX_COMMAND
                       "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")
    string(REPLACE " " "," FLAGS_RELWITHDEBINFO "${FLAGS_RELWITHDEBINFO}")
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO
        "${FLAGS_RELWITHDEBINFO}"
        PARENT_SCOPE)

    message(
      STATUS "Converting CMAKE_CUDA_FLAGS to CUDA_NVCC_FLAGS:\n"
             "    CUDA_NVCC_FLAGS                = ${FLAGS}\n"
             "    CUDA_NVCC_FLAGS_DEBUG          = ${FLAGS_DEBUG}\n"
             "    CUDA_NVCC_FLAGS_RELEASE        = ${FLAGS_RELEASE}\n"
             "    CUDA_NVCC_FLAGS_RELWITHDEBINFO = ${FLAGS_RELWITHDEBINFO}\n"
             "    CUDA_NVCC_FLAGS_MINSIZEREL     = ${FLAGS_MINSIZEREL}")
  endif()
endfunction()

include(CheckCXXCompilerFlag)

# ##############################################################################
# CHeck if given flag is supported and append it to provided outputvar Also
# define HAS_UPPER_CASE_FLAG_NAME variable Usage:
# append_cxx_flag_if_supported("-Werror" CMAKE_CXX_FLAGS)
function(append_cxx_flag_if_supported flag outputvar)
  string(TOUPPER "HAS${flag}" _FLAG_NAME)
  string(REGEX REPLACE "[=-]" "_" _FLAG_NAME "${_FLAG_NAME}")
  # GCC silents unknown -Wno-XXX flags, so we detect the corresponding -WXXX.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    string(REGEX REPLACE "Wno-" "W" new_flag "${flag}")
  else()
    set(new_flag ${flag})
  endif()
  check_cxx_compiler_flag("${new_flag}" ${_FLAG_NAME})
  if(${_FLAG_NAME})
    string(APPEND ${outputvar} " ${flag}")
    set(${outputvar}
        "${${outputvar}}"
        PARENT_SCOPE)
  endif()
endfunction()

function(target_compile_options_if_supported target flag)
  set(_compile_options "")
  append_cxx_flag_if_supported("${flag}" _compile_options)
  if(NOT "${_compile_options}" STREQUAL "")
    target_compile_options(${target} PRIVATE ${flag})
  endif()
endfunction()
