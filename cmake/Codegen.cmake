# This ill-named file does a number of things:
# - Installs Caffe2 header files (this has nothing to do with code generation)
# - Configures caffe2/core/macros.h
# - Creates an ATen target for its generated C++ files and adds it
#   as a dependency
# - Reads build lists defined in build_variables.bzl

################################################################################
# Helper functions
################################################################################

function(filter_list output input)
    unset(result)
    foreach(filename ${${input}})
        foreach(pattern ${ARGN})
            if("${filename}" MATCHES "${pattern}")
                list(APPEND result "${filename}")
            endif()
        endforeach()
    endforeach()
    set(${output} ${result} PARENT_SCOPE)
endfunction()

function(filter_list_exclude output input)
    unset(result)
    foreach(filename ${${input}})
        foreach(pattern ${ARGN})
            if(NOT "${filename}" MATCHES "${pattern}")
                list(APPEND result "${filename}")
            endif()
        endforeach()
    endforeach()
    set(${output} ${result} PARENT_SCOPE)
endfunction()

################################################################################

# ---[ Write the macros file
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/../caffe2/core/macros.h.in
    ${CMAKE_BINARY_DIR}/caffe2/core/macros.h)

# ---[ Installing the header files
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../caffe2
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
if(NOT INTERN_BUILD_ATEN_OPS)
  install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/core
          DESTINATION include/ATen
          FILES_MATCHING PATTERN "*.h")
endif()
install(FILES ${CMAKE_BINARY_DIR}/caffe2/core/macros.h
        DESTINATION include/caffe2/core)

# ---[ ATen specific
if(INTERN_BUILD_ATEN_OPS)
  if(MSVC)
    set(OPT_FLAG "/fp:strict ")
  else(MSVC)
    set(OPT_FLAG "-O3 ")
    if("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
      set(OPT_FLAG " ")
    endif()
  endif(MSVC)

  if(C_AVX_FOUND)
    if(MSVC)
      set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/../aten/src/TH/vector/AVX.cpp PROPERTIES COMPILE_FLAGS "${OPT_FLAG}/arch:AVX ${CXX_AVX_FLAGS}")
    else(MSVC)
      set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/../aten/src/TH/vector/AVX.cpp PROPERTIES COMPILE_FLAGS "${OPT_FLAG} ${CXX_AVX_FLAGS}")
    endif(MSVC)
  endif(C_AVX_FOUND)

  if(NOT MSVC AND NOT "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/../aten/src/TH/THAllocator.cpp PROPERTIES COMPILE_FLAGS "-fno-openmp")
  endif()

  file(GLOB cpu_kernel_cpp_in "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/cpu/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/quantized/cpu/kernels/*.cpp")

  list(APPEND CPU_CAPABILITY_NAMES "DEFAULT")
  list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}")

  if(CXX_AVX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX_CPU_DEFINITION")
    list(APPEND CPU_CAPABILITY_NAMES "AVX")
    if(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX")
    else(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -mavx")
    endif(MSVC)
  endif(CXX_AVX_FOUND)

  if(CXX_AVX2_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX2_CPU_DEFINITION")

    # Some versions of GCC pessimistically split unaligned load and store
    # instructions when using the default tuning. This is a bad choice on
    # new Intel and AMD processors so we disable it when compiling with AVX2.
    # See https://stackoverflow.com/questions/52626726/why-doesnt-gcc-resolve-mm256-loadu-pd-as-single-vmovupd#tab-top
    check_cxx_compiler_flag("-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store" COMPILER_SUPPORTS_NO_AVX256_SPLIT)
    if(COMPILER_SUPPORTS_NO_AVX256_SPLIT)
      set(CPU_NO_AVX256_SPLIT_FLAGS "-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store")
    endif(COMPILER_SUPPORTS_NO_AVX256_SPLIT)

    list(APPEND CPU_CAPABILITY_NAMES "AVX2")
    if(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX2")
    else(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -mavx2 -mfma ${CPU_NO_AVX256_SPLIT_FLAGS}")
    endif(MSVC)
  endif(CXX_AVX2_FOUND)

  if(CXX_VSX_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_VSX_CPU_DEFINITION")
    LIST(APPEND CPU_CAPABILITY_NAMES "VSX")
    LIST(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}  ${CXX_VSX_FLAGS}")
  endif(CXX_VSX_FOUND)

  list(LENGTH CPU_CAPABILITY_NAMES NUM_CPU_CAPABILITY_NAMES)
  math(EXPR NUM_CPU_CAPABILITY_NAMES "${NUM_CPU_CAPABILITY_NAMES}-1")

  # The sources list might get reordered later based on the capabilites.
  # See NOTE [ Linking AVX and non-AVX files ]
  foreach(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
    foreach(IMPL ${cpu_kernel_cpp_in})
      string(REPLACE "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/" "" NAME ${IMPL})
      list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
      set(NEW_IMPL ${CMAKE_BINARY_DIR}/aten/src/ATen/${NAME}.${CPU_CAPABILITY}.cpp)
      configure_file(${IMPL} ${NEW_IMPL} COPYONLY)
      set(cpu_kernel_cpp ${NEW_IMPL} ${cpu_kernel_cpp}) # Create list of copies
      list(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
      if(MSVC)
        set(EXTRA_FLAGS "/DCPU_CAPABILITY=${CPU_CAPABILITY} /DCPU_CAPABILITY_${CPU_CAPABILITY}")
      else(MSVC)
        set(EXTRA_FLAGS "-DCPU_CAPABILITY=${CPU_CAPABILITY} -DCPU_CAPABILITY_${CPU_CAPABILITY}")
      endif(MSVC)
      # Disable certain warnings for GCC-9.X
      if(CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0.0))
        if(("${NAME}" STREQUAL "native/cpu/GridSamplerKernel.cpp") AND ("${CPU_CAPABILITY}" STREQUAL "DEFAULT"))
          # See https://github.com/pytorch/pytorch/issues/38855
          set(EXTRA_FLAGS "${EXTRA_FLAGS} -Wno-uninitialized")
        endif()
        if("${NAME}" STREQUAL "native/quantized/cpu/kernels/QuantizedOpKernels.cpp")
          # See https://github.com/pytorch/pytorch/issues/38854
          set(EXTRA_FLAGS "${EXTRA_FLAGS} -Wno-deprecated-copy")
        endif()
      endif()
      set_source_files_properties(${NEW_IMPL} PROPERTIES COMPILE_FLAGS "${FLAGS} ${EXTRA_FLAGS}")
    endforeach()
  endforeach()
  list(APPEND ATen_CPU_SRCS ${cpu_kernel_cpp})

  file(GLOB_RECURSE all_python "${CMAKE_CURRENT_LIST_DIR}/../tools/codegen/*.py")

  set(GEN_ROCM_FLAG)
  if(USE_ROCM)
    set(GEN_ROCM_FLAG --rocm)
  endif()

  set(CUSTOM_BUILD_FLAGS)
  if(INTERN_BUILD_MOBILE)
    if(USE_VULKAN)
      list(APPEND CUSTOM_BUILD_FLAGS --backend_whitelist CPU QuantizedCPU Vulkan)
    else()
      list(APPEND CUSTOM_BUILD_FLAGS --backend_whitelist CPU QuantizedCPU)
    endif()
  endif()

  if(STATIC_DISPATCH_BACKENDS)
    message(STATUS "Custom build with static dispatch backends: ${STATIC_DISPATCH_BACKENDS}")
    list(APPEND CUSTOM_BUILD_FLAGS
      --static_dispatch_backends ${STATIC_DISPATCH_BACKENDS})
  endif()

  if(SELECTED_OP_LIST)
    # With static dispatch we can omit the OP_DEPENDENCY flag. It will not calculate the transitive closure
    # of used ops. It only needs to register used root ops.
    if(NOT STATIC_DISPATCH_BACKENDS AND NOT OP_DEPENDENCY)
      message(INFO "Use default op dependency graph .yaml file for custom build with dynamic dispatch.")
      set(OP_DEPENDENCY ${CMAKE_CURRENT_LIST_DIR}/../tools/code_analyzer/default_op_deps.yaml)
    endif()
    execute_process(
      COMMAND
      "${PYTHON_EXECUTABLE}" ${CMAKE_CURRENT_LIST_DIR}/../tools/code_analyzer/gen_op_registration_allowlist.py
      --op-dependency "${OP_DEPENDENCY}"
      --root-ops "${SELECTED_OP_LIST}"
      OUTPUT_VARIABLE OP_REGISTRATION_WHITELIST
    )
    separate_arguments(OP_REGISTRATION_WHITELIST)
    message(STATUS "Custom build with op registration whitelist: ${OP_REGISTRATION_WHITELIST}")
    list(APPEND CUSTOM_BUILD_FLAGS
      --force_schema_registration
      --op_registration_whitelist ${OP_REGISTRATION_WHITELIST})
  endif()

  set(GEN_COMMAND
      "${PYTHON_EXECUTABLE}" -m tools.codegen.gen
      --source-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen
      --install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen
      ${GEN_ROCM_FLAG}
      ${CUSTOM_BUILD_FLAGS}
      ${GEN_VULKAN_FLAGS}
  )

  execute_process(
      COMMAND ${GEN_COMMAND}
        --output-dependencies ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_cpp.txt
      RESULT_VARIABLE RETURN_VALUE
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
  )
  if(NOT RETURN_VALUE EQUAL 0)
      message(STATUS ${generated_cpp})
      message(FATAL_ERROR "Failed to get generated_cpp list")
  endif()
  # FIXME: the file/variable name lists cpp, but these list both cpp and .h files
  file(READ ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_cpp.txt generated_cpp)
  file(READ ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_cpp.txt-cuda cuda_generated_cpp)
  file(READ ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_cpp.txt-core core_generated_cpp)

  file(GLOB_RECURSE all_templates "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/templates/*")

  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/aten/src/ATen)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/aten/src/ATen/core)

  add_custom_command(OUTPUT ${generated_cpp} ${cuda_generated_cpp} ${core_generated_cpp}
    COMMAND ${GEN_COMMAND}
    DEPENDS ${all_python} ${all_templates}
      ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/native_functions.yaml
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    )

  # Generated headers used from a CUDA (.cu) file are
  # not tracked correctly in CMake. We make the libATen.so depend explicitly
  # on building the generated ATen files to workaround.
  add_custom_target(ATEN_CPU_FILES_GEN_TARGET DEPENDS ${generated_cpp} ${core_generated_cpp})
  add_custom_target(ATEN_CUDA_FILES_GEN_TARGET DEPENDS ${cuda_generated_cpp})
  add_library(ATEN_CPU_FILES_GEN_LIB INTERFACE)
  add_library(ATEN_CUDA_FILES_GEN_LIB INTERFACE)
  add_dependencies(ATEN_CPU_FILES_GEN_LIB ATEN_CPU_FILES_GEN_TARGET)
  add_dependencies(ATEN_CUDA_FILES_GEN_LIB ATEN_CUDA_FILES_GEN_TARGET)
endif()

function(append_filelist name outputvar)
  set(_rootdir "${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/")
  # configure_file adds its input to the list of CMAKE_RERUN dependencies
  configure_file(
      ${PROJECT_SOURCE_DIR}/tools/build_variables.bzl
      ${PROJECT_BINARY_DIR}/caffe2/build_variables.bzl)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "exec(open('${PROJECT_SOURCE_DIR}/tools/build_variables.bzl').read());print(';'.join(['${_rootdir}' + x for x in ${name}]))"
    WORKING_DIRECTORY "${_rootdir}"
    RESULT_VARIABLE _retval
    OUTPUT_VARIABLE _tempvar)
  if(NOT _retval EQUAL 0)
    message(FATAL_ERROR "Failed to fetch filelist ${name} from build_variables.bzl")
  endif()
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  list(APPEND ${outputvar} ${_tempvar})
  set(${outputvar} "${${outputvar}}" PARENT_SCOPE)
endfunction()

set(NUM_CPU_CAPABILITY_NAMES ${NUM_CPU_CAPABILITY_NAMES} PARENT_SCOPE)
set(CPU_CAPABILITY_FLAGS ${CPU_CAPABILITY_FLAGS} PARENT_SCOPE)
