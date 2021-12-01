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

  if(NOT MSVC AND NOT "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/MapAllocator.cpp PROPERTIES COMPILE_FLAGS "-fno-openmp")
  endif()

  file(GLOB cpu_kernel_cpp_in "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/cpu/*.cpp" "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/quantized/cpu/kernels/*.cpp")

  list(APPEND CPU_CAPABILITY_NAMES "DEFAULT")
  list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}")


  if(CXX_AVX512_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX512_CPU_DEFINITION")
    list(APPEND CPU_CAPABILITY_NAMES "AVX512")
    if(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX512")
    else(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma")
    endif(MSVC)
  endif(CXX_AVX512_FOUND)

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
    if(DEFINED ENV{ATEN_AVX512_256})
      if($ENV{ATEN_AVX512_256} MATCHES "TRUE")
        if(CXX_AVX512_FOUND)
          message("-- ATen AVX2 kernels will use 32 ymm registers")
          if(MSVC)
            list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX512")
          else(MSVC)
            list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -march=native ${CPU_NO_AVX256_SPLIT_FLAGS}")
          endif(MSVC)
        endif(CXX_AVX512_FOUND)
      endif()
    else()
      if(MSVC)
        list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX2")
      else(MSVC)
        list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -mavx2 -mfma ${CPU_NO_AVX256_SPLIT_FLAGS}")
      endif(MSVC)
    endif()
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
      file(RELATIVE_PATH NAME "${PROJECT_SOURCE_DIR}/aten/src/ATen/" "${IMPL}")
      list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
      set(NEW_IMPL ${CMAKE_BINARY_DIR}/aten/src/ATen/${NAME}.${CPU_CAPABILITY}.cpp)
      configure_file("${PROJECT_SOURCE_DIR}/cmake/IncludeSource.cpp.in" ${NEW_IMPL})
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

  if(SELECTED_OP_LIST)
    if(TRACING_BASED)
      message(STATUS "Running tracing-based selective build given operator list: ${SELECTED_OP_LIST}")
      list(APPEND CUSTOM_BUILD_FLAGS
        --op_selection_yaml_path ${SELECTED_OP_LIST})
    elseif(NOT STATIC_DISPATCH_BACKEND)
      message(WARNING
        "You have to run tracing-based selective build with dynamic dispatch.\n"
        "Switching to STATIC_DISPATCH_BACKEND=CPU."
      )
      set(STATIC_DISPATCH_BACKEND CPU)
    endif()
  endif()

  if(STATIC_DISPATCH_BACKEND)
    message(STATUS "Custom build with static dispatch backend: ${STATIC_DISPATCH_BACKEND}")
    list(APPEND CUSTOM_BUILD_FLAGS
      --static_dispatch_backend ${STATIC_DISPATCH_BACKEND})
  endif()

  set(GEN_COMMAND
      "${PYTHON_EXECUTABLE}" -m tools.codegen.gen
      --source-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen
      --install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen
      ${GEN_ROCM_FLAG}
      ${CUSTOM_BUILD_FLAGS}
      ${GEN_VULKAN_FLAGS}
  )

  foreach(gen_type "headers" "sources" "declarations_yaml")
    execute_process(
        COMMAND ${GEN_COMMAND}
          --generate ${gen_type}
          --output-dependencies ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_${gen_type}.txt
        RESULT_VARIABLE RETURN_VALUE
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    )

    if(NOT RETURN_VALUE EQUAL 0)
      message(FATAL_ERROR "Failed to get generated_${gen_type} list")
    endif()

    file(READ "${CMAKE_BINARY_DIR}/aten/src/ATen/generated_${gen_type}.txt" "generated_${gen_type}")
    file(READ "${CMAKE_BINARY_DIR}/aten/src/ATen/generated_${gen_type}.txt-cuda" "cuda_generated_${gen_type}")
    file(READ "${CMAKE_BINARY_DIR}/aten/src/ATen/generated_${gen_type}.txt-core" "core_generated_${gen_type}")
  endforeach()

  file(GLOB_RECURSE header_templates "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/templates/*\.h")
  file(GLOB_RECURSE source_templates "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/templates/*\.cpp")

  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/aten/src/ATen)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/aten/src/ATen/core)

  add_custom_command(
    OUTPUT ${generated_headers} ${cuda_generated_headers} ${core_generated_headers}
    COMMAND ${GEN_COMMAND} --generate headers
    DEPENDS ${all_python} ${header_templates}
      ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/native_functions.yaml
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    )

  add_custom_command(
    OUTPUT ${generated_sources} ${cuda_generated_sources} ${core_generated_sources}
    COMMAND ${GEN_COMMAND} --generate sources
    DEPENDS ${all_python} ${source_templates}
      ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/native_functions.yaml
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    )

  add_custom_command(
    OUTPUT
      ${generated_declarations_yaml}
      ${cuda_generated_declarations_yaml}
      ${core_generated_declarations_yaml}
    COMMAND ${GEN_COMMAND} --generate declarations_yaml
    DEPENDS ${all_python}
      ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/native_functions.yaml
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    )

  # Generated headers used from a CUDA (.cu) file are
  # not tracked correctly in CMake. We make the libATen.so depend explicitly
  # on building the generated ATen files to workaround.
  add_custom_target(ATEN_CPU_FILES_GEN_TARGET DEPENDS
      ${generated_headers} ${core_generated_headers}
      ${generated_sources} ${core_generated_sources}
      ${generated_declarations_yaml})
  add_custom_target(ATEN_CUDA_FILES_GEN_TARGET DEPENDS
      ${cuda_generated_headers} ${cuda_generated_sources})
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
