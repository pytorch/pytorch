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

# -- [ Deterine commit hash
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "from tools.generate_torch_version import get_sha;print(get_sha('.'), end='')"
    OUTPUT_VARIABLE COMMIT_SHA
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
)

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

  file(GLOB_RECURSE all_python "${CMAKE_CURRENT_LIST_DIR}/../torchgen/*.py")

  # Handle files that may need sm89/sm90a/sm100a flags (stable/nightly
  # builds are not built for these archs).
  if(USE_CUDA)
    # The stable/nightly builds do not enable some SM architectures,
    # like 89/90a/100a.  Still, some files need to be built for these
    # architecturs specifically.  This function makes it possible to
    # enable building given file for a specific such architecture, in
    # case if PyTorch is built for corresponding other architecture;
    # for example, it will enable building for SM 90a in case PyTorch
    # built for SM 90, etc.  For examples of how to use the function,
    # see below the function itself.
    function(_BUILD_FOR_ADDITIONAL_ARCHS file archs)
      torch_cuda_get_nvcc_gencode_flag(_existing_arch_flags)

      set(_file_compile_flags "")
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
        foreach(_arch ${archs})
          if("${_arch}" STREQUAL "89")
            if(_existing_arch_flags MATCHES ".*compute_86.*")
              list(APPEND _file_compile_flags "-gencode;arch=compute_89,code=sm_89")
            endif()
          endif()
          if("${_arch}" STREQUAL "90a")
            if(_existing_arch_flags MATCHES ".*compute_90.*")
              list(APPEND _file_compile_flags "-gencode;arch=compute_90a,code=sm_90a")
            endif()
          endif()
          if("${_arch}" STREQUAL "100a")
            if(_existing_arch_flags MATCHES ".*compute_100.*")
              list(APPEND _file_compile_flags "-gencode;arch=compute_100a,code=sm_100a")
            endif()
          endif()
        endforeach()
      endif()
      list(JOIN _file_compile_flags " " _file_compile_flags)

      set_source_files_properties(${file} PROPERTIES COMPILE_FLAGS "${_file_compile_flags}")
    endfunction()

    _BUILD_FOR_ADDITIONAL_ARCHS(
      "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/cuda/RowwiseScaledMM.cu"
      "89;90a;100a")
    _BUILD_FOR_ADDITIONAL_ARCHS(
      "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/cuda/ScaledGroupMM.cu"
      "90a")
    _BUILD_FOR_ADDITIONAL_ARCHS(
      "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/cuda/GroupMM.cu"
      "90a")

  endif()

  set(GEN_ROCM_FLAG)
  if(USE_ROCM)
    set(GEN_ROCM_FLAG --rocm)
  endif()

  set(GEN_MPS_FLAG)
  if(USE_MPS)
    set(GEN_MPS_FLAG --mps)
  endif()

  set(GEN_XPU_FLAG)
  if(USE_XPU)
    set(GEN_XPU_FLAG --xpu)
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
    message(STATUS "Custom build with static dispatch backends: ${STATIC_DISPATCH_BACKEND}")
    list(LENGTH STATIC_DISPATCH_BACKEND len)
    list(APPEND CUSTOM_BUILD_FLAGS
      --static_dispatch_backend ${STATIC_DISPATCH_BACKEND})
  endif()

  # Codegen unboxing
  if(USE_LIGHTWEIGHT_DISPATCH)
    file(GLOB_RECURSE all_unboxing_script "${CMAKE_CURRENT_LIST_DIR}/../tools/jit/*.py")
    list(APPEND CUSTOM_BUILD_FLAGS --skip_dispatcher_op_registration)
    set(GEN_UNBOXING_COMMAND
        "${Python_EXECUTABLE}" -m tools.jit.gen_unboxing
        --source-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen
        --install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen
        )
    if(SELECTED_OP_LIST)
      list(APPEND GEN_UNBOXING_COMMAND
              --TEST_ONLY_op_registration_allowlist_yaml_path "${SELECTED_OP_LIST}")
    endif()
    set("GEN_UNBOXING_COMMAND_sources"
        ${GEN_UNBOXING_COMMAND}
        --output-dependencies ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_unboxing_sources.cmake
        )
    message(STATUS "Generating sources for lightweight dispatch")
    execute_process(
        COMMAND ${GEN_UNBOXING_COMMAND_sources} --dry-run
        RESULT_VARIABLE RETURN_VALUE
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    )
    if(NOT RETURN_VALUE EQUAL 0)
      message(FATAL_ERROR "Failed to get generated_unboxing_sources list")
    endif()

    include("${CMAKE_BINARY_DIR}/aten/src/ATen/generated_unboxing_sources.cmake")
    add_custom_command(
        COMMENT "Generating ATen unboxing sources"
        OUTPUT
        ${generated_unboxing_sources}
        ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_unboxing_sources.cmake
        COMMAND ${GEN_UNBOXING_COMMAND_sources}
        DEPENDS ${all_unboxing_script} ${sources_templates}
        ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/native_functions.yaml
        ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/tags.yaml
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    )
  else() # Otherwise do not generate or include sources into build.
    set(generated_unboxing_sources "")
  endif()

  set(GEN_PER_OPERATOR_FLAG)
  if(USE_PER_OPERATOR_HEADERS)
    list(APPEND GEN_PER_OPERATOR_FLAG "--per-operator-headers")
  endif()

  set(GEN_COMMAND
      "${Python_EXECUTABLE}" -m torchgen.gen
      --source-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen
      --install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen
      ${GEN_PER_OPERATOR_FLAG}
      ${GEN_ROCM_FLAG}
      ${GEN_MPS_FLAG}
      ${GEN_XPU_FLAG}
      ${CUSTOM_BUILD_FLAGS}
  )

  file(GLOB_RECURSE headers_templates "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/templates/*\.h")
  file(GLOB_RECURSE sources_templates "${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/templates/*\.cpp")
  set(declarations_yaml_templates "")

  foreach(gen_type "headers" "sources" "declarations_yaml")
    # The codegen outputs may change dynamically as PyTorch is
    # developed, but add_custom_command only supports dynamic inputs.
    #
    # We work around this by generating a .cmake file which is
    # included below to set the list of output files. If that file
    # ever changes then cmake will be re-run automatically because it
    # was included and so we get fully dynamic outputs.

    set("GEN_COMMAND_${gen_type}"
        ${GEN_COMMAND}
        --generate ${gen_type}
        --output-dependencies ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_${gen_type}.cmake
    )

    # Dry run to bootstrap the output variables
    execute_process(
        COMMAND ${GEN_COMMAND_${gen_type}} --dry-run
        RESULT_VARIABLE RETURN_VALUE
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    )

    if(NOT RETURN_VALUE EQUAL 0)
      message(FATAL_ERROR "Failed to get generated_${gen_type} list")
    endif()

    include("${CMAKE_BINARY_DIR}/aten/src/ATen/generated_${gen_type}.cmake")
    include("${CMAKE_BINARY_DIR}/aten/src/ATen/core_generated_${gen_type}.cmake")
    include("${CMAKE_BINARY_DIR}/aten/src/ATen/cpu_vec_generated_${gen_type}.cmake")
    include("${CMAKE_BINARY_DIR}/aten/src/ATen/cuda_generated_${gen_type}.cmake")
    include("${CMAKE_BINARY_DIR}/aten/src/ATen/ops_generated_${gen_type}.cmake")
    if(USE_XPU)
        include("${CMAKE_BINARY_DIR}/aten/src/ATen/xpu_generated_${gen_type}.cmake")
    endif()
    message(STATUS "${gen_type} outputs: ${gen_outputs}")
    set(OUTPUT_LIST
      ${generated_${gen_type}}
      ${cuda_generated_${gen_type}}
      ${core_generated_${gen_type}}
      ${cpu_vec_generated_${gen_type}}
      ${ops_generated_${gen_type}}
      ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_${gen_type}.cmake
      ${CMAKE_BINARY_DIR}/aten/src/ATen/ops_generated_${gen_type}.cmake
      ${CMAKE_BINARY_DIR}/aten/src/ATen/core_generated_${gen_type}.cmake
      ${CMAKE_BINARY_DIR}/aten/src/ATen/cpu_vec_generated_${gen_type}.cmake
      ${CMAKE_BINARY_DIR}/aten/src/ATen/cuda_generated_${gen_type}.cmake)
    if(USE_XPU)
      list(APPEND OUTPUT_LIST
        ${xpu_generated_${gen_type}}
        ${CMAKE_BINARY_DIR}/aten/src/ATen/xpu_generated_${gen_type}.cmake
      )
    endif()

    add_custom_command(
      COMMENT "Generating ATen ${gen_type}"
      OUTPUT ${OUTPUT_LIST}
      COMMAND ${GEN_COMMAND_${gen_type}}
      DEPENDS ${all_python} ${${gen_type}_templates}
        ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/native_functions.yaml
        ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/tags.yaml
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
    )
  endforeach()

  # Generated headers used from a CUDA (.cu) file are
  # not tracked correctly in CMake. We make the libATen.so depend explicitly
  # on building the generated ATen files to workaround.
  add_custom_target(ATEN_CPU_FILES_GEN_TARGET DEPENDS
      ${generated_headers} ${core_generated_headers} ${cpu_vec_generated_headers} ${ops_generated_headers}
      ${generated_sources} ${core_generated_sources} ${cpu_vec_generated_sources} ${ops_generated_sources}
      ${generated_declarations_yaml} ${generated_unboxing_sources})
  add_custom_target(ATEN_CUDA_FILES_GEN_TARGET DEPENDS
      ${cuda_generated_headers} ${cuda_generated_sources})
  add_library(ATEN_CPU_FILES_GEN_LIB INTERFACE)
  add_library(ATEN_CUDA_FILES_GEN_LIB INTERFACE)
  add_dependencies(ATEN_CPU_FILES_GEN_LIB ATEN_CPU_FILES_GEN_TARGET)
  add_dependencies(ATEN_CUDA_FILES_GEN_LIB ATEN_CUDA_FILES_GEN_TARGET)

  if(USE_PER_OPERATOR_HEADERS)
    target_compile_definitions(ATEN_CPU_FILES_GEN_LIB INTERFACE AT_PER_OPERATOR_HEADERS)
    target_compile_definitions(ATEN_CUDA_FILES_GEN_LIB INTERFACE AT_PER_OPERATOR_HEADERS)
  endif()

  if(USE_XPU)
    add_custom_target(ATEN_XPU_FILES_GEN_TARGET DEPENDS
        ${xpu_generated_headers} ${xpu_generated_sources})
    add_library(ATEN_XPU_FILES_GEN_LIB INTERFACE)
    add_dependencies(ATEN_XPU_FILES_GEN_LIB ATEN_XPU_FILES_GEN_TARGET)

    if(USE_PER_OPERATOR_HEADERS)
      target_compile_definitions(ATEN_XPU_FILES_GEN_LIB INTERFACE AT_PER_OPERATOR_HEADERS)
    endif()
  endif()
  # Handle source files that need to be compiled multiple times for
  # different vectorization options
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
        list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -mavx2 -mfma -mf16c ${CPU_NO_AVX256_SPLIT_FLAGS}")
      endif(MSVC)
    endif()
  endif(CXX_AVX2_FOUND)

  if(CXX_VSX_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_VSX_CPU_DEFINITION")
    LIST(APPEND CPU_CAPABILITY_NAMES "VSX")
    LIST(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}  ${CXX_VSX_FLAGS}")
  endif(CXX_VSX_FOUND)

  if(CXX_ZVECTOR_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_ZVECTOR_CPU_DEFINITION")
    LIST(APPEND CPU_CAPABILITY_NAMES "ZVECTOR")
    LIST(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}  ${CXX_ZVECTOR_FLAGS}")
  endif(CXX_ZVECTOR_FOUND)

  if(CXX_SVE_FOUND AND CXX_SVE256_FOUND AND CXX_ARM_BF16_FOUND)
    list(APPEND CPU_CAPABILITY_NAMES "SVE256")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_SVE_CPU_DEFINITION -DHAVE_SVE256_CPU_DEFINITION -DHAVE_ARM_BF16_CPU_DEFINITION")
    if("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -O2 -march=armv8-a+sve+bf16 -D__ARM_FEATURE_BF16 -DCPU_CAPABILITY_SVE -msve-vector-bits=256")
    else()
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG} -march=armv8-a+sve+bf16 -D__ARM_FEATURE_BF16 -DCPU_CAPABILITY_SVE -msve-vector-bits=256")
    endif()
  endif()

  list(LENGTH CPU_CAPABILITY_NAMES NUM_CPU_CAPABILITY_NAMES)
  math(EXPR NUM_CPU_CAPABILITY_NAMES "${NUM_CPU_CAPABILITY_NAMES}-1")

  # The sources list might get reordered later based on the capabilites.
  # See NOTE [ Linking AVX and non-AVX files ]
  foreach(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
    function(process_vec NAME)
      list(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
      set(NEW_IMPL ${CMAKE_BINARY_DIR}/aten/src/ATen/${NAME}.${CPU_CAPABILITY}.cpp)
      configure_file("${PROJECT_SOURCE_DIR}/cmake/IncludeSource.cpp.in" ${NEW_IMPL})
      set(cpu_kernel_cpp ${NEW_IMPL} ${cpu_kernel_cpp} PARENT_SCOPE) # Create list of copies
      list(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
      if(MSVC)
        set(EXTRA_FLAGS "/DCPU_CAPABILITY=${CPU_CAPABILITY} /DCPU_CAPABILITY_${CPU_CAPABILITY}")
      else(MSVC)
        set(EXTRA_FLAGS "-DCPU_CAPABILITY=${CPU_CAPABILITY} -DCPU_CAPABILITY_${CPU_CAPABILITY}")
      endif(MSVC)

      # Only parallelize the SortingKernel for now to avoid side effects
      if(${NAME} STREQUAL "native/cpu/SortingKernel.cpp" AND NOT MSVC AND USE_OMP)
        string(APPEND EXTRA_FLAGS " -D_GLIBCXX_PARALLEL")
      endif()

      # Disable certain warnings for GCC-9.X
      if(CMAKE_COMPILER_IS_GNUCXX)
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
    endfunction()
    foreach(IMPL ${cpu_kernel_cpp_in})
      file(RELATIVE_PATH NAME "${PROJECT_SOURCE_DIR}/aten/src/ATen/" "${IMPL}")
      process_vec("${NAME}")
    endforeach()
    foreach(IMPL ${cpu_vec_generated_sources})
      file(RELATIVE_PATH NAME "${CMAKE_BINARY_DIR}/aten/src/ATen/" "${IMPL}")
      process_vec("${NAME}")
    endforeach()
  endforeach()
  list(APPEND ATen_CPU_SRCS ${cpu_kernel_cpp})
endif()

function(append_filelist name outputvar)
  set(_rootdir "${Torch_SOURCE_DIR}/")
  # configure_file adds its input to the list of CMAKE_RERUN dependencies
  configure_file(
      ${PROJECT_SOURCE_DIR}/build_variables.bzl
      ${PROJECT_BINARY_DIR}/caffe2/build_variables.bzl)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "exec(open('${PROJECT_SOURCE_DIR}/build_variables.bzl').read());print(';'.join(['${_rootdir}' + x for x in ${name}]))"
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
