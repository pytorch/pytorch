# ---[ Declare source file lists

# Caffe2_{CPU,GPU}_SRCS is the list that will have all the related source
# files for CPU and GPU respectively. They will be filled with the
# CMakeLists.txt files under each folder respectively.
set(Caffe2_CPU_SRCS)
set(Caffe2_GPU_SRCS)

# Caffe2_{CPU,GPU}_TEST_SRCS is the list that will have all the related source
# files for CPU and GPU tests respectively.
set(Caffe2_CPU_TEST_SRCS)
set(Caffe2_GPU_TEST_SRCS)

# ---[ Add respective subdirectories
# Note: the folders that are being commented out have not been properly
# addressed yet.

add_subdirectory(proto)

add_subdirectory(contrib)
add_subdirectory(core)
add_subdirectory(cuda_rtc)
add_subdirectory(db)
add_subdirectory(distributed)
# add_subdirectory(experiments) # note, we may remove this folder at some point
add_subdirectory(image)
add_subdirectory(video)
add_subdirectory(mkl)
add_subdirectory(mobile)
add_subdirectory(mpi)
add_subdirectory(operators)
add_subdirectory(perfkernels)
add_subdirectory(python)
add_subdirectory(queue)
add_subdirectory(sgd)
if (BUILD_SHARE_DIR)
  add_subdirectory(share)
endif()
# add_subdirectory(test) # todo: use caffe2_gtest_main instead of gtest_main because we will need to call GlobalInit
add_subdirectory(transforms)
add_subdirectory(utils)

# Advanced: if we have white list specified, we will do intersections for all
# main lib srcs.
if (CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_CPU_SRCS CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_GPU_SRCS CAFFE2_WHITELISTED_FILES)
endif()

# Debug messages - if you want to get a list of source files, enable the
# following.
if (FALSE)
  message(STATUS "CPU sources: ")
  foreach(tmp ${Caffe2_CPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU sources: ")
  foreach(tmp ${Caffe2_GPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU test sources: ")
  foreach(tmp ${Caffe2_CPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU test sources: ")
  foreach(tmp ${Caffe2_GPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()
endif()

# ---[ Generate and install header files.

# Write the macros file.
configure_file(
    ${PROJECT_SOURCE_DIR}/caffe2/core/macros.h.in
    ${PROJECT_BINARY_DIR}/caffe2/core/macros.h)

# Installing the header files
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(FILES ${PROJECT_BINARY_DIR}/caffe2/core/macros.h
        DESTINATION include/caffe2/core)


# ---[ List of libraries to link with

# In the static linking + clang mode, cmake will fail to identify the build
# order because the lib becomes one single string -Wl,-force-load,libCaffe2_CPU.so
# As a result, we will create a Caffe2_MAIN_LIBS_ORDER variable simply to
# enforce the dependency.
set(Caffe2_MAIN_LIBS_ORDER)
set(Caffe2_MAIN_LIBS)

# Compile exposed libraries.
add_library(caffe2 ${Caffe2_CPU_SRCS} $<TARGET_OBJECTS:Caffe_PROTO> $<TARGET_OBJECTS:Caffe2_PROTO>)
target_link_libraries(caffe2 ${Caffe2_DEPENDENCY_LIBS})
install(TARGETS caffe2 DESTINATION lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
caffe_add_linker_flag(caffe2 Caffe2_CPU_LINK)
list(APPEND Caffe2_MAIN_LIBS_ORDER caffe2 Caffe2_PROTO)
list(APPEND Caffe2_MAIN_LIBS ${Caffe2_CPU_LINK})
if (Caffe2_EXTERNAL_DEPENDENCIES)
  add_dependencies(caffe2 ${Caffe2_EXTERNAL_DEPENDENCIES})
endif()

# ---[ CUDA library.
if(USE_CUDA)
  CUDA_ADD_LIBRARY(caffe2_gpu ${Caffe2_GPU_SRCS})
  list(APPEND Caffe2_MAIN_LIBS_ORDER caffe2_gpu)
  add_dependencies(caffe2_gpu Caffe2_PROTO)
  if (BUILD_SHARED_LIBS)
    target_link_libraries(
        caffe2_gpu
        caffe2
        ${Caffe2_DEPENDENCY_LIBS}
        ${Caffe2_CUDA_DEPENDENCY_LIBS})
  else()
    target_link_libraries(
        caffe2_gpu
        ${Caffe2_DEPENDENCY_LIBS}
        ${Caffe2_CUDA_DEPENDENCY_LIBS})
  endif()
  caffe_add_linker_flag(caffe2_gpu Caffe2_GPU_LINK)
  list(APPEND Caffe2_MAIN_LIBS_ORDER caffe2_gpu)
  list(APPEND Caffe2_MAIN_LIBS ${Caffe2_GPU_LINK})
  if (Caffe2_EXTERNAL_DEPENDENCIES)
    add_dependencies(caffe2_gpu ${Caffe2_EXTERNAL_DEPENDENCIES})
  endif()
  install(TARGETS caffe2_gpu DESTINATION lib)
endif()

# ---[ Test binaries.
if (BUILD_TEST)
  set(Caffe2_ALL_TEST_SRCS ${Caffe2_CPU_TEST_SRCS})
  if (USE_CUDA)
    list(APPEND Caffe2_ALL_TEST_SRCS ${Caffe2_GPU_TEST_SRCS})
  endif()

  foreach(test_src ${Caffe2_ALL_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    add_dependencies(${test_name} ${Caffe2_MAIN_LIBS_ORDER})
    if (USE_CUDA)
      target_link_libraries(
          ${test_name} ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS}
          ${Caffe2_CUDA_DEPENDENCY_LIBS} gtest_main)
    else()
      target_link_libraries(
          ${test_name} ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS}
          gtest_main)
    endif()
    if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.0)
      target_compile_features(${test_name} PRIVATE cxx_range_for)
    endif()
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    install(TARGETS ${test_name} DESTINATION test)
  endforeach()
endif()


if (BUILD_PYTHON)
  # ---[ Python.
  add_library(caffe2_pybind11_state MODULE ${Caffe2_CPU_PYTHON_SRCS})
  add_dependencies(caffe2_pybind11_state ${Caffe2_MAIN_LIBS_ORDER})
  set_target_properties(caffe2_pybind11_state PROPERTIES PREFIX "")
  if (APPLE)
    set_target_properties(caffe2_pybind11_state PROPERTIES SUFFIX ".so")
  elseif (MSVC)
    set_target_properties(caffe2_pybind11_state PROPERTIES SUFFIX ".pyd")
  endif()
  set_target_properties(
      caffe2_pybind11_state PROPERTIES LIBRARY_OUTPUT_DIRECTORY
      ${CMAKE_BINARY_DIR}/caffe2/python)
  target_link_libraries(
      caffe2_pybind11_state ${Caffe2_CPU_LINK} ${Caffe2_DEPENDENCY_LIBS}
      ${Caffe2_PYTHON_DEPENDENCY_LIBS})
  install(TARGETS caffe2_pybind11_state DESTINATION caffe2/python)

  if(USE_CUDA)
    add_library(caffe2_pybind11_state_gpu MODULE ${Caffe2_GPU_PYTHON_SRCS})
    add_dependencies(caffe2_pybind11_state_gpu ${Caffe2_MAIN_LIBS_ORDER})
    set_target_properties(caffe2_pybind11_state_gpu PROPERTIES PREFIX "")
    if (APPLE)
      set_target_properties(caffe2_pybind11_state_gpu PROPERTIES SUFFIX ".so")
    elseif (MSVC)
      set_target_properties(caffe2_pybind11_state_gpu PROPERTIES SUFFIX ".pyd")
    endif()
    set_target_properties(
        caffe2_pybind11_state_gpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY
        ${CMAKE_BINARY_DIR}/caffe2/python)
    target_link_libraries(
        caffe2_pybind11_state_gpu ${Caffe2_CPU_LINK} ${Caffe2_GPU_LINK} ${Caffe2_DEPENDENCY_LIBS}
        ${Caffe2_CUDA_DEPENDENCY_LIBS} ${Caffe2_PYTHON_DEPENDENCY_LIBS})
    install(TARGETS caffe2_pybind11_state_gpu DESTINATION caffe2/python)
  endif()

  if (MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
    # If we are building under windows, we will copy the file from
    # build/caffe2/python/{Debug,Release}/caffe2_pybind11_state.pyd
    # to its parent folder so that we can do in-build execution.
    add_custom_target(windows_python_copy_lib ALL)
    add_dependencies(windows_python_copy_lib caffe2_pybind11_state)
    add_custom_command(
        TARGET windows_python_copy_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:caffe2_pybind11_state>
        ${CMAKE_BINARY_DIR}/caffe2/python)
    if (USE_CUDA)
      add_dependencies(windows_python_copy_lib caffe2_pybind11_state_gpu)
      add_custom_command(
          TARGET windows_python_copy_lib POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:caffe2_pybind11_state_gpu>
          ${CMAKE_BINARY_DIR}/caffe2/python)
    endif()
  endif()

  # Finally, Copy all python files to build directory
  # Generate and create all needed __init__.py files, if they aren't already
  # present in the current source tree.
  message(STATUS "Automatically generating missing __init__.py files.")
  caffe_autogen_init_py_files()

  # Create a custom target that copies all python files.
  file(GLOB_RECURSE PYTHON_SRCS RELATIVE ${PROJECT_SOURCE_DIR}
       "${PROJECT_SOURCE_DIR}/caffe2/*.py")
  add_custom_target(python_copy_files ALL)
  if(MSVC AND CMAKE_GENERATOR MATCHES "Ninja")
    # ninja fails when the command line is too long so we split
    # the target into several. This would be beneficial for VS also
    # since it build targets in parallel but not custom commands
    foreach(python_src ${PYTHON_SRCS})
      get_filename_component(dir ${python_src} DIRECTORY)
      string(SHA1 name_hash "${python_src}")
      # get_filename_component(name_we ${python_src} NAME_WE)
      add_custom_target(python_copy_files_${name_hash}
          COMMAND ${CMAKE_COMMAND} -E copy
          ${PROJECT_SOURCE_DIR}/${python_src} ${CMAKE_BINARY_DIR}/${dir})
      add_dependencies(python_copy_files python_copy_files_${name_hash})
    endforeach()
  else()
    foreach(python_src ${PYTHON_SRCS})
      get_filename_component(dir ${python_src} DIRECTORY)
      add_custom_command(
          TARGET python_copy_files PRE_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          ${PROJECT_SOURCE_DIR}/${python_src} ${CMAKE_BINARY_DIR}/${dir})
    endforeach()
  endif()

  # Install commands
  # Pick up static python files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe2 DESTINATION .
          FILES_MATCHING PATTERN "*.py")
  # Caffe proto files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe DESTINATION .
          FILES_MATCHING PATTERN "*.py")
  # Caffe2 proto files
  install(DIRECTORY ${CMAKE_BINARY_DIR}/caffe2 DESTINATION .
          FILES_MATCHING PATTERN "*.py")
endif()


# ---[ Any builds that should happen after the main targets should be added here.
# Binaries
if (BUILD_BINARY)
  add_subdirectory(binaries)
  # There is a linking issue that happens in some of the Windows builds.
  # TODO(Yangqing): after the module redesing, enable this back.
  if (NOT MSVC AND BUILD_SHARE_DIR)
    add_subdirectory(share/contrib/binaries)
  endif()
endif()
