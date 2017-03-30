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

add_subdirectory(binaries)
add_subdirectory(contrib)
add_subdirectory(core)
add_subdirectory(cuda_rtc)
add_subdirectory(db)
add_subdirectory(distributed)
# add_subdirectory(experiments) # note, we may remove this folder at some point
add_subdirectory(image)
add_subdirectory(mkl)
add_subdirectory(mpi)
add_subdirectory(operators)
add_subdirectory(python)
add_subdirectory(queue)
add_subdirectory(sgd)
# add_subdirectory(test) # todo: use caffe2_gtest_main instead of gtest_main because we will need to call GlobalInit
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

# ---[ List of libraries to link with

# In the static linking + clang mode, cmake will fail to identify the build
# order because the lib becomes one single string -Wl,-force-load,libCaffe2_CPU.so
# As a result, we will create a Caffe2_MAIN_LIBS_ORDER variable simply to
# enforce the dependency.
set(Caffe2_MAIN_LIBS_ORDER ${Caffe2_MAIN_LIBS})

# Compile exposed libraries.
add_library(Caffe2_CPU ${Caffe2_CPU_SRCS} $<TARGET_OBJECTS:Caffe_PROTO> $<TARGET_OBJECTS:Caffe2_PROTO>)
target_link_libraries(Caffe2_CPU ${Caffe2_DEPENDENCY_LIBS})
install(TARGETS Caffe2_CPU DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
list(APPEND Caffe2_MAIN_LIBS_ORDER Caffe2_CPU)
if (BUILD_SHARED_LIBS)
  list(APPEND Caffe2_MAIN_LIBS Caffe2_CPU)
else()
  caffe_add_whole_archive_flag(Caffe2_CPU tmp)
  list(APPEND Caffe2_MAIN_LIBS ${tmp})
endif()

# Write the macros file.
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/caffe2/core)
file(WRITE ${PROJECT_BINARY_DIR}/caffe2/core/macros.h
     "// Automatically generated header file for caffe2 macros. These \n"
     "// macros are used to build the Caffe2 binary, and if you are \n"
     "// building a dependent library, they will need to be set as well \n"
     "// for your program to link correctly.\n\n"
     "#pragma once\n\n")
get_directory_property(tmp DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
foreach(item ${tmp})
  if (${item} MATCHES "CAFFE2.*")
    file(APPEND ${PROJECT_BINARY_DIR}/caffe2/core/macros.h
         "\n#ifndef ${item}\n#define ${item}\n#endif // ${item}\n")
  endif()
endforeach()

# Installing the header files
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/
        FILES_MATCHING PATTERN "*.h")
install(FILES ${PROJECT_BINARY_DIR}/caffe2/core/macros.h
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/caffe2/core/)

# ---[ CUDA library.
if(USE_CUDA)
  CUDA_ADD_LIBRARY(Caffe2_GPU ${Caffe2_GPU_SRCS})
  list(APPEND Caffe2_MAIN_LIBS_ORDER Caffe2_GPU)
  if (BUILD_SHARED_LIBS)
    target_link_libraries(Caffe2_GPU Caffe2_CPU ${Caffe2_DEPENDENCY_LIBS})
    list(APPEND Caffe2_MAIN_LIBS Caffe2_GPU)
  else()
    target_link_libraries(Caffe2_GPU ${Caffe2_DEPENDENCY_LIBS})
    caffe_add_whole_archive_flag(Caffe2_GPU tmp)
    list(APPEND Caffe2_MAIN_LIBS ${tmp})
  endif()
  install(TARGETS Caffe2_GPU DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  link_directories(${CMAKE_INSTALL_PREFIX}/lib)
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
    target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS} gtest_main)
    if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.0)
      target_compile_features(${test_name} PRIVATE cxx_range_for)
    endif()
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    install(TARGETS ${test_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
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
      caffe2_pybind11_state ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS}
      ${Caffe2_PYTHON_DEPENDENCY_LIBS})
  install(
      TARGETS caffe2_pybind11_state DESTINATION
      ${CMAKE_INSTALL_PREFIX}/caffe2/python)

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
        caffe2_pybind11_state_gpu ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS}
        ${Caffe2_PYTHON_DEPENDENCY_LIBS})
    install(
        TARGETS caffe2_pybind11_state_gpu DESTINATION
        ${CMAKE_INSTALL_PREFIX}/caffe2/python)
  endif()

  if (MSVC)
    # If we are building under windows, we will copy the file from
    # build/caffe2/python/{Debug,Release}/caffe2_pybind11_state.pyd
    # to its parent folder so that we can do in-build execution.
    add_custom_target(windows_python_copy_lib ALL)
    add_dependencies(windows_python_copy_lib caffe2_pybind11_state)
    add_custom_command(
        TARGET windows_python_copy_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/caffe2/python/${CMAKE_BUILD_TYPE}/caffe2_pybind11_state.pyd
        ${CMAKE_BINARY_DIR}/caffe2/python)
    if (USE_CUDA)
      add_dependencies(windows_python_copy_lib caffe2_pybind11_state_gpu)
      add_custom_command(
          TARGET windows_python_copy_lib POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_BINARY_DIR}/caffe2/python/${CMAKE_BUILD_TYPE}/caffe2_pybind11_state_gpu.pyd
          ${CMAKE_BINARY_DIR}/caffe2/python)
    endif()
  endif()
endif()

# ---[ Binaries.
if (BUILD_BINARY)
  set(Caffe2_ALL_BINARY_SRCS ${Caffe2_CPU_BINARY_SRCS})
  if (USE_CUDA)
    list(APPEND Caffe2_ALL_BINARY_SRCS ${Caffe2_GPU_BINARY_SRCS})
  endif()

  foreach(binary_src ${Caffe2_ALL_BINARY_SRCS})
    get_filename_component(bin_name ${binary_src} NAME_WE)
    add_executable(${bin_name} ${binary_src})
    add_dependencies(${bin_name} ${Caffe2_MAIN_LIBS_ORDER})
    target_link_libraries(${bin_name} ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS})
    install(TARGETS ${bin_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/binaries)
  endforeach()
endif()
