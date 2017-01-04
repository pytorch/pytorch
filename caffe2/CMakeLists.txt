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

add_subdirectory(binaries)
add_subdirectory(contrib)
add_subdirectory(core)
add_subdirectory(cuda_rtc)
add_subdirectory(db)
# add_subdirectory(distributed)
# add_subdirectory(experiments) # note, we may remove this folder at some point
add_subdirectory(image)
add_subdirectory(mkl)
add_subdirectory(mpi)
add_subdirectory(operators)
# add_subdirectory(proto) # note, proto is processed above already.
add_subdirectory(python)
add_subdirectory(queue)
add_subdirectory(sgd)
# add_subdirectory(test) # todo: use caffe2_gtest_main instead of gtest_main because we will need to call GlobalInit
add_subdirectory(utils)

# Debug messages - after cmake is fully tested we should remove these prints.
message(STATUS "After core, cpu sources: ")
foreach(tmp ${Caffe2_CPU_SRCS})
message(STATUS "  " ${tmp})
endforeach()

message(STATUS "After core, gpu sources: ")
foreach(tmp ${Caffe2_GPU_SRCS})
message(STATUS "  " ${tmp})
endforeach()

# ---[ List of libraries to link with
set(Caffe2_MAIN_LIBS Caffe_PROTO)

# Compile protobufs.
add_subdirectory(proto)
include_directories(BEFORE ${CMAKE_BINARY_DIR})
list(APPEND Caffe2_MAIN_LIBS Caffe2_PROTO)

# Compile exposed libraries.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
add_library(Caffe2_CPU ${Caffe2_CPU_SRCS})
target_link_libraries(Caffe2_CPU ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS})
install(TARGETS Caffe2_CPU DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
list(APPEND Caffe2_MAIN_LIBS Caffe2_CPU)

# ---[ CUDA library.
if(HAVE_CUDA)
  CUDA_ADD_LIBRARY(Caffe2_GPU ${Caffe2_GPU_SRCS})
  target_link_libraries(Caffe2_GPU ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS})
  install(TARGETS Caffe2_GPU DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  link_directories(${CMAKE_INSTALL_PREFIX}/lib)
  list(APPEND Caffe2_MAIN_LIBS Caffe2_GPU)
endif()

# ---[ Test binaries.
if (BUILD_TEST)
  foreach(test_src ${Caffe2_CPU_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS} gtest_main)
    if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.0)
      target_compile_features(${test_name} PRIVATE cxx_range_for)
    endif()
    install(TARGETS ${test_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
  endforeach()

  if (HAVE_CUDA)
    foreach(test_src ${Caffe2_GPU_TEST_SRCS})
      get_filename_component(test_name ${test_src} NAME_WE)
      add_executable(${test_name} "${test_src}")
      target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS} gtest_main)
      if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.0)
        target_compile_features(${test_name} PRIVATE cxx_range_for)
      endif()
      install(TARGETS ${test_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
    endforeach()
  endif()
endif()

# ---[ Python.
add_library(caffe2_pybind11_state SHARED ${Caffe2_CPU_PYTHON_SRCS})
set_target_properties(caffe2_pybind11_state PROPERTIES PREFIX "")
target_link_libraries(caffe2_pybind11_state ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS})
install(TARGETS caffe2_pybind11_state DESTINATION ${CMAKE_INSTALL_PREFIX}/caffe2/python)

if(HAVE_CUDA)
  add_library(caffe2_pybind11_state_gpu SHARED ${Caffe2_GPU_PYTHON_SRCS})
  set_target_properties(caffe2_pybind11_state_gpu PROPERTIES PREFIX "")
  target_link_libraries(caffe2_pybind11_state_gpu ${Caffe2_MAIN_LIBS} ${Caffe2_DEPENDENCY_LIBS})
  install(TARGETS caffe2_pybind11_state_gpu DESTINATION ${CMAKE_INSTALL_PREFIX}/caffe2/python)
endif()

# ---[ Binaries.
foreach(binary_src ${Caffe2_CPU_BINARY_SRCS})
  get_filename_component(bin_name ${binary_src} NAME_WE)
  add_executable(${bin_name} ${binary_src})
  target_link_libraries(${bin_name} ${Caffe2_MAIN_LIBS} benchmark ${Caffe2_DEPENDENCY_LIBS})
  install(TARGETS ${bin_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/binaries)
endforeach()

if(HAVE_CUDA)
  foreach(binary_src ${Caffe2_GPU_BINARY_SRCS})
    get_filename_component(bin_name ${binary_src} NAME_WE)
    add_executable(${bin_name} ${binary_src})
    target_link_libraries(${bin_name} ${Caffe2_MAIN_LIBS} cnmem benchmark ${Caffe2_DEPENDENCY_LIBS})
    install(TARGETS ${bin_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/binaries)
  endforeach()
endif()

