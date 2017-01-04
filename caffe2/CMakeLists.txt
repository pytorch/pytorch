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

# ---[ Protobuf
set(Caffe2_LINK_LIBS Caffe_PROTO)

# Compile protobufs.
add_subdirectory(proto)
include_directories(BEFORE ${CMAKE_BINARY_DIR})
list(APPEND Caffe2_LINK_LIBS Caffe2_PROTO)


# ---[ Add respective subdirectories
# Note: the folders that are being commented out have not been properly
# addressed yet.

# add_subdirectory(binaries)
# add_subdirectory(contrib)
add_subdirectory(core)
# add_subdirectory(cuda_rtc)
add_subdirectory(db)
# add_subdirectory(distributed)
# add_subdirectory(experiments) # note, we may remove this folder at some point
add_subdirectory(image)
# add_subdirectory(mkl)
# add_subdirectory(mpi)
add_subdirectory(operators)
# add_subdirectory(proto) # note, proto is processed above already.
# add_subdirectory(python)
add_subdirectory(queue)
add_subdirectory(sgd)
# add_subdirectory(test) # todo: use caffe2_gtest_main instead of gtest_main because we will need to call GlobalInit
add_subdirectory(utils)



message(STATUS "After core, cpu sources: ")
foreach(tmp ${Caffe2_CPU_SRCS})
message(STATUS "  " ${tmp})
endforeach()

message(STATUS "After core, gpu sources: ")
foreach(tmp ${Caffe2_GPU_SRCS})
message(STATUS "  " ${tmp})
endforeach()

# Compile exposed libraries.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
add_library(Caffe2_CPU ${Caffe2_CPU_SRCS})
target_link_libraries(Caffe2_CPU ${Caffe2_LINK_LIBS} ${Caffe2_LINKER_LIBS})
install(TARGETS Caffe2_CPU DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
list(APPEND Caffe2_LINK_LIBS Caffe2_CPU)

# ---[ CUDA library
if(HAVE_CUDA)
  CUDA_ADD_LIBRARY(Caffe2_GPU ${Caffe2_GPU_SRCS})
  target_link_libraries(Caffe2_GPU ${Caffe2_LINK_LIBS} ${Caffe2_LINKER_LIBS})
  install(TARGETS Caffe2_GPU DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  link_directories(${CMAKE_INSTALL_PREFIX}/lib)
  list(APPEND Caffe2_LINK_LIBS Caffe2_GPU)
  list(APPEND Caffe2_LINK Caffe2_GPU)
endif()

# Compile test binaries.
foreach(test_src ${Caffe2_CPU_TEST_SRCS})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} "${test_src}")
  target_link_libraries(${test_name} ${Caffe2_LINK} gtest_main)
  if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.0)
    target_compile_features(${test_name} PRIVATE cxx_range_for)
  endif()
  install(TARGETS ${test_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
endforeach()

# ---[ Python
add_subdirectory(python)

# ---[ Binaries
set(Caffe2_CPU_BINARY_SRCS
    "convert_caffe_image_db.cc"
    "convert_db.cc"
    "db_throughput.cc"
    "fb_run_plan_mpi.cc"
    "make_cifar_db.cc"
    "make_image_db.cc"
    "make_mnist_db.cc"
    "predictor_verifier.cc"
    "print_registered_core_operators.cc"
    "run_plan.cc"
    "run_plan_mpi.cc"
    "speed_benchmark.cc"
    "split_db.cc"
    "zmq_feeder.cc"
)

set(Caffe2_GPU_BINARY_SRCS
    "core_overhead_benchmark.cc"
    "inspect_gpus.cc"
    "print_core_object_sizes.cc"
)

prepend(Caffe2_CPU_BINARY_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/binaries/" "${Caffe2_CPU_BINARY_SRCS}")
prepend(Caffe2_GPU_BINARY_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/binaries/" "${Caffe2_GPU_BINARY_SRCS}")

# ZMQ src
set(Caffe2_ZMQ_BINARY_SRCS
    "zmq_feeder.cc"
)
# Prune out ZMQ if no support
if(NOT USE_ZMQ)
  message(STATUS "Excluding ZMQ binary")
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/binaries/" "${Caffe2_ZMQ_BINARY_SRCS}")
  exclude(Caffe2_CPU_BINARY_SRCS "${Caffe2_CPU_BINARY_SRCS}" "${tmp}")
  exclude(Caffe2_GPU_BINARY_SRCS "${Caffe2_GPU_BINARY_SRCS}" "${tmp}")
endif()

# MPI-based binaries
set(Caffe2_MPI_BINARY_SRCS
    "fb_run_plan_mpi.cc"
    "run_plan_mpi.cc"
)
if(NOT USE_MPI)
  message(STATUS "Excluding MPI binaries")
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/binaries/" "${Caffe2_MPI_BINARY_SRCS}")
  exclude(Caffe2_CPU_BINARY_SRCS "${Caffe2_CPU_BINARY_SRCS}" "${tmp}")
  exclude(Caffe2_GPU_BINARY_SRCS "${Caffe2_GPU_BINARY_SRCS}" "${tmp}")
endif()

# OpenCV-based binaries
set(Caffe2_OPENCV_BINARY_SRCS
    "convert_encoded_to_raw_leveldb.cc"
    "make_image_db.cc"
)
if(NOT USE_OPENCV)
  message(STATUS "Excluding OPENCV binaries")
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/binaries/" "${Caffe2_OPENCV_BINARY_SRCS}")
  exclude(Caffe2_CPU_BINARY_SRCS "${Caffe2_CPU_BINARY_SRCS}" "${tmp}")
  exclude(Caffe2_GPU_BINARY_SRCS "${Caffe2_GPU_BINARY_SRCS}" "${tmp}")
endif()

foreach(binary_src ${Caffe2_CPU_BINARY_SRCS})
  get_filename_component(bin_name ${binary_src} NAME_WE)
  add_executable(${bin_name} ${binary_src})
  target_link_libraries(${bin_name} ${Caffe2_LINK} benchmark ${Caffe2_LINKER_LIBS})
  install(TARGETS ${bin_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/binaries)
endforeach()

if(HAVE_CUDA)
  foreach(binary_src ${Caffe2_GPU_BINARY_SRCS})
    get_filename_component(bin_name ${binary_src} NAME_WE)
    add_executable(${bin_name} ${binary_src})
    target_link_libraries(${bin_name} ${Caffe2_LINK} cnmem benchmark ${Caffe2_LINKER_LIBS})
    install(TARGETS ${bin_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/binaries)
  endforeach()
endif()
