cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(simple-torch-test)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(simple-torch-test simple-torch-test.cpp)
target_include_directories(simple-torch-test PRIVATE  ${TORCH_INCLUDE_DIRS})
target_link_libraries(simple-torch-test "${TORCH_LIBRARIES}")
set_property(TARGET simple-torch-test PROPERTY CXX_STANDARD 17)

find_package(CUDAToolkit 11.8)

target_link_libraries(simple-torch-test CUDA::cudart CUDA::cufft CUDA::cusparse CUDA::cublas CUDA::cusolver)
find_library(CUDNN_LIBRARY NAMES cudnn)
target_link_libraries(simple-torch-test  ${CUDNN_LIBRARY} )
if(MSVC)
  file(GLOB TORCH_DLLS  "$ENV{CUDA_PATH}/bin/cudnn64_8.dll" "$ENV{NVTOOLSEXT_PATH}/bin/x64/*.dll")
  message("dlls to copy "  ${TORCH_DLLS})
  add_custom_command(TARGET simple-torch-test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:simple-torch-test>)
endif(MSVC)
