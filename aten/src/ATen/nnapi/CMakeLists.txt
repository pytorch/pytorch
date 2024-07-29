# Define this to build the NNAPI binding out of tree.
if(PYTORCH_NNAPI_STANDALONE)
  cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
  project(pytorch_nnapi)

  set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard whose features are requested to build this target.")
  find_package(Torch REQUIRED)

  set(NNAPI_SRCS
    nnapi_bind.cpp
    nnapi_wrapper.cpp
    nnapi_model_loader.cpp
    )

  add_library(pytorch_nnapi SHARED ${NNAPI_SRCS})
  target_link_libraries(pytorch_nnapi torch)
else()
  # Building within the PyTorch tree.
  file(GLOB ATen_NNAPI_SRCS "*.cpp")
  set(ATen_NNAPI_SRCS ${ATen_NNAPI_SRCS} PARENT_SCOPE)
endif()
