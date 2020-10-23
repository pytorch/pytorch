cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(pytorch_nnapi)

# Define this to build the NNAPI binding out of tree.
if(PYTORCH_NNAPI_STANDALONE)
  set(CMAKE_CXX_STANDARD 14)
  find_package(Torch REQUIRED)
endif()

set(NNAPI_SRCS
  nnapi_bind.cpp
  nnapi_wrapper.cpp
  nnapi_model_loader.cpp
  )

# Static build on Android so we can just bundle with the benchmarker
# or with PyTorch, but use shared on host so we don't load by accident.
if(ANDROID)
  add_library(pytorch_nnapi STATIC ${NNAPI_SRCS})
else()
  add_library(pytorch_nnapi SHARED ${NNAPI_SRCS})
endif()

target_link_libraries(pytorch_nnapi torch)
