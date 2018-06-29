#pragma once
#include <ATen/Error.h>
#define AT_CUDNN_CHECK(STATUS)                                                 \
  if (STATUS != CUDNN_STATUS_SUCCESS) {                                        \
    if (STATUS == CUDNN_STATUS_NOT_SUPPORTED) {                                \
      AT_ERROR(                                                                \
          "CuDNN error: ",                                                     \
          cudnnGetErrorString(STATUS),                                         \
          ". This error may appear if you passed in a non-contiguous input."); \
    } else {                                                                   \
      AT_ERROR("CuDNN error: ", cudnnGetErrorString(STATUS));                  \
    }                                                                          \
  }
#define AT_CUDA_CHECK(STATUS)                             \
  if (STATUS != cudaSuccess) {                            \
    AT_ERROR("CUDA error: ", cudaGetErrorString(STATUS)); \
  }
