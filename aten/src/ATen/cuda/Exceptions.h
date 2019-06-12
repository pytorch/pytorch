#pragma once

#include "c10/util/Exception.h"

#define AT_CUDNN_CHECK(EXPR)                                                     \
  do {                                                                           \
    cudnnStatus_t status = EXPR;                                                 \
    if (status != CUDNN_STATUS_SUCCESS) {                                        \
      if (status == CUDNN_STATUS_NOT_SUPPORTED) {                                \
        AT_ERROR(                                                                \
            "cuDNN error: ",                                                     \
            cudnnGetErrorString(status),                                         \
            ". This error may appear if you passed in a non-contiguous input."); \
      } else {                                                                   \
        AT_ERROR("cuDNN error: ", cudnnGetErrorString(status));                  \
      }                                                                          \
    }                                                                            \
  } while (0)

#define AT_CUDA_CHECK(EXPR)                                \
  do {                                                     \
    cudaError_t __err = EXPR;                              \
    if (__err != cudaSuccess) {                            \
      AT_ERROR("CUDA error: ", cudaGetErrorString(__err)); \
    }                                                      \
  } while (0)
