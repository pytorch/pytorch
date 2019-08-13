#pragma once

#include <ATen/Context.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAException.h>

// See Note [CHECK macro]
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

#define AT_CUDA_CHECK(EXPR) C10_CUDA_CHECK(EXPR)

// For CUDA Driver API
//
// This is here instead of in c10 because NVRTC is loaded dynamically via a stub
// in ATen, and we need to use its nvrtcGetErrorString.
// See NOTE [ USE OF NVRTC AND DRIVER API ].
#ifndef __HIP_PLATFORM_HCC__

#define AT_CUDA_DRIVER_CHECK(EXPR)                                                                               \
  do {                                                                                                           \
    CUresult __err = EXPR;                                                                                       \
    if (__err != CUDA_SUCCESS) {                                                                                 \
      const char* err_str;                                                                                       \
      CUresult get_error_str_err C10_UNUSED = at::globalContext().getNVRTC().cuGetErrorString(__err, &err_str);  \
      if (get_error_str_err != CUDA_SUCCESS) {                                                                   \
        AT_ERROR("CUDA driver error: unknown error");                                                            \
      } else {                                                                                                   \
        AT_ERROR("CUDA driver error: ", err_str);                                                                \
      }                                                                                                          \
    }                                                                                                            \
  } while (0)

#else

#define AT_CUDA_DRIVER_CHECK(EXPR)                                                \
  do {                                                                            \
    CUresult __err = EXPR;                                                        \
    if (__err != CUDA_SUCCESS) {                                                  \
      AT_ERROR("CUDA driver error: ", static_cast<int>(__err));                   \
    }                                                                             \
  } while (0)

#endif

// For CUDA NVRTC
//
// Note: As of CUDA 10, nvrtc error code 7, NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
// incorrectly produces the error string "NVRTC unknown error."
// The following maps it correctly.
//
// This is here instead of in c10 because NVRTC is loaded dynamically via a stub
// in ATen, and we need to use its nvrtcGetErrorString.
// See NOTE [ USE OF NVRTC AND DRIVER API ].
#define AT_CUDA_NVRTC_CHECK(EXPR)                                                                   \
  do {                                                                                              \
    nvrtcResult __err = EXPR;                                                                       \
    if (__err != NVRTC_SUCCESS) {                                                                   \
      if (static_cast<int>(__err) != 7) {                                                           \
        AT_ERROR("CUDA NVRTC error: ", at::globalContext().getNVRTC().nvrtcGetErrorString(__err));  \
      } else {                                                                                      \
        AT_ERROR("CUDA NVRTC error: NVRTC_ERROR_BUILTIN_OPERATION_FAILURE");                        \
      }                                                                                             \
    }                                                                                               \
  } while (0)
