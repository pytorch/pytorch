#pragma once

#include <ATen/cuda/CUDAContext.h>

// cuSparse Generic API added in CUDA 10.1
// Windows support added in CUDA 11.0
// ROCm is not enabled
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && ((CUSPARSE_VERSION >= 10300) || (CUSPARSE_VERSION >= 11000 && defined(_WIN32)))
#define AT_USE_CUSPARSE_GENERIC_API() 1
#else
#define AT_USE_CUSPARSE_GENERIC_API() 0
#endif
