#pragma once

#if defined(USE_CUDA) && !defined(__HIP_PLATFORM_HCC__)
#define USE_CUDA_NOT_ROCM
#endif
