#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef _WIN32
# if defined(ATen_cuda_EXPORTS) || defined(caffe2_gpu_EXPORTS) || defined(CAFFE2_CUDA_BUILD_MAIN_LIB)
#  define AT_CUDA_API __declspec(dllexport)
# else
#  define AT_CUDA_API __declspec(dllimport)
# endif
#elif defined(__GNUC__)
#if defined(ATen_cuda_EXPORTS) || defined(caffe2_gpu_EXPORTS)
#define AT_CUDA_API __attribute__((__visibility__("default")))
#else
#define AT_CUDA_API
#endif
#else
# define AT_CUDA_API
#endif
