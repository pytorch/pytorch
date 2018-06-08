#pragma once

#ifdef _WIN32
# if defined(ATen_cuda_EXPORTS) || defined(caffe2_gpu_EXPORTS)
#  define AT_CUDA_API __declspec(dllexport)
# else
#  define AT_CUDA_API __declspec(dllimport)
# endif
#else
# define AT_CUDA_API
#endif
