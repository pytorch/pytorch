#pragma once

#ifdef _WIN32
# if defined(ATen_cuda_EXPORTS) || CAFFE2_BUILD_GPU_LIB
#  define AT_CUDA_API __declspec(dllexport)
# else
#  define AT_CUDA_API __declspec(dllimport)
# endif
#else
# define AT_CUDA_API
#endif
