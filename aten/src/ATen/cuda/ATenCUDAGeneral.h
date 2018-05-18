#pragma once

#ifdef _WIN32
# ifdef ATen_cuda_EXPORTS
#  define AT_CUDA_API __declspec(dllexport)
# else
#  define AT_CUDA_API __declspec(dllimport)
# endif
#else
# define AT_CUDA_API
#endif
