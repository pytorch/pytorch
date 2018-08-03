#pragma once

#ifdef _WIN32
# if defined(ATen_cpu_EXPORTS) || CAFFE2_BUILD_MAIN_LIB
#  define AT_API __declspec(dllexport)
# else
#  define AT_API __declspec(dllimport)
# endif
#else
# define AT_API
#endif
