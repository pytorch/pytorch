#pragma once

// TODO: Move this to something like ATenCoreGeneral.h
#ifdef _WIN32
# if defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
#  define AT_CORE_API __declspec(dllexport)
# else
#  define AT_CORE_API __declspec(dllimport)
# endif
#else
# define AT_CORE_API
#endif

namespace at {

AT_CORE_API int CoreTest();

}
